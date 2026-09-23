// Dumps the connected MCP tool catalog once.
// Usage: node bench/catalog.ts [config=bench/config.ts] [output]
import { mkdirSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { query, type McpServerStatus } from "@anthropic-ai/claude-agent-sdk";
import type { BenchConfig } from "./types.ts";

function isBenchConfigModule(value: unknown): value is { default: BenchConfig } {
  if (typeof value !== "object" || value === null || !("default" in value)) return false;
  const config = value.default;
  return typeof config === "object" && config !== null && "base" in config && typeof config.base === "function";
}

const configPath = resolve(process.argv[2] ?? "bench/config.ts");
const configName = basename(configPath).match(/^config(?:\.([^.]+))?\.ts$/)?.[1] ?? "demo";
const outputPath = resolve(process.argv[3] ?? `bench/catalog.${configName}.json`);
const loaded: unknown = await import(pathToFileURL(configPath).href);
if (!isBenchConfigModule(loaded)) throw new Error(`${configPath} does not export a valid benchmark config`);

const handle = query({
  prompt: "Reply with OK without calling a tool.",
  options: { ...loaded.default.base(), maxTurns: 1 },
});

try {
  await handle.initializationResult();
  const deadline = Date.now() + 15_000;
  let statuses: McpServerStatus[] = [];
  while (Date.now() < deadline) {
    statuses = await handle.mcpServerStatus();
    if (statuses.length > 0 && statuses.every((server) => server.status !== "pending" && (server.tools?.length ?? 0) > 0)) break;
    await new Promise((resolveWait) => setTimeout(resolveWait, 250));
  }
  if (statuses.length === 0) throw new Error("No MCP servers reported before the 15 second catalog timeout");
  const failed = statuses.filter((server) => server.status !== "connected");
  if (failed.length > 0)
    throw new Error(`MCP servers not connected: ${failed.map((server) => `${server.name}:${server.status}`).join(", ")}`);
  const catalog = statuses.flatMap((server) =>
    (server.tools ?? []).map((tool) => ({
      name: tool.name.startsWith("mcp__") ? tool.name : `mcp__${server.name}__${tool.name}`,
      description: tool.description ?? "",
      annotations: tool.annotations,
    })),
  );
  if (catalog.length === 0) throw new Error("Connected MCP servers reported no tools");
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify(catalog, null, 2)}\n`);
  console.log(`Wrote ${catalog.length} tools to ${outputPath}`);
} finally {
  await handle.return();
}
