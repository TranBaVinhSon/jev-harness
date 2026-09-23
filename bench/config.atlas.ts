import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { z } from "zod";
import { beforeAfterArms } from "./arms.ts";
import { AtlasGateway, atlasToolName } from "./atlas/bridge.ts";
import { AtlasPool } from "./atlas/sandbox.ts";
import type { BenchConfig, Task } from "./types.ts";

const taskSchema = z.object({
  id: z.string(),
  prompt: z.string(),
  enabledTools: z.array(z.string()),
  claims: z.array(z.string()),
  servers: z.array(z.string()),
});

function loadAtlasTasks(path: string): Task[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line, index) => {
      try {
        const task = taskSchema.parse(JSON.parse(line));
        return { ...task, check: () => false };
      } catch (error) {
        throw new Error(`${path}:${index + 1}: ${String(error)}`);
      }
    });
}

const defaultServers = [
  "arxiv",
  "calculator",
  "cli-mcp-server",
  "clinicaltrialsgov-mcp-server",
  "context7",
  "ddg-search",
  "desktop-commander",
  "fetch",
  "filesystem",
  "git",
  "mcp-code-executor",
  "mcp-server-code-runner",
  "memory",
  "met-museum",
  "open-library",
  "osm-mcp-server",
  "pubmed",
  "weather",
  "whois",
  "wikipedia",
];
const atlasServers = (process.env.ATLAS_SERVERS?.split(",").filter(Boolean) ?? defaultServers).sort();
const available = new Set(atlasServers);
const requestedTasks = new Set(process.env.TASKS?.split(",").filter(Boolean) ?? []);
const tasks = loadAtlasTasks(resolve("bench/atlas/tasks.jsonl")).filter(
  (task) =>
    task.servers?.every((server) => available.has(server)) &&
    (requestedTasks.size === 0 || requestedTasks.has(task.id)),
);

// Filled once the pool is up: bridged tool name -> the description Claude's own ToolSearch sees.
const descriptions = new Map<string, string>();
const arms = beforeAfterArms({
  catalog: ({ task }) =>
    (task.enabledTools ?? []).flatMap((name) => {
      const description = descriptions.get(atlasToolName(name));
      return description === undefined ? [] : [{ name: `mcp__atlas__${atlasToolName(name)}`, description }];
    }),
  serverOf: (name) => name.split("__")[2]?.split("_")[0] ?? name,
  trimMatcher: "mcp__atlas__.*",
  spillDirectory: resolve(".context/atlas-spills"),
});

const concurrency = Number(process.env.CONCURRENCY ?? 4);
const firstPort = Number(process.env.ATLAS_PORT ?? 1984);
const pool = await AtlasPool.start({
  size: concurrency,
  firstPort,
  servers: atlasServers,
  envFile: process.env.ATLAS_ENV_FILE,
});
for (const tool of await new AtlasGateway({ baseUrl: `http://127.0.0.1:${firstPort}` }).listTools())
  descriptions.set(tool.name, tool.description ?? "");

const config: BenchConfig = {
  tasks,
  arms,
  repeats: Number(process.env.REPEATS ?? 3),
  concurrency,
  lease: (task) => pool.lease(task),
  teardown: () => pool.close(),
  base: () => ({
    maxTurns: 40,
    systemPrompt: "Answer the request using the available tools. End with a final answer grounded in the tool results.",
    tools: ["ToolSearch"],
    allowedTools: ["mcp__atlas__*", "mcp__jev__read_spill"],
    permissionMode: "dontAsk",
    settingSources: [],
    strictMcpConfig: true,
    env: {
      ...process.env,
      ENABLE_CLAUDEAI_MCP_SERVERS: "false",
      CLAUDE_CODE_MCP_STARTUP_WAIT_MS: process.env.CLAUDE_CODE_MCP_STARTUP_WAIT_MS ?? "180000",
    },
  }),
};

export default config;
