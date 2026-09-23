import { execFile } from "node:child_process";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import type { Options } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type { Task, Trace } from "../types.ts";

const exec = promisify(execFile);
const healthSchema = z.object({ status: z.literal("health_and_client_connection_ok") });
const TEMPLATE_PATH = "/agent-environment/src/agent_environment/mcp_server_template.json";
const templateSchema = z
  .object({ mcpServers: z.record(z.string(), z.object({ command: z.string(), args: z.array(z.string()).optional() }).loose()) })
  .loose();
const STATEFUL_SERVERS = new Set([
  "filesystem",
  "memory",
  "git",
  "mongodb",
  "desktop-commander",
  "cli-mcp-server",
  "mcp-code-executor",
  "mcp-server-code-runner",
  "e2b-server",
]);

type Container = { name: string; port: number };

export type AtlasPoolOptions = {
  size: number;
  firstPort?: number;
  servers?: string[];
  envFile?: string;
  image?: string;
};

async function docker(...args: string[]): Promise<string> {
  const result = await exec("docker", args, { maxBuffer: 10 * 1024 * 1024 });
  return result.stdout.trim();
}

// The image pins each Python MCP server but not its `mcp` dependency, so at container start uvx
// resolves mcp 2.x (released 2026-07-28) and seven keyless servers crash on import. A global uv
// config would also re-resolve the gateway's own environment, so pin mcp per uvx server instead.
async function pinnedTemplate(image: string): Promise<string> {
  const template = templateSchema.parse(JSON.parse(await docker("run", "--rm", "--entrypoint", "cat", image, TEMPLATE_PATH)));
  for (const server of Object.values(template.mcpServers)) {
    if (server.command === "uvx") server.args = ["--with", "mcp<2", ...(server.args ?? [])];
  }
  const path = join(mkdtempSync(join(tmpdir(), "jev-atlas-")), "mcp_server_template.json");
  writeFileSync(path, JSON.stringify(template, null, 2));
  return path;
}

async function waitForHealth(port: number): Promise<void> {
  const deadline = Date.now() + 600_000;
  let lastError = "container did not answer";
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`);
      const body = await response.text();
      if (response.ok && healthSchema.safeParse(JSON.parse(body)).success) return;
      lastError = `HTTP ${response.status}: ${body.slice(0, 300)}`;
    } catch (error) {
      lastError = String(error);
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 500));
  }
  throw new Error(`Atlas container on port ${port} failed its health check: ${lastError}`);
}

export class AtlasPool {
  readonly #containers: Container[];
  readonly #available: Container[];
  readonly #waiters: ((container: Container) => void)[] = [];
  #closed = false;

  private constructor(containers: Container[]) {
    this.#containers = containers;
    this.#available = [...containers];
  }

  static async start(options: AtlasPoolOptions): Promise<AtlasPool> {
    const envFile = resolve(options.envFile ?? "bench/atlas/.env");
    if (!existsSync(envFile)) throw new Error(`Atlas env file does not exist: ${envFile}`);
    const image = options.image ?? "ghcr.io/scaleapi/mcp-atlas:1.2.7";
    const firstPort = options.firstPort ?? 1984;
    const template = await pinnedTemplate(image);
    const containers: Container[] = [];
    try {
      await Promise.all(
        Array.from({ length: options.size }, async (_, index) => {
          const container = { name: `jev-atlas-${process.pid}-${index}`, port: firstPort + index };
          await docker(
            "run",
            "-d",
            "--name",
            container.name,
            "-p",
            `${container.port}:1984`,
            "--env-file",
            envFile,
            "-v",
            `${template}:${TEMPLATE_PATH}:ro`,
            "-e",
            `ENABLED_SERVERS=${(options.servers ?? []).join(",")}`,
            image,
          );
          containers.push(container);
          await waitForHealth(container.port);
        }),
      );
    } catch (error) {
      await Promise.all(containers.map((container) => docker("rm", "-f", container.name).catch(() => "")));
      throw error;
    }
    const pool = new AtlasPool(containers);
    process.once("exit", () => {
      for (const container of containers) {
        try {
          execFile("docker", ["rm", "-f", container.name]);
        } catch {
          // The daemon may already be unavailable during process teardown.
        }
      }
    });
    return pool;
  }

  async #acquire(): Promise<Container> {
    if (this.#closed) throw new Error("Atlas pool is closed");
    const available = this.#available.pop();
    if (available) return available;
    return new Promise((resolveContainer) => this.#waiters.push(resolveContainer));
  }

  #return(container: Container): void {
    const waiter = this.#waiters.shift();
    if (waiter) waiter(container);
    else this.#available.push(container);
  }

  async lease(task: Task): Promise<{ options: Options; release: (trace: Trace) => Promise<void> }> {
    const container = await this.#acquire();
    let released = false;
    return {
      options: {
        mcpServers: {
          atlas: {
            type: "stdio",
            command: process.execPath,
            args: [resolve("bench/atlas/bridge.ts")],
            env: {
              ATLAS_URL: `http://127.0.0.1:${container.port}`,
              ATLAS_ENABLED_TOOLS: JSON.stringify(task.enabledTools ?? []),
              ATLAS_MODE: process.env.ATLAS_MODE ?? "",
            },
          },
        },
      },
      release: async () => {
        if (released) return;
        released = true;
        try {
          if (task.servers?.some((server) => STATEFUL_SERVERS.has(server))) {
            await docker("restart", container.name);
            await waitForHealth(container.port);
          }
        } finally {
          this.#return(container);
        }
      },
    };
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    await Promise.all(this.#containers.map((container) => docker("rm", "-f", container.name).catch(() => "")));
  }
}
