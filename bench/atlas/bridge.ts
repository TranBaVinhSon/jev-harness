import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ContentBlockSchema,
  ListToolsRequestSchema,
  ToolAnnotationsSchema,
  ToolSchema,
  type CallToolResult,
  type Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

const enabledToolsSchema = z.array(z.string());
const withoutNulls = (value: unknown) =>
  typeof value === "object" && value !== null ? Object.fromEntries(Object.entries(value).filter(([, field]) => field !== null)) : value;
// Atlas serializes unset optional fields as null, which MCP's schemas reject. Its /call-tool
// returns content only, so outputSchema is dropped: clients reject results without structuredContent.
const atlasToolSchema = z.object({
  name: z.string(),
  description: z.string().nullish(),
  inputSchema: ToolSchema.shape.inputSchema,
  annotations: z.preprocess(withoutNulls, ToolAnnotationsSchema).nullish(),
});
const atlasContentSchema = z.array(z.preprocess(withoutNulls, ContentBlockSchema));
// Atlas reports a failed tool execution as HTTP 500 with a detail message.
const toolErrorSchema = z.object({ detail: z.string() });

export function atlasToolName(name: string): string {
  if (name.length <= 64) return name;
  const suffix = createHash("sha256").update(name).digest("hex").slice(0, 15);
  return `${name.slice(0, 48)}_${suffix}`;
}

export type AtlasGatewayOptions = {
  baseUrl: string;
  enabledTools?: ReadonlySet<string>;
  fetch?: typeof fetch;
};

export class AtlasGateway {
  readonly #baseUrl: string;
  readonly #enabledTools: ReadonlySet<string> | undefined;
  readonly #fetch: typeof fetch;
  readonly #originalNames = new Map<string, string>();

  constructor(options: AtlasGatewayOptions) {
    this.#baseUrl = options.baseUrl.replace(/\/$/, "");
    this.#enabledTools = options.enabledTools;
    this.#fetch = options.fetch ?? fetch;
  }

  async #json(url: string, init: RequestInit): Promise<unknown> {
    const response = await this.#fetch(`${this.#baseUrl}${url}`, init);
    if (!response.ok) throw new Error(`Atlas ${url} failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
    return response.json();
  }

  async listTools(): Promise<Tool[]> {
    const tools = z.array(atlasToolSchema).parse(await this.#json("/list-tools", { method: "POST" }));
    return tools
      .filter((tool) => !this.#enabledTools || this.#enabledTools.has(tool.name))
      .map((tool) => {
        const shortName = atlasToolName(tool.name);
        this.#originalNames.set(shortName, tool.name);
        return {
          name: shortName,
          description: tool.description ?? undefined,
          inputSchema: tool.inputSchema,
          annotations: tool.annotations ?? undefined,
        };
      });
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<CallToolResult> {
    let original = this.#originalNames.get(name);
    if (!original) {
      await this.listTools();
      original = this.#originalNames.get(name);
    }
    if (!original) throw new Error(`Unknown or disabled Atlas tool: ${name}`);
    const response = await this.#fetch(`${this.#baseUrl}/call-tool`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tool_name: original, tool_args: args, use_cache: true }),
    });
    const body = await response.text();
    if (response.ok) return { content: atlasContentSchema.parse(JSON.parse(body)) };
    const failure = response.status === 500 ? toolErrorSchema.safeParse(JSON.parse(body)) : undefined;
    if (failure?.success) return { content: [{ type: "text", text: failure.data.detail }], isError: true };
    throw new Error(`Atlas /call-tool failed (${response.status}): ${body.slice(0, 500)}`);
  }
}

export function createAtlasBridge(options: AtlasGatewayOptions): Server {
  const gateway = new AtlasGateway(options);
  const server = new Server({ name: "atlas", version: "1.0.0" }, { capabilities: { tools: {} } });
  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: await gateway.listTools() }));
  server.setRequestHandler(CallToolRequestSchema, (request) =>
    gateway.callTool(request.params.name, request.params.arguments ?? {}),
  );
  return server;
}

function enabledToolsFromEnv(): ReadonlySet<string> | undefined {
  if (process.env.ATLAS_MODE === "full") return undefined;
  const parsed = enabledToolsSchema.parse(JSON.parse(process.env.ATLAS_ENABLED_TOOLS ?? "[]"));
  return new Set(parsed);
}

export async function runAtlasBridge(): Promise<void> {
  const server = createAtlasBridge({
    baseUrl: process.env.ATLAS_URL ?? "http://127.0.0.1:1984",
    enabledTools: enabledToolsFromEnv(),
  });
  await server.connect(new StdioServerTransport());
}

const entry = process.argv[1];
if (entry && import.meta.url === pathToFileURL(entry).href) await runAtlasBridge();
