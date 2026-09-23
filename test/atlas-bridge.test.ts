import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";
import { z } from "zod";
import { AtlasGateway } from "../bench/atlas/bridge.ts";

test("Atlas bridge filters tools, shortens long names, and reverses names on calls", async () => {
  const inputSchema = { type: "object", properties: { path: { type: "string", default: null } } };
  const longName = `filesystem_${"read_deep_directory_".repeat(4)}`;
  let calledName = "";
  const server = createServer((request, response) => {
    if (request.url === "/list-tools") {
      response.setHeader("content-type", "application/json");
      response.end(
        JSON.stringify([
          {
            name: longName,
            title: null,
            description: "Read files",
            inputSchema,
            outputSchema: { type: "object", properties: { result: { type: "string" } } },
            annotations: { title: null, readOnlyHint: true, destructiveHint: null },
            _meta: null,
          },
          { name: "github_search", title: null, description: null, inputSchema, annotations: null, _meta: null },
        ]),
      );
      return;
    }
    if (request.url === "/call-tool") {
      let body = "";
      request.setEncoding("utf8");
      request.on("data", (chunk: string) => {
        body += chunk;
      });
      request.on("end", () => {
        const parsed = z
          .object({ tool_name: z.string(), tool_args: z.record(z.string(), z.unknown()), use_cache: z.boolean() })
          .parse(JSON.parse(body));
        calledName = parsed.tool_name;
        response.setHeader("content-type", "application/json");
        if (parsed.tool_args.path === "/missing") {
          response.statusCode = 500;
          response.end(JSON.stringify({ detail: "Failed to call tool: file not found" }));
          return;
        }
        response.end(JSON.stringify([{ type: "text", text: "file contents", annotations: null, _meta: null }]));
      });
      return;
    }
    response.statusCode = 404;
    response.end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Test server did not bind to a TCP port");

  try {
    const gateway = new AtlasGateway({
      baseUrl: `http://127.0.0.1:${address.port}`,
      enabledTools: new Set([longName]),
    });
    const tools = await gateway.listTools();
    assert.equal(tools.length, 1);
    assert.ok(tools[0].name.length <= 64);
    assert.deepEqual(tools[0].inputSchema, inputSchema, "input schemas pass through unchanged");
    assert.deepEqual(tools[0].annotations, { readOnlyHint: true });
    assert.equal(tools[0].outputSchema, undefined, "calls return content only, so no output schema is advertised");
    assert.deepEqual(await gateway.callTool(tools[0].name, { path: "/data/file" }), {
      content: [{ type: "text", text: "file contents" }],
    });
    assert.equal(calledName, longName);
    assert.deepEqual(await gateway.callTool(tools[0].name, { path: "/missing" }), {
      content: [{ type: "text", text: "Failed to call tool: file not found" }],
      isError: true,
    });
  } finally {
    await new Promise<void>((resolveClose, reject) =>
      server.close((error) => (error ? reject(error) : resolveClose())),
    );
  }
});
