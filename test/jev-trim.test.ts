import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { z } from "zod";
import { chunkOutput, createSpillServer, trimToolOutput } from "../src/jev-trim.ts";
import { choiceClient, failingClient } from "./helpers.ts";

const longText = Array.from({ length: 600 }, (_, index) => `line ${index}: ${"x".repeat(20)}`).join("\n");

test("trim preserves MCP content-array shape and keeps the last chunk", async () => {
  const chunks = chunkOutput(longText);
  assert.ok(chunks.length >= 3);
  const probabilities = Object.fromEntries([
    ...chunks.map((chunk, index) => [chunk.id, index === 0 ? 0.91 : 0.001]),
    ["none_of_these", 0.001],
  ]);
  const result = await trimToolOutput({
    task: "Find the first lines",
    sessionId: "session-1",
    toolName: "mcp__logs__read",
    toolInput: { query: "error" },
    toolOutput: [{ type: "text", text: longText }],
    toolUseId: "tool-1",
    minChars: 100,
    client: choiceClient("c0", probabilities),
  });
  assert.equal(result.kind, "trimmed");
  if (result.kind !== "trimmed") return;
  const parsed = z.tuple([z.object({ type: z.literal("text"), text: z.string() })]).parse(result.output);
  assert.match(parsed[0].text, /line 0:/);
  assert.match(parsed[0].text, /line 599:/);
  assert.match(parsed[0].text, /elided/);
  assert.ok(result.trimmedChars < result.originalChars);
});

test("trim preserves the Bash object shape", async () => {
  const chunks = chunkOutput(longText);
  const probabilities = Object.fromEntries([
    ...chunks.map((chunk, index) => [chunk.id, index === 0 ? 0.95 : 0.001]),
    ["none_of_these", 0.001],
  ]);
  const result = await trimToolOutput({
    task: "Read the start",
    sessionId: "session-1",
    toolName: "Bash",
    toolInput: { command: "long-command" },
    toolOutput: { stdout: longText, stderr: "warning", interrupted: false },
    toolUseId: "tool-2",
    minChars: 100,
    client: choiceClient("c0", probabilities),
  });
  assert.equal(result.kind, "trimmed");
  if (result.kind !== "trimmed") return;
  const parsed = z.object({ stdout: z.string(), stderr: z.string(), interrupted: z.boolean() }).parse(result.output);
  assert.equal(parsed.stderr, "warning");
  assert.equal(parsed.interrupted, false);
  assert.match(parsed.stdout, /elided/);
});

test("trim passes through on the escape choice", async () => {
  const chunks = chunkOutput(longText);
  const probabilities = Object.fromEntries([
    ...chunks.map((chunk) => [chunk.id, 0.001]),
    ["none_of_these", 0.99],
  ]);
  const result = await trimToolOutput({
    task: "Keep all output",
    sessionId: "session-1",
    toolName: "mcp__logs__read",
    toolInput: {},
    toolOutput: [{ type: "text", text: longText }],
    toolUseId: "tool-3",
    minChars: 100,
    client: choiceClient("none_of_these", probabilities),
  });
  assert.deepEqual(result, { kind: "pass", reason: "escape" });
});

test("trim names a spill id and read_spill returns capped matching chunks", async () => {
  const directory = mkdtempSync(join(tmpdir(), "jev-spill-test-"));
  try {
    const chunks = chunkOutput(longText);
    const probabilities = Object.fromEntries([
      ...chunks.map((chunk, index) => [chunk.id, index === 0 ? 0.95 : 0.001]),
      ["none_of_these", 0.001],
    ]);
    const result = await trimToolOutput({
      task: "Find the first lines",
      sessionId: "session-1",
      toolName: "mcp__logs__read",
      toolInput: {},
      toolOutput: [{ type: "text", text: longText }],
      toolUseId: "tool-spill",
      minChars: 100,
      spillDirectory: directory,
      client: choiceClient("c0", probabilities),
    });
    assert.equal(result.kind, "trimmed");
    if (result.kind !== "trimmed" || !result.spillId) return;
    assert.match(JSON.stringify(result.output), new RegExp(result.spillId));
    assert.match(JSON.stringify(result.output), /read_spill/);

    const server = createSpillServer(directory);
    const client = new Client({ name: "spill-test", version: "1.0.0" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.instance.connect(serverTransport);
    await client.connect(clientTransport);
    const response = await client.callTool({ name: "read_spill", arguments: { id: result.spillId, query: "line 300" } });
    const parsed = z.object({ content: z.array(z.object({ type: z.string(), text: z.string().optional() }).loose()) }).parse(response);
    const text = parsed.content.find((content) => content.type === "text")?.text;
    assert.ok(text);
    assert.match(text, /line 300:/);
    assert.ok(text.length <= 8_000);
    await client.close();
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("trim never classifies error results", async () => {
  const result = await trimToolOutput({
    task: "Read logs",
    sessionId: "session-1",
    toolName: "mcp__logs__read",
    toolInput: {},
    toolOutput: { content: [{ type: "text", text: longText }], isError: true },
    toolUseId: "tool-error",
    minChars: 100,
    client: failingClient(),
  });
  assert.deepEqual(result, { kind: "pass", reason: "error" });
});
