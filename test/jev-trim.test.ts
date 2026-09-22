import assert from "node:assert/strict";
import test from "node:test";
import { z } from "zod";
import { chunkOutput, trimToolOutput } from "../src/jev-trim.ts";
import { choiceClient } from "./helpers.ts";

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
