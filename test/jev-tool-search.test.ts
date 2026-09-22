import assert from "node:assert/strict";
import test from "node:test";
import type { PreToolUseHookInput } from "@anthropic-ai/claude-agent-sdk";
import { jevToolSearch, type CatalogTool } from "../src/jev-tool-search.ts";
import type { JevDecision } from "../src/jev.ts";
import { choiceClient, failingClient, sequenceChoiceClient, timeoutClient } from "./helpers.ts";

const catalog: CatalogTool[] = [
  { name: "mcp__billing__tax", description: "invoice tax region" },
  { name: "mcp__crm__owner", description: "account owner" },
];

function input(query: string): PreToolUseHookInput {
  return {
    hook_event_name: "PreToolUse",
    session_id: "session-1",
    transcript_path: "/tmp/transcript.jsonl",
    cwd: "/tmp",
    tool_name: "ToolSearch",
    tool_input: { query, max_results: 1 },
    tool_use_id: "tool-1",
  };
}

const hookOptions = { signal: new AbortController().signal };

test("tool search rewrites a keyword query and records the choice", async () => {
  const decisions: JevDecision[] = [];
  const hook = jevToolSearch({
    catalog: () => catalog,
    request: () => "Find the tax region",
    client: choiceClient("mcp__billing__tax", {
      mcp__billing__tax: 0.92,
      mcp__crm__owner: 0.06,
      none_of_these: 0.02,
    }),
    sink: (decision) => {
      decisions.push(decision);
    },
  });
  const result = await hook(input("tax region"), "tool-1", hookOptions);
  assert.ok("hookSpecificOutput" in result && result.hookSpecificOutput);
  assert.equal(result.hookSpecificOutput.hookEventName, "PreToolUse");
  assert.deepEqual(result.hookSpecificOutput.updatedInput, {
    query: "select:mcp__billing__tax,mcp__crm__owner",
    max_results: 2,
  });
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0].result.kind, "choice");
  assert.equal(decisions[0].inputTokens, 17);
});

test("tool search narrows a 300-tool catalog by server before ranking tools", async () => {
  const servers = ["billing", "crm", "hr", "support", "jira", "slack", "gh", "pg", "s3", "k8s", "pd", "mail"];
  const largeCatalog: CatalogTool[] = Array.from({ length: 300 }, (_, index) => ({
    name: `mcp__${servers[index % servers.length]}__tool_${index}`,
    description: `Tool ${index}`,
  }));
  largeCatalog[12] = { name: "mcp__billing__get_tax_region", description: "invoice tax region" };
  const serverProbabilities = Object.fromEntries([
    ...servers.map((server) => [server, server === "billing" ? 0.6 : server === "crm" ? 0.2 : server === "hr" ? 0.15 : 0]),
    ["none_of_these", 0.05],
  ]);
  const narrowed = largeCatalog.filter((tool) => ["billing", "crm", "hr"].some((server) => tool.name.startsWith(`mcp__${server}__`)));
  const toolProbabilities = Object.fromEntries([
    ...narrowed.map((tool) => [tool.name, tool.name.endsWith("get_tax_region") ? 0.9 : 0.001]),
    ["none_of_these", 0.001],
  ]);
  const decisions: JevDecision[] = [];
  const hook = jevToolSearch({
    catalog: () => largeCatalog,
    request: () => "Find the tax region",
    client: sequenceChoiceClient([
      { answer: "billing", probabilities: serverProbabilities },
      { answer: "mcp__billing__get_tax_region", probabilities: toolProbabilities },
    ]),
    sink: (decision) => {
      decisions.push(decision);
    },
  });
  const result = await hook(input("tax region"), "tool-1", hookOptions);
  assert.ok("hookSpecificOutput" in result && result.hookSpecificOutput);
  assert.equal(result.hookSpecificOutput.hookEventName, "PreToolUse");
  assert.equal(decisions.length, 2);
  assert.equal(decisions[0].details?.candidateCount, 12);
  assert.equal(decisions[1].details?.candidateCount, 75);
});

test("tool search passes through on the escape choice", async () => {
  const hook = jevToolSearch({
    catalog: () => catalog,
    request: () => "Use a built-in tool",
    client: choiceClient("none_of_these", {
      mcp__billing__tax: 0.05,
      mcp__crm__owner: 0.05,
      none_of_these: 0.9,
    }),
  });
  assert.deepEqual(await hook(input("files"), "tool-1", hookOptions), {});
});

test("tool search passes through when Jev fails", async () => {
  const decisions: JevDecision[] = [];
  const hook = jevToolSearch({
    catalog: () => catalog,
    request: () => "Find tax",
    client: failingClient(),
    sink: (decision) => {
      decisions.push(decision);
    },
  });
  assert.deepEqual(await hook(input("tax"), "tool-1", hookOptions), {});
  assert.equal(decisions[0].result.kind, "error");
});

test("tool search passes through when Jev times out", async () => {
  const hook = jevToolSearch({
    catalog: () => catalog,
    request: () => "Find tax",
    client: timeoutClient(),
  });
  assert.deepEqual(await hook(input("tax"), "tool-1", hookOptions), {});
});
