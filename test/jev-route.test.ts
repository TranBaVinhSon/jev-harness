import assert from "node:assert/strict";
import test from "node:test";
import type { PreToolUseHookInput } from "@anthropic-ai/claude-agent-sdk";
import { jevSubagentRoute, routeSession, type RoutePolicy } from "../src/jev-route.ts";
import { choiceClient, failingClient } from "./helpers.ts";

const policy: RoutePolicy = {
  lookup: { model: "haiku", effort: "low" },
  specifiedChange: { model: "sonnet", effort: "medium" },
  strong: { model: "opus", effort: "high" },
};

const routeProbabilities = {
  lookup: 0.92,
  specified_change: 0.03,
  open_reasoning: 0.03,
  unclear: 0.02,
};

test("session routing selects a cheap model for a confident lookup", async () => {
  const route = await routeSession({
    task: "Find a symbol",
    sessionId: "run-1",
    policy,
    client: choiceClient("lookup", routeProbabilities),
  });
  assert.deepEqual(route.options, { model: "haiku", effort: "low" });
  assert.equal(route.routed, true);
});

test("session routing keeps the strong model below the confidence threshold", async () => {
  const route = await routeSession({
    task: "Maybe change something",
    sessionId: "run-2",
    policy,
    client: choiceClient("specified_change", routeProbabilities, 0.6),
  });
  assert.deepEqual(route.options, { model: "opus", effort: "high" });
  assert.equal(route.routed, false);
});

test("session routing keeps the strong model when Jev fails", async () => {
  const route = await routeSession({
    task: "Find a symbol",
    sessionId: "run-3",
    policy,
    client: failingClient(),
  });
  assert.deepEqual(route.options, { model: "opus", effort: "high" });
  assert.equal(route.routed, false);
});

function agentInput(toolInput: unknown): PreToolUseHookInput {
  return {
    hook_event_name: "PreToolUse",
    session_id: "session-1",
    transcript_path: "/tmp/transcript.jsonl",
    cwd: "/tmp",
    tool_name: "Agent",
    tool_input: toolInput,
    tool_use_id: "tool-1",
  };
}

test("subagent routing fills an unset model and respects explicit or fork models", async () => {
  const hook = jevSubagentRoute({
    request: () => "Main task",
    lookupModel: "haiku",
    specifiedChangeModel: "sonnet",
    client: choiceClient("lookup", routeProbabilities),
  });
  const hookOptions = { signal: new AbortController().signal };
  const routed = await hook(agentInput({ prompt: "Find the file" }), "tool-1", hookOptions);
  assert.ok("hookSpecificOutput" in routed && routed.hookSpecificOutput);
  assert.equal(routed.hookSpecificOutput.hookEventName, "PreToolUse");
  if (routed.hookSpecificOutput.hookEventName !== "PreToolUse") return;
  assert.deepEqual(routed.hookSpecificOutput.updatedInput, { prompt: "Find the file", model: "haiku" });
  assert.deepEqual(await hook(agentInput({ prompt: "Find it", model: "opus" }), "tool-2", hookOptions), {});
  assert.deepEqual(await hook(agentInput({ prompt: "Find it", subagent_type: "fork" }), "tool-3", hookOptions), {});
});
