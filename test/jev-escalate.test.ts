import assert from "node:assert/strict";
import test from "node:test";
import type { PostToolBatchHookInput, PostToolUseFailureHookInput } from "@anthropic-ai/claude-agent-sdk";
import { jevEscalation, type EscalationRecord } from "../src/jev-escalate.ts";
import { answersClient, failingClient } from "./helpers.ts";

const hookOptions = { signal: new AbortController().signal };

function failure(id: string, name = "Read", input: unknown = { path: "missing" }): PostToolUseFailureHookInput {
  return {
    hook_event_name: "PostToolUseFailure",
    session_id: "session-1",
    transcript_path: "/tmp/transcript.jsonl",
    cwd: "/tmp",
    tool_name: name,
    tool_input: input,
    tool_use_id: id,
    error: "not found",
  };
}

function batch(
  calls: { id: string; name?: string; input?: unknown; response?: unknown }[],
): PostToolBatchHookInput {
  return {
    hook_event_name: "PostToolBatch",
    session_id: "session-1",
    transcript_path: "/tmp/transcript.jsonl",
    cwd: "/tmp",
    tool_calls: calls.map((call) => ({
      tool_name: call.name ?? "Read",
      tool_input: call.input ?? { path: "missing" },
      tool_use_id: call.id,
      tool_response: call.response,
    })),
  };
}

function progressClient(blocker: "none" | "deeper_reasoning" | "environment" | "needs_user") {
  return answersClient({
    stuck: { type: "noul", noul: 0.93 },
    harder: { type: "noul", noul: 0.2 },
    blocker: {
      type: "choice",
      choice: blocker,
      confidence: 0.9,
      probabilities: { none: 0.02, deeper_reasoning: 0.03, environment: 0.03, needs_user: 0.02, [blocker]: 0.9 },
    },
  });
}

test("escalation fires once after code and Jev signals agree", async () => {
  const efforts: string[] = [];
  const hook = jevEscalation({
    session: {
      setModel: async () => {},
      applyFlagSettings: async (settings) => {
        if (settings.effortLevel) efforts.push(settings.effortLevel);
      },
    },
    action: { kind: "effort", level: "high" },
    client: progressClient("deeper_reasoning"),
    onEscalate: (record) => records.push(record),
  });
  const records: EscalationRecord[] = [];

  await hook(failure("tool-1"), "tool-1", hookOptions);
  await hook(failure("tool-2"), "tool-2", hookOptions);
  await hook(batch([{ id: "tool-1" }, { id: "tool-2" }]), undefined, hookOptions);
  await hook(batch([{ id: "tool-3", response: "ok" }]), undefined, hookOptions);
  assert.deepEqual(efforts, ["high"]);
  assert.deepEqual(records, [
    {
      batch: 1,
      action: { kind: "effort", level: "high" },
      signals: { failures: true, repeated: true, longRunning: false },
    },
  ]);
});

test("environment and needs-user blockers never escalate", async () => {
  for (const blocker of ["environment", "needs_user"] as const) {
    let calls = 0;
    const hook = jevEscalation({
      session: {
        setModel: async () => {
          calls += 1;
        },
        applyFlagSettings: async () => {
          calls += 1;
        },
      },
      action: { kind: "model", model: "opus" },
      client: progressClient(blocker),
    });
    await hook(failure("tool-1"), "tool-1", hookOptions);
    await hook(failure("tool-2"), "tool-2", hookOptions);
    await hook(batch([{ id: "tool-1" }, { id: "tool-2" }]), undefined, hookOptions);
    assert.equal(calls, 0);
  }
});

test("escalation skips Jev without a code signal and passes through when Jev fails", async () => {
  let calls = 0;
  const session = {
    setModel: async () => {
      calls += 1;
    },
    applyFlagSettings: async () => {
      calls += 1;
    },
  };
  const noSignal = jevEscalation({ session, action: { kind: "model", model: "opus" }, client: failingClient() });
  assert.deepEqual(
    await noSignal(batch([{ id: "tool-1", input: { path: "a" }, response: "ok" }]), undefined, hookOptions),
    {},
  );

  const failed = jevEscalation({ session, action: { kind: "model", model: "opus" }, client: failingClient() });
  await failed(failure("tool-2"), "tool-2", hookOptions);
  await failed(failure("tool-3"), "tool-3", hookOptions);
  assert.deepEqual(await failed(batch([{ id: "tool-2" }, { id: "tool-3" }]), undefined, hookOptions), {});
  assert.equal(calls, 0);
});
