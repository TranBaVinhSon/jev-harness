import assert from "node:assert/strict";
import test from "node:test";
import { choice, noul } from "@typesafe-ai/sdk";
import { askMany, type JevDecision } from "../src/jev.ts";
import { answersClient } from "./helpers.ts";

test("askMany returns every answer from one request and records each decision", async () => {
  const decisions: JevDecision[] = [];
  const answers = await askMany({
    hook: "progress",
    sessionId: "session-1",
    state: { batch: 3 },
    questions: {
      stuck: noul("Is the agent stuck?"),
      blocker: choice("What blocks progress?", { none: null, environment: null }),
    },
    client: answersClient(
      {
        stuck: { type: "noul", noul: 0.91 },
        blocker: {
          type: "choice",
          choice: "environment",
          confidence: 0.88,
          probabilities: { none: 0.12, environment: 0.88 },
        },
      },
      17,
    ),
    sink: (decision) => {
      decisions.push(decision);
    },
  });

  assert.equal(answers.stuck.noul, 0.91);
  assert.equal(answers.blocker.choice, "environment");
  assert.deepEqual(
    decisions.map((decision) => decision.hook),
    ["progress.stuck", "progress.blocker"],
  );
  assert.equal(decisions.reduce((total, decision) => total + decision.inputTokens, 0), 17);
});
