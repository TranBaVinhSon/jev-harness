import assert from "node:assert/strict";
import test from "node:test";
import { applyCoverage, parseCsv, runRowSchema, type RunRow } from "../bench/atlas/grade.ts";

test("Atlas grading CSV produces coverage and pass fields", () => {
  const row: RunRow = {
    task: "task-1",
    arm: "after",
    rep: 0,
    answer: "A grounded answer",
    trace: { tools: [], decisions: [] },
  };
  const scored = parseCsv("TASK,coverage_score\ntask-1,0.875\n");
  const coverage = new Map(scored.map((score) => [`after\u00000\u0000${score.TASK}`, Number(score.coverage_score)]));
  const graded = applyCoverage([row], coverage, new Map([["task-1", { claims: ["The answer is grounded."] }]]));
  assert.equal(graded[0].coverage, 0.875);
  assert.equal(graded[0].ok, true);
  assert.equal(graded[0].unnoticedMiss, false);
});

test("graded rows keep every trace field the report reads", () => {
  const searches = [{ query: "select:mcp__atlas__wikipedia_search", matches: ["mcp__atlas__wikipedia_search"] }];
  const decision = { hook: "trim-output", ms: 300, inputTokens: 10, result: { kind: "choice", answer: "c0" } };
  const row = runRowSchema.parse({
    task: "task-1",
    arm: "after",
    rep: 0,
    answer: "A grounded answer",
    claudeUsd: 0.1,
    trace: { tools: [], searches, decisions: [decision] },
  });
  const [graded] = applyCoverage([row], new Map(), new Map());
  assert.equal(graded.claudeUsd, 0.1);
  assert.deepEqual(graded.trace.searches, searches);
  assert.deepEqual(graded.trace.decisions, [decision]);
});
