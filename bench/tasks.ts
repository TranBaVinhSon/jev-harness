import { readFileSync } from "node:fs";
import { z } from "zod";
import type { AnswerExpectation, Task, Trace } from "./types.ts";

const answerExpectationSchema = z.union([
  z.string().transform((value): AnswerExpectation => ({ kind: "substring", value })),
  z.object({ substring: z.string() }).transform(({ substring }): AnswerExpectation => ({ kind: "substring", value: substring })),
  z
    .object({ regex: z.string(), flags: z.string().optional() })
    .transform(({ regex, flags }): AnswerExpectation => ({ kind: "regex", pattern: regex, flags })),
]);

const taskSchema = z.object({
  id: z.string().min(1),
  prompt: z.string().min(1),
  expectTools: z.array(z.string().min(1)).optional(),
  expectAnswer: answerExpectationSchema.optional(),
  setup: z.json().optional(),
});

export type TaskCheck = (answer: string, trace: Trace) => boolean;

export function answerMatches(answer: string, expectation: AnswerExpectation | undefined): boolean {
  if (!expectation) return true;
  switch (expectation.kind) {
    case "substring":
      return answer.includes(expectation.value);
    case "regex":
      return new RegExp(expectation.pattern, expectation.flags).test(answer);
    default: {
      const exhaustive: never = expectation;
      return exhaustive;
    }
  }
}

export function loadTasks(path: string, checks: Record<string, TaskCheck> = {}): Task[] {
  const entries = readFileSync(path, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return taskSchema.parse(JSON.parse(line));
      } catch (error) {
        throw new Error(`${path}:${index + 1}: ${String(error)}`);
      }
    });

  const seen = new Set<string>();
  return entries.map((entry) => {
    if (seen.has(entry.id)) throw new Error(`${path}: duplicate task id ${entry.id}`);
    seen.add(entry.id);
    return {
      ...entry,
      check: (answer: string, trace: Trace) => {
        const used = new Set(trace.tools.map((tool) => tool.name));
        const expectedToolsPass = entry.expectTools?.every((tool) => used.has(tool)) ?? true;
        return expectedToolsPass && answerMatches(answer, entry.expectAnswer) && (checks[entry.id]?.(answer, trace) ?? true);
      },
    };
  });
}
