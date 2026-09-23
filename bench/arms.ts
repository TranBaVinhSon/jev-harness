// The before/after arms every benchmark config shares, so the comparison is identical across task sets.
import type { EffortLevel, Options } from "@anthropic-ai/claude-agent-sdk";
import { jevEscalation, rulesEscalation, type EscalationAction } from "../src/jev-escalate.ts";
import type { JevDecisionSink } from "../src/jev.ts";
import { jevToolSearch, type CatalogTool } from "../src/jev-tool-search.ts";
import { createSpillServer, jevTrim } from "../src/jev-trim.ts";
import type { Arm, ArmContext, HookFactory } from "./types.ts";

export type ArmSetup = {
  catalog: (context: ArmContext) => CatalogTool[];
  serverOf?: (name: string) => string;
  trimMatcher: string;
  spillDirectory: string;
};

function effortFrom(value: string | undefined): EffortLevel | undefined {
  switch (value) {
    case "low":
    case "medium":
    case "high":
    case "xhigh":
    case "max":
      return value;
    default:
      return undefined;
  }
}

// bench/probes/effort-cache.ts on claude-opus-5 (2026-09-22): raising effort mid-session re-wrote the
// whole prompt cache, so the cheaper setup is a cheaper model and escalation switches to the strong one.
const strongModel = process.env.BENCH_MODEL ?? "claude-opus-5";
const strongEffort = effortFrom(process.env.BENCH_EFFORT) ?? "high";
const cheapModel = process.env.BENCH_CHEAP_MODEL ?? "claude-sonnet-5";
const cheapEffort = effortFrom(process.env.BENCH_CHEAP_EFFORT) ?? (cheapModel === strongModel ? "low" : strongEffort);
const escalation: EscalationAction =
  cheapModel === strongModel ? { kind: "effort", level: strongEffort } : { kind: "model", model: strongModel };
// With no cheaper setup there is nothing to escalate from, so the escalation arms and hooks drop out.
const canEscalate = cheapModel !== strongModel || cheapEffort !== strongEffort;

const withToolSearch = { env: { ENABLE_TOOL_SEARCH: "true" } };
const strong = (): Options => ({ ...withToolSearch, model: strongModel, effort: strongEffort });
const cheap = (): Options => ({ ...withToolSearch, model: cheapModel, effort: cheapEffort });

const sink =
  ({ trace }: ArmContext): JevDecisionSink =>
  (decision) => {
    trace.decisions.push(decision);
  };

export function beforeAfterArms(setup: ArmSetup): Arm[] {
  const spill = (): Options => ({ mcpServers: { jev: createSpillServer(setup.spillDirectory) } });

  const tools =
    (shadow: boolean): HookFactory =>
    (context) => ({
      PreToolUse: [
        {
          matcher: "ToolSearch",
          hooks: [
            jevToolSearch({
              catalog: () => setup.catalog(context),
              serverOf: setup.serverOf,
              request: () => context.task.prompt,
              shadow,
              sink: sink(context),
            }),
          ],
        },
      ],
    });

  const trim =
    (shadow: boolean): HookFactory =>
    (context) => ({
      PostToolUse: [
        {
          matcher: setup.trimMatcher,
          hooks: [
            jevTrim({
              request: () => context.task.prompt,
              spillDirectory: setup.spillDirectory,
              shadow,
              sink: sink(context),
            }),
          ],
        },
      ],
    });

  const escalate =
    (mode: "jev" | "rules", shadow = false): HookFactory =>
    (context) => {
      const hook = (mode === "jev" ? jevEscalation : rulesEscalation)({
        session: context.session,
        action: escalation,
        request: () => context.task.prompt,
        shadow,
        sink: sink(context),
        onEscalate: (record) => {
          context.trace.escalation = record;
        },
      });
      return { PostToolUseFailure: [{ hooks: [hook] }], PostToolBatch: [{ hooks: [hook] }] };
    };

  const escalateIf = (mode: "jev" | "rules", shadow = false) => (canEscalate ? [escalate(mode, shadow)] : []);
  const arms: Arm[] = [
    { name: "before", options: strong },
    ...(canEscalate
      ? [
          { name: "fixed-cheap", options: cheap },
          { name: "escalate-rules", options: cheap, hooks: escalateIf("rules") },
        ]
      : []),
    ...(process.env.TYPESAFE_API_KEY
      ? [
          { name: "+tools", options: strong, hooks: [tools(false)] },
          { name: "+trim", options: () => ({ ...strong(), ...spill() }), hooks: [trim(false)] },
          ...(canEscalate ? [{ name: "+escalate", options: cheap, hooks: escalateIf("jev") }] : []),
          {
            name: "after",
            options: () => ({ ...cheap(), ...spill() }),
            hooks: [tools(false), trim(false), ...escalateIf("jev")],
          },
          { name: "after-shadow", options: strong, hooks: [tools(true), trim(true), ...escalateIf("jev", true)] },
        ]
      : []),
  ];
  const requested = new Set(process.env.ARMS?.split(",").filter(Boolean) ?? []);
  const unknown = [...requested].filter((name) => !arms.some((arm) => arm.name === name));
  if (unknown.length > 0)
    throw new Error(`Unknown arms ${unknown.join(", ")}. Available: ${arms.map((arm) => arm.name).join(", ")} (Jev arms need TYPESAFE_API_KEY).`);
  return arms.filter((arm) => requested.size === 0 || requested.has(arm.name));
}
