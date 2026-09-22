import type { HookCallback } from "@anthropic-ai/claude-agent-sdk";
import { TypeSafeClient, choice, type ChoiceResponse } from "@typesafe-ai/sdk";
import { z } from "zod";
import { ask, getJevClient, type JevDecisionSink } from "./jev.ts";

export type CatalogTool = { name: string; description: string };

export type JevToolSearchOptions = {
  catalog: () => CatalogTool[] | Promise<CatalogTool[]>;
  request: (sessionId: string) => string;
  shadow?: boolean;
  sink?: JevDecisionSink;
  client?: TypeSafeClient;
};

const toolSearchInputSchema = z.object({ query: z.string(), max_results: z.number().int().positive().optional() });
const MAX_CHOICES = 255;
const MAX_CANDIDATES = MAX_CHOICES - 1;
const NONE = "none_of_these";
const NONE_DESCRIPTION = "No listed tool or server fits the request.";
const SERVER_MASS = 0.9;
const MAX_SERVERS = 3;
const EXTRA_FOR_SELECT = 2;

const serverOf = (name: string) => name.split("__")[1] ?? name;

function ranked(answer: ChoiceResponse): [string, number][] {
  return Object.entries(answer.probabilities).sort((left, right) => right[1] - left[1]);
}

function selectedNames(query: string): string[] {
  if (!query.startsWith("select:")) return [];
  return query
    .slice("select:".length)
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
}

function picksFor(answer: ChoiceResponse, selected: string[], maxResults: number | undefined): string[] {
  if (answer.choice === NONE) return [];
  const names = ranked(answer).map(([name]) => name).filter((name) => name !== NONE);
  return selected.length > 0
    ? [...selected, ...names.filter((name) => !selected.includes(name)).slice(0, EXTRA_FOR_SELECT)]
    : names.slice(0, Math.max(3, maxResults ?? 5));
}

export function jevToolSearch(options: JevToolSearchOptions): HookCallback {
  return async (input, _toolUseId, hookOptions) => {
    if (input.hook_event_name !== "PreToolUse") return {};
    const parsed = toolSearchInputSchema.safeParse(input.tool_input);
    if (!parsed.success) return {};

    const args = parsed.data;
    const selected = selectedNames(args.query);
    const client = options.client ?? getJevClient("hot");

    try {
      let candidates = await options.catalog();

      if (candidates.length > MAX_CANDIDATES) {
        const servers = Map.groupBy(candidates, (tool) => serverOf(tool.name));
        if (servers.size > MAX_CANDIDATES) return {};
        const serverAnswer = await ask({
          hook: "tool-search.server",
          sessionId: input.session_id,
          state: { user_request: options.request(input.session_id), tool_search_query: args.query },
          question: choice(
            "Which tool server is most likely to contain the tool needed for this request?",
            Object.fromEntries([
              ...[...servers].map(([server, tools]) => [
                server,
                tools.map((tool) => tool.name.split("__").pop()).join(", "),
              ]),
              [NONE, NONE_DESCRIPTION],
            ]),
          ),
          client,
          sink: options.sink,
          signal: hookOptions.signal,
          details: () => ({ stage: "server", query: args.query, candidateCount: servers.size }),
        });
        if (serverAnswer.choice === NONE) return {};

        const keep: string[] = [];
        let mass = 0;
        for (const [server, probability] of ranked(serverAnswer)) {
          if (server === NONE) continue;
          if (mass >= SERVER_MASS || keep.length === MAX_SERVERS) break;
          keep.push(server);
          mass += probability;
        }
        candidates = keep.flatMap((server) => servers.get(server) ?? []);
      }

      if (candidates.length === 0 || candidates.length > MAX_CANDIDATES) return {};

      const toolAnswer = await ask({
        hook: "tool-search.tool",
        sessionId: input.session_id,
        state: { user_request: options.request(input.session_id), tool_search_query: args.query },
        question: choice(
          "Which tool best provides what this search needs?",
          Object.fromEntries([
            ...candidates.map((tool) => [tool.name, tool.description.slice(0, 400)]),
            [NONE, NONE_DESCRIPTION],
          ]),
        ),
        client,
        sink: options.sink,
        signal: hookOptions.signal,
        details: (answer) => ({
          stage: "tool",
          query: args.query,
          candidateCount: candidates.length,
          selected,
          picked: picksFor(answer, selected, args.max_results),
          applied: !options.shadow && answer.choice !== NONE,
        }),
      });
      const picked = picksFor(toolAnswer, selected, args.max_results);
      if (picked.length === 0 || options.shadow) return {};

      return {
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          updatedInput: { ...args, query: `select:${picked.join(",")}`, max_results: picked.length },
        },
      };
    } catch {
      return {};
    }
  };
}
