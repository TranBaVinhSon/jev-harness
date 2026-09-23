import {
  createSdkMcpServer,
  query,
  tool,
  type HookCallback,
  type Query,
  type SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";

type CallUsage = { promptTokens: number; cacheRead: number; cacheWrite: number };

let liveQuery: Query | undefined;
let switched = false;

const switchEffort: HookCallback = async (input) => {
  if (input.hook_event_name !== "PostToolBatch" || switched) return {};
  if (!liveQuery) throw new Error("Query was not initialized before the tool batch completed");
  switched = true;
  await liveQuery.applyFlagSettings({ effortLevel: "high" });
  return {};
};

const probe = createSdkMcpServer({
  name: "probe",
  tools: [
    tool("ping", "Return a short probe value.", {}, async () => ({
      content: [{ type: "text", text: "pong" }],
    })),
  ],
});

let finishInput: () => void = () => {};
const inputFinished = new Promise<void>((resolve) => {
  finishInput = resolve;
});
async function* prompt(): AsyncGenerator<SDKUserMessage> {
  yield {
    type: "user",
    message: { role: "user", content: "Call the ping tool twice, sequentially, then answer with both results." },
    parent_tool_use_id: null,
  };
  await inputFinished;
}

liveQuery = query({
  prompt: prompt(),
  options: {
    model: process.env.BENCH_MODEL ?? "claude-opus-5",
    effort: "low",
    maxTurns: 5,
    tools: [],
    // Padding keeps the first prompt above every model's minimum cacheable prefix.
    systemPrompt: [
      "You are a warehouse assistant. The current inventory is below.",
      "| bin | part | units |",
      ...Array.from({ length: 600 }, (_, row) => `| A${1000 + row} | P-${(row * 37) % 9000} | ${(row * 13) % 97} |`),
    ].join("\n"),
    mcpServers: { probe },
    allowedTools: ["mcp__probe__ping"],
    permissionMode: "dontAsk",
    settingSources: [],
    strictMcpConfig: true,
    hooks: { PostToolBatch: [{ hooks: [switchEffort] }] },
  },
});

const calls = new Map<string, CallUsage>();
try {
  for await (const message of liveQuery) {
    if (message.type === "assistant" && message.parent_tool_use_id === null) {
      const usage = message.message.usage;
      const cacheRead = usage.cache_read_input_tokens ?? 0;
      const cacheWrite = usage.cache_creation_input_tokens ?? 0;
      calls.set(message.message.id, { promptTokens: usage.input_tokens + cacheRead + cacheWrite, cacheRead, cacheWrite });
    }
    if (message.type === "result") finishInput();
  }
} finally {
  finishInput();
}

// The hook switches effort after the first tool batch, so call 2 is the first call after the switch
// and call 2 -> 3 is an unswitched control. A cached system prompt alone would read above zero, so
// compare against the whole previous prompt.
const [first, second, third] = calls.values();
const reused = (previous?: CallUsage, next?: CallUsage) =>
  previous && next ? Number((next.cacheRead / previous.promptTokens).toFixed(3)) : null;
const afterSwitch = reused(first, second);
const control = reused(second, third);
const cacheSurvived = afterSwitch !== null && afterSwitch >= 0.9;
console.log(
  JSON.stringify(
    {
      switched,
      calls: [...calls.values()],
      reusedAfterSwitch: afterSwitch,
      reusedWithoutSwitch: control,
      cacheSurvived,
      recommendation: cacheSurvived
        ? "Use one model at low effort and escalate effort."
        : "Start on the cheaper model and escalate to the production model.",
    },
    null,
    2,
  ),
);
