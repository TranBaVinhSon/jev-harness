// Demo config with overlapping MCP tool names. Copy this file to
// bench/config.<app>.ts and replace the servers, prompt, models, and task file.
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createSdkMcpServer, tool, type EffortLevel, type Options } from "@anthropic-ai/claude-agent-sdk";
import type { JevDecisionSink } from "../src/jev.ts";
import { jevSubagentRoute, routeSession, type RoutePolicy } from "../src/jev-route.ts";
import { jevToolSearch, type CatalogTool } from "../src/jev-tool-search.ts";
import { jevTrim } from "../src/jev-trim.ts";
import { loadTasks } from "./tasks.ts";
import type { Arm, ArmContext, BenchConfig, HookFactory } from "./types.ts";

const definitions: Record<string, Record<string, string>> = {
  crm: {
    get_customer_profile: "Look up a customer's company, address, and account owner.",
    get_loyalty_tier: "Return a customer's loyalty program tier.",
    list_customer_contacts: "List contact people at a customer company.",
    get_account_health: "Return the customer success health score for an account.",
    update_customer_notes: "Append a note to a customer's CRM record.",
    get_customer_segment: "Return the market segment for a customer.",
  },
  billing: {
    get_customer_invoices: "List invoices issued to a customer.",
    get_outstanding_balance: "Return the unpaid amount a customer owes.",
    get_payment_method: "Return the card or bank account a customer pays with.",
    get_subscription_plan: "Return the pricing plan for a customer subscription.",
    issue_refund: "Refund a customer payment.",
    get_tax_region: "Return the tax jurisdiction printed on customer invoices.",
  },
  support: {
    get_customer_tickets: "List support tickets opened by a customer.",
    get_ticket_sla: "Return the response-time SLA for a support tier.",
    get_csat_score: "Return the average satisfaction rating for closed support tickets.",
    escalate_ticket: "Escalate a ticket to the on-call engineer.",
    get_known_issues: "List current product incidents.",
    get_support_tier: "Return the support package a customer purchased.",
  },
  hr: {
    get_employee_profile: "Look up an employee profile.",
    get_pto_balance: "Return an employee's remaining paid time off.",
    get_org_chart: "Return reporting relationships.",
    get_payroll_region: "Return an employee's payroll jurisdiction.",
    list_open_roles: "List open job requisitions.",
    get_onboarding_status: "Return an employee's onboarding progress.",
  },
};

const codes = new Map(
  Object.entries(definitions)
    .flatMap(([server, tools]) => Object.keys(tools).map((name) => `${server}.${name}`))
    .map((key, index) => [key, `REF-${4101 + index * 7}`]),
);

function codeOf(server: string, name: string): string {
  const code = codes.get(`${server}.${name}`);
  if (!code) throw new Error(`Missing demo result code for ${server}.${name}`);
  return code;
}

const alwaysLoaded = new Set([
  "crm.get_customer_profile",
  "billing.get_outstanding_balance",
  "support.get_customer_tickets",
  "support.get_known_issues",
]);

function servers(tuned = false): Options["mcpServers"] {
  return Object.fromEntries(
    Object.entries(definitions).map(([server, tools]) => [
      server,
      createSdkMcpServer({
        name: server,
        tools: Object.entries(tools).map(([name, description]) =>
          tool(
            name,
            tuned ? `${description} This tool belongs to the ${server} category.` : description,
            {},
            async () => ({ content: [{ type: "text", text: `Result code: ${codeOf(server, name)}` }] }),
            { alwaysLoad: tuned && alwaysLoaded.has(`${server}.${name}`) },
          ),
        ),
      }),
    ]),
  );
}

const catalog: CatalogTool[] = Object.entries(definitions).flatMap(([server, tools]) =>
  Object.entries(tools).map(([name, description]) => ({ name: `mcp__${server}__${name}`, description })),
);

const tasks = loadTasks(fileURLToPath(new URL("./tasks/demo.jsonl", import.meta.url)));
const toolSearch = (enabled: boolean): Options => ({ env: { ENABLE_TOOL_SEARCH: enabled ? "true" : "false" } });
const decisionSink = ({ trace }: ArmContext): JevDecisionSink => (decision) => {
  trace.decisions.push(decision);
};

function toolSearchHooks(shadow: boolean): HookFactory {
  return (context) => ({
    PreToolUse: [
      {
        matcher: "ToolSearch",
        hooks: [
          jevToolSearch({
            catalog: () => catalog,
            request: () => context.task.prompt,
            shadow,
            sink: decisionSink(context),
          }),
        ],
      },
    ],
  });
}

function trimHooks(shadow: boolean): HookFactory {
  return (context) => ({
    PostToolUse: [
      {
        matcher: "mcp__.*|Bash|WebFetch",
        hooks: [
          jevTrim({
            request: () => context.task.prompt,
            shadow,
            spillDirectory: resolve(".context/bench-spills"),
            sink: decisionSink(context),
          }),
        ],
      },
    ],
  });
}

const strongModel = process.env.BENCH_MODEL ?? "claude-opus-5";
const routePolicy: RoutePolicy = {
  lookup: { model: process.env.BENCH_LOOKUP_MODEL ?? "haiku", effort: "low" },
  specifiedChange: { model: process.env.BENCH_CHANGE_MODEL ?? "sonnet", effort: "medium" },
  strong: { model: strongModel, effort: effortFromEnv(process.env.BENCH_EFFORT) },
};

function routeHooks(shadow: boolean): HookFactory {
  return (context) => ({
    PreToolUse: [
      {
        matcher: "Agent",
        hooks: [
          jevSubagentRoute({
            request: () => context.task.prompt,
            lookupModel: routePolicy.lookup.model,
            specifiedChangeModel: routePolicy.specifiedChange.model,
            shadow,
            sink: decisionSink(context),
          }),
        ],
      },
    ],
  });
}

function effortFromEnv(value: string | undefined): EffortLevel {
  switch (value) {
    case "low":
    case "medium":
    case "high":
    case "xhigh":
    case "max":
      return value;
    default:
      return "high";
  }
}

function routeArm(name: string, shadow: boolean): Arm {
  return {
    name,
    hooks: [routeHooks(shadow)],
    options: async (context) => {
      const route = await routeSession({
        task: context.task.prompt,
        sessionId: context.runId,
        policy: routePolicy,
        sink: decisionSink(context),
      });
      return shadow ? routePolicy.strong : route.options;
    },
  };
}

const arms: Arm[] = [
  { name: "baseline", options: () => toolSearch(true) },
  { name: "all-tools", options: () => toolSearch(false) },
  {
    name: "tuned-search",
    options: () => ({
      ...toolSearch(true),
      mcpServers: servers(true),
      systemPrompt:
        "You are the internal operations assistant. Tool categories: crm owns profiles and accounts; billing owns invoices and payments; support owns tickets and service levels; hr owns employee data. Call company tools and quote exact result codes.",
    }),
  },
  { name: "low-effort", options: () => ({ ...toolSearch(true), effort: "low" }) },
  ...(process.env.TYPESAFE_API_KEY
    ? [
        { name: "jev-shadow", options: () => toolSearch(true), hooks: [toolSearchHooks(true)] },
        { name: "jev", options: () => toolSearch(true), hooks: [toolSearchHooks(false)] },
        { name: "trim-shadow", options: () => toolSearch(true), hooks: [trimHooks(true)] },
        { name: "trim", options: () => toolSearch(true), hooks: [trimHooks(false)] },
        routeArm("route-shadow", true),
        routeArm("route", false),
      ]
    : []),
];

const requestedTasks = new Set(process.env.TASKS?.split(",").filter(Boolean) ?? []);
const requestedArms = new Set(process.env.ARMS?.split(",").filter(Boolean) ?? []);

const config: BenchConfig = {
  tasks: tasks.filter((task) => requestedTasks.size === 0 || requestedTasks.has(task.id)),
  arms: arms.filter((arm) => requestedArms.size === 0 || requestedArms.has(arm.name)),
  repeats: Number(process.env.REPEATS ?? 3),
  concurrency: Number(process.env.CONCURRENCY ?? 4),
  base: () => ({
    model: strongModel,
    effort: effortFromEnv(process.env.BENCH_EFFORT),
    maxTurns: 12,
    systemPrompt:
      "You are the internal operations assistant. Answer by calling company tools. Customer and employee names are existing records, so do not ask for clarification. Quote exact result codes.",
    tools: ["ToolSearch"],
    mcpServers: servers(),
    allowedTools: Object.keys(definitions).map((server) => `mcp__${server}__*`),
    permissionMode: "dontAsk",
    settingSources: [],
    strictMcpConfig: true,
    env: {
      ...process.env,
      ENABLE_CLAUDEAI_MCP_SERVERS: "false",
      CLAUDE_CODE_MCP_STARTUP_WAIT_MS: "15000",
    },
  }),
};

export default config;
