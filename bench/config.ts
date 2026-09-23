// Demo config with overlapping MCP tool names. Copy this file to
// bench/config.<app>.ts and replace the servers, prompt, and task file.
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createSdkMcpServer, tool, type Options } from "@anthropic-ai/claude-agent-sdk";
import type { CatalogTool } from "../src/jev-tool-search.ts";
import { beforeAfterArms } from "./arms.ts";
import { loadTasks } from "./tasks.ts";
import type { BenchConfig } from "./types.ts";

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

function servers(): Options["mcpServers"] {
  return Object.fromEntries(
    Object.entries(definitions).map(([server, tools]) => [
      server,
      createSdkMcpServer({
        name: server,
        tools: Object.entries(tools).map(([name, description]) =>
          tool(name, description, {}, async () => ({
            content: [{ type: "text", text: `Result code: ${codeOf(server, name)}` }],
          })),
        ),
      }),
    ]),
  );
}

const catalog: CatalogTool[] = Object.entries(definitions).flatMap(([server, tools]) =>
  Object.entries(tools).map(([name, description]) => ({ name: `mcp__${server}__${name}`, description })),
);

const tasks = loadTasks(fileURLToPath(new URL("./tasks/demo.jsonl", import.meta.url)));
const requestedTasks = new Set(process.env.TASKS?.split(",").filter(Boolean) ?? []);

const config: BenchConfig = {
  tasks: tasks.filter((task) => requestedTasks.size === 0 || requestedTasks.has(task.id)),
  arms: beforeAfterArms({
    catalog: () => catalog,
    trimMatcher: "mcp__.*|Bash|WebFetch",
    spillDirectory: resolve(".context/bench-spills"),
  }),
  repeats: Number(process.env.REPEATS ?? 3),
  concurrency: Number(process.env.CONCURRENCY ?? 4),
  base: () => ({
    maxTurns: 12,
    systemPrompt:
      "You are the internal operations assistant. Answer by calling company tools. Customer and employee names are existing records, so do not ask for clarification. Quote exact result codes.",
    tools: ["ToolSearch"],
    mcpServers: servers(),
    allowedTools: [...Object.keys(definitions).map((server) => `mcp__${server}__*`), "mcp__jev__read_spill"],
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
