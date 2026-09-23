// A Claude judge that mirrors mcp-atlas services/scoring/score_claims.py (f24ba3f): the same per-claim
// prompt, schema, claim cleanup, and 1 / 0.5 / 0 scoring, run through the Agent SDK for machines
// without an OpenAI-compatible judge endpoint. Scores are comparable across arms, not to the leaderboard.
import { query } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

const outcomeScore = { fulfilled: 1, partially_fulfilled: 0.5, not_fulfilled: 0 } as const;
const judgementSchema = z.object({ coverage_outcome: z.enum(["fulfilled", "partially_fulfilled", "not_fulfilled"]) }).loose();
const responseSchema = {
  type: "object",
  properties: {
    claim_text: { type: "string" },
    coverage_outcome: { type: "string", enum: ["fulfilled", "partially_fulfilled", "not_fulfilled"] },
    justification: { type: "string" },
    confidence_level: { type: "number" },
  },
  required: ["claim_text", "coverage_outcome", "justification", "confidence_level"],
};

export function cleanClaim(text: string): string {
  return text
    .trim()
    .replace(/^[-*•·◦‣⁃]\s*/, "")
    .replace(/^\d+[.)]\s*/, "")
    .replaceAll("“", '"')
    .replace(/[”"]/g, '"')
    .replaceAll("‘", "'")
    .replaceAll("’", "'")
    .replaceAll("–", "-")
    .replaceAll("—", "-")
    .replaceAll("…", "...")
    .replace(/[.\s]*["']+ $/, "")
    .replace(/["']+\.*$/, "")
    .trim();
}

function claimPrompt(claim: string, response: string): string {
  return `You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
${claim}
MODEL RESPONSE TO ANALYZE:
${response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning.`;
}

type Judgement = { score: number } | { error: string };

async function judgeOnce(model: string, claim: string, response: string): Promise<Judgement> {
  try {
    for await (const message of query({
      prompt: claimPrompt(claim, response),
      options: {
        model,
        maxTurns: 3,
        tools: [],
        outputFormat: { type: "json_schema", schema: responseSchema },
        settingSources: [],
        persistSession: false,
        env: { ...process.env, ENABLE_CLAUDEAI_MCP_SERVERS: "false" },
      },
    })) {
      if (message.type !== "result") continue;
      if (message.subtype !== "success") return { error: message.subtype };
      const judgement = judgementSchema.safeParse(message.structured_output);
      return judgement.success ? { score: outcomeScore[judgement.data.coverage_outcome] } : { error: "invalid structured output" };
    }
    return { error: "no result" };
  } catch (error) {
    return { error: String(error).slice(0, 200) };
  }
}

// score_claims.py retries failed calls before scoring a claim not_fulfilled; without retries a
// transient failure under concurrency turns a correct answer into a zero.
async function scoreClaim(model: string, claim: string, response: string): Promise<Judgement> {
  let judgement: Judgement = { error: "not attempted" };
  for (let attempt = 0; attempt < 4; attempt++) {
    judgement = await judgeOnce(model, claim, response);
    if ("score" in judgement) return judgement;
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 2_000 * 2 ** attempt));
  }
  return judgement;
}

export type JudgeItem = { key: string; claims: string[]; response: string };

export async function claudeCoverage(items: JudgeItem[], model: string, concurrency = 8): Promise<Map<string, number>> {
  const work = items.flatMap((item) => item.claims.map((claim, index) => ({ item, index, claim: cleanClaim(claim) })));
  const scores = new Map(items.map((item) => [item.key, new Array<number>(item.claims.length).fill(0)]));
  const failures: string[] = [];
  let next = 0;
  await Promise.all(
    Array.from({ length: concurrency }, async () => {
      while (next < work.length) {
        const { item, index, claim } = work[next++];
        const judgement = await scoreClaim(model, claim, item.response);
        const itemScores = scores.get(item.key);
        if (itemScores && "score" in judgement) itemScores[index] = judgement.score;
        if ("error" in judgement) failures.push(judgement.error);
      }
    }),
  );
  if (failures.length > 0)
    console.warn(`${failures.length} of ${work.length} claim evaluations failed after retries and scored 0: ${[...new Set(failures)].join("; ")}`);
  return new Map(
    [...scores].flatMap(([key, values]) =>
      values.length === 0 ? [] : [[key, Math.round((1000 * values.reduce((sum, value) => sum + value, 0)) / values.length) / 1000]],
    ),
  );
}
