# Jev agent benchmark

This repository measures whether Jev lowers the cost per passed Claude Agent SDK task without reducing the pass rate. Claude chooses tools and skills. Jev ranks deferred tools, trims large tool results with a read-back path, and recommends one mid-session model or effort escalation after observed lack of progress.

## Install and test

```sh
npm install
npm run typecheck
npm test
```

Run the local demo with a cheap model:

```sh
TYPESAFE_API_KEY=... BENCH_MODEL=sonnet BENCH_CHEAP_MODEL=haiku REPEATS=1 ARMS=before,after npm run bench
npm run report -- bench/results/<run>.jsonl
```

The main arms are `before`, `after`, `+tools`, `+trim`, `+escalate`, `fixed-cheap`, `escalate-rules`, and `after-shadow`. `bench/arms.ts` defines them once for every config. `before` runs `BENCH_MODEL` (default `claude-opus-5`) at `BENCH_EFFORT` (default `high`). `after` starts on `BENCH_CHEAP_MODEL` (default `claude-sonnet-5`) at the same effort and switches to `BENCH_MODEL` once, when both code signals and Jev identify a deeper-reasoning blocker. If `BENCH_CHEAP_MODEL` equals `BENCH_MODEL`, `after` starts at `BENCH_CHEAP_EFFORT` (default `low`) and raises effort instead. If both the model and the effort match, there is nothing to escalate, so the escalation arms drop out and `after` adds only tool ranking and trimming. Jev arms need `TYPESAFE_API_KEY`.

Compare one model with and without Jev:

```sh
BENCH_MODEL=claude-sonnet-5 BENCH_CHEAP_MODEL=claude-sonnet-5 BENCH_CHEAP_EFFORT=high \
  JEV_HOT_TIMEOUT_MS=6000 ARMS=before,after node --env-file=bench/atlas/.env bench/run.ts
```

The Jev hooks use a client with a 1.5 s timeout and no retries, and they pass through when it expires. Jev took 0.4 to 5.2 s per call from a laptop on 2026-09-23, so set `JEV_HOT_TIMEOUT_MS` high enough that the benchmark measures Jev's decisions rather than its timeouts. The report's Jev latency columns and p95 wall time show what that costs.

The cache probe decides between those two actions:

```sh
BENCH_MODEL=claude-opus-5 npm run probe:effort-cache
```

It raises effort after the first tool batch and compares the next call's cache reads with the previous prompt. On 2026-09-22, `claude-opus-5` reused 0% of an 11K-token cached prompt after the effort change and 100% on the following unswitched call. Raising effort therefore costs a full cache rewrite, the same as switching models, so the defaults escalate the model.

The report prints the before/after metrics and paired 95% bootstrap intervals. It returns `SHIP` only when all three conditions hold:

- the upper bound for the change in cost per passed task is below zero;
- the lower bound for the pass-rate change is at least -3 points;
- p95 wall time rises no more than 10%.

Set `BENCH_PRICE_TABLE` to a JSON map when a provider uses prices other than the report defaults. The SDK's `costUSD` remains authoritative. The price table only divides that total among uncached input, cache write, cache read, and output.

## MCP-Atlas

Prerequisites:

- Docker with at least 10 GB available;
- `uv`;
- Anthropic and `TYPESAFE_API_KEY` credentials;
- judge credentials in `EVAL_LLM_API_KEY` and optionally `EVAL_LLM_BASE_URL`;
- a local checkout of [scaleapi/mcp-atlas](https://github.com/scaleapi/mcp-atlas), installed with its Python dependencies;
- Atlas server credentials in `bench/atlas/.env` (an empty file is enough for the keyless servers).

Export the public 500-task dataset:

```sh
KEYLESS=arxiv,calculator,cli-mcp-server,clinicaltrialsgov-mcp-server,context7,ddg-search,desktop-commander,fetch,filesystem,git,mcp-code-executor,mcp-server-code-runner,memory,met-museum,open-library,osm-mcp-server,pubmed,weather,whois,wikipedia
ATLAS_SERVERS=$KEYLESS,github,brave-search,national-parks,twelvedata,weather-data npm run atlas:export
cat bench/atlas/coverage.json
```

`coverage.json` reports how many tasks the configured servers unlock. A task counts only when every server in its enabled tools is available. The 20 keyless servers in `bench/config.atlas.ts` unlock 30 tasks, and adding the five free-tier servers above unlocks 85. Aim for at least 150 tasks before the main run.

Run the five-task smoke:

```sh
ATLAS_SERVERS=... TASKS=<id1>,<id2>,<id3>,<id4>,<id5> \
  BENCH_MODEL=sonnet BENCH_CHEAP_MODEL=haiku REPEATS=1 ARMS=before,after npm run atlas:bench
```

The runner starts one `ghcr.io/scaleapi/mcp-atlas:1.2.7` container per concurrency slot. Startup takes a few minutes because the image installs its MCP servers on boot. Every task leases one container. A task that uses a stateful server causes that container to restart before the next lease. The pool pins `mcp<2` for the image's `uvx` servers. Without the pin, seven keyless servers fail on import against mcp 2.x.

Grade and report a completed run:

```sh
MCP_ATLAS_DIR=/path/to/mcp-atlas \
  EVAL_LLM_MODEL=gemini/gemini-3.1-pro-preview \
  npm run atlas:grade -- bench/results/<run>.jsonl

npm run report -- bench/results/<run>.graded.jsonl
```

Without an OpenAI-compatible judge endpoint, set `ATLAS_JUDGE=claude` and `EVAL_LLM_MODEL=claude-opus-5`. `bench/atlas/judge.ts` then scores each claim through the Agent SDK with the same prompt, schema, and scoring as `score_claims.py`. Those scores compare arms fairly but are not comparable to the public leaderboard.

The grader runs each arm and repeat through the same judge, sets `coverage`, marks a task passed at coverage 0.75, and flags unnoticed trim misses when omitted chunks contain ground-truth values and the agent never called `read_spill`.

## App-specific tasks

Copy `bench/config.ts` to `bench/config.<app>.ts`, replace the MCP servers and task file, and keep the arm definitions unchanged. A benchmark task may specify `expectTools`, `expectAnswer`, and an application-specific `check`. Use `lease` when every run needs an isolated sandbox. Add skill advice only in the app benchmark because MCP-Atlas has no skills.
