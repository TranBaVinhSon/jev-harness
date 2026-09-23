#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=4.0"]
# ///

import argparse
import ast
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


TOOL_NAME_MAPPINGS = {
    "brave_brave_web_search": "brave-search_brave_web_search",
    "MongoDB_aggregate": "mongodb_aggregate",
    "MongoDB_collection-schema": "mongodb_collection-schema",
    "MongoDB_count": "mongodb_count",
    "MongoDB_find": "mongodb_find",
    "MongoDB_list-collections": "mongodb_list-collections",
    "MongoDB_list-databases": "mongodb_list-databases",
}


def list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [item.strip() for item in value.split(",") if item.strip()]
    return parsed if isinstance(parsed, list) else []


def tool_names(value: Any) -> list[str]:
    names: list[str] = []
    for item in list_value(value):
        if isinstance(item, str):
            names.append(TOOL_NAME_MAPPINGS.get(item, item))
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            name = item["name"]
            names.append(TOOL_NAME_MAPPINGS.get(name, name))
    return names


def claims(value: Any) -> list[str]:
    return [str(item).strip() for item in list_value(value) if str(item).strip()]


def server_of(tool: str) -> str:
    return tool.split("_", 1)[0].lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the public MCP-Atlas split for the Jev benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("ScaleAI/MCP-Atlas", split="train")
    tasks: list[dict[str, Any]] = []
    ground_truth: list[dict[str, str]] = []
    required_sets: Counter[str] = Counter()
    per_server: Counter[str] = Counter()

    for row in dataset:
        enabled = tool_names(row["ENABLED_TOOLS"])
        required = sorted({server_of(tool) for tool in enabled})
        claim_list = claims(row["GTFA_CLAIMS"])
        tasks.append(
            {
                "id": str(row["TASK"]),
                "prompt": str(row["PROMPT"]),
                "enabledTools": enabled,
                "claims": claim_list,
                "servers": required,
            }
        )
        ground_truth.append(
            {
                "TASK": str(row["TASK"]),
                "PROMPT": str(row["PROMPT"]),
                "GTFA_CLAIMS": json.dumps(claim_list, ensure_ascii=False),
            }
        )
        required_sets[",".join(required)] += 1
        for server in required:
            per_server[server] += 1

    with (args.output_dir / "tasks.jsonl").open("w", encoding="utf-8") as output:
        for task in tasks:
            output.write(json.dumps(task, ensure_ascii=False) + "\n")
    with (args.output_dir / "gt.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["TASK", "PROMPT", "GTFA_CLAIMS"])
        writer.writeheader()
        writer.writerows(ground_truth)

    available = {server for server in os.getenv("ATLAS_SERVERS", "").split(",") if server}
    unlocked = sum(1 for task in tasks if set(task["servers"]).issubset(available)) if available else None
    coverage = {
        "totalTasks": len(tasks),
        "availableServers": sorted(available),
        "unlockedTasks": unlocked,
        "tasksByRequiredServerSet": dict(sorted(required_sets.items())),
        "tasksUsingServer": dict(sorted(per_server.items())),
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {args.output_dir}")
    if unlocked is not None:
        print(f"Configured ATLAS_SERVERS unlock {unlocked} tasks")


if __name__ == "__main__":
    main()
