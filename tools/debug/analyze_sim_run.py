#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict


DFG_OP_RE = re.compile(r"=\s*([A-Za-z0-9_.]+)\b")


def load_json(path):
    return json.loads(pathlib.Path(path).read_text())


def summarize_dfg(path):
    counts = Counter()
    in_handshake_func = False
    brace_depth = 0
    for line in pathlib.Path(path).read_text().splitlines():
        if not in_handshake_func:
            if "handshake.func" in line:
                in_handshake_func = True
                brace_depth = line.count("{") - line.count("}")
            continue
        brace_depth += line.count("{") - line.count("}")
        if brace_depth < 0:
            break
        match = DFG_OP_RE.search(line)
        if not match:
            if brace_depth == 0:
                break
            continue
        op = match.group(1)
        if op in ("join", "return"):
            counts[f"handshake.{op}"] += 1
        else:
            counts[op] += 1
        if brace_depth == 0:
            break
    return counts


def summarize_trace(trace_obj):
    modules = {mod["hw_node_id"]: mod for mod in trace_obj.get("modules", [])}
    fires = Counter()
    for event in trace_obj.get("events", []):
        if event.get("event_kind") == "node_fire":
            fires[event["hw_node_id"]] += 1
    return modules, fires


def aggregate_module_summaries(result_obj, trace_modules, trace_fires):
    rows = []
    summaries = result_obj.get("final_state", {}).get("module_summaries", [])
    for summary in summaries:
        hw_node_id = summary["hw_node_id"]
        trace_info = trace_modules.get(hw_node_id, {})
        counters = {it["name"]: it["value"] for it in summary.get("counters", [])}
        rows.append(
            {
                "hw_node_id": hw_node_id,
                "name": summary.get("name", trace_info.get("name", "")),
                "kind": summary.get("kind", trace_info.get("kind", "")),
                "trace_fire_count": trace_fires.get(hw_node_id, 0),
                "logical_fire_count": summary.get("logical_fire_count", 0),
                "input_capture_count": summary.get("input_capture_count", 0),
                "output_transfer_count": summary.get("output_transfer_count", 0),
                "has_pending_work": summary.get("has_pending_work", False),
                "collected_token_count": summary.get("collected_token_count", 0),
                "debug_state": summary.get("debug_state", ""),
                "counters": counters,
            }
        )
    return rows


def print_counter_table(title, counter_map):
    print(title)
    for key in sorted(counter_map):
        print(f"  {key}: {counter_map[key]}")


def print_module_rows(rows, name_filter):
    grouped = defaultdict(list)
    for row in rows:
        if name_filter and name_filter not in row["name"]:
            continue
        grouped[row["name"]].append(row)

    print("Module summaries")
    for name in sorted(grouped):
        for row in sorted(grouped[name], key=lambda it: it["hw_node_id"]):
            interesting = (
                row["trace_fire_count"]
                or row["logical_fire_count"]
                or row["input_capture_count"]
                or row["output_transfer_count"]
                or row["has_pending_work"]
                or row["collected_token_count"]
                or row["counters"]
            )
            if not interesting:
                continue
            print(
                f"  hw={row['hw_node_id']:4d} kind={row['kind']:16s} "
                f"trace_fire={row['trace_fire_count']:4d} "
                f"logical_fire={row['logical_fire_count']:4d} "
                f"input_capture={row['input_capture_count']:4d} "
                f"output_transfer={row['output_transfer_count']:4d} "
                f"pending={int(row['has_pending_work'])} "
                f"name={row['name']}"
            )
            for key in sorted(row["counters"]):
                print(f"    counter {key}={row['counters'][key]}")
            if row["debug_state"]:
                print(f"    state   {row['debug_state']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfg", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--name-filter", default="")
    args = parser.parse_args()

    dfg_counts = summarize_dfg(args.dfg)
    trace_obj = load_json(args.trace)
    result_obj = load_json(args.result)
    trace_modules, trace_fires = summarize_trace(trace_obj)
    rows = aggregate_module_summaries(result_obj, trace_modules, trace_fires)

    print_counter_table("DFG op inventory", dfg_counts)
    print()
    print_module_rows(rows, args.name_filter)
    print()
    print("Termination summary")
    final_state = result_obj.get("final_state", {})
    for key in (
        "obligations_satisfied",
        "quiescent",
        "done",
        "deadlocked",
        "idle_cycle_streak",
        "outstanding_memory_request_count",
        "completed_memory_response_count",
    ):
        print(f"  {key}: {final_state.get(key)}")


if __name__ == "__main__":
    main()
