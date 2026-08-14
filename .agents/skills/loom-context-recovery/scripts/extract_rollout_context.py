#!/usr/bin/env python3
"""Extract provenance and native directives from an agent JSONL rollout."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterator


JsonObject = dict[str, Any]


def read_records(path: Path) -> Iterator[tuple[int, JsonObject]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield line_number, value


def first_session_meta(path: Path) -> JsonObject:
    for _, record in read_records(path):
        if record.get("type") != "session_meta":
            continue
        payload = record.get("payload")
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"{path}: no session_meta record")


def without_transport_timestamp(record: JsonObject) -> JsonObject:
    normalized = dict(record)
    normalized.pop("timestamp", None)
    if normalized.get("type") == "session_meta":
        payload = normalized.get("payload")
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.pop("history_mode", None)
            normalized["payload"] = payload
    elif normalized.get("type") == "response_item":
        payload = normalized.get("payload")
        if isinstance(payload, dict) and payload.get("content") is None:
            payload = dict(payload)
            payload.pop("content", None)
            normalized["payload"] = payload
    return normalized


def turn_id(record: JsonObject) -> str | None:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("turn_id")
    if isinstance(candidate, str):
        return candidate
    metadata = payload.get("internal_chat_message_metadata_passthrough")
    if isinstance(metadata, dict):
        candidate = metadata.get("turn_id")
        if isinstance(candidate, str):
            return candidate
    return None


def is_turn_terminal(record: JsonObject) -> bool:
    payload = record.get("payload")
    return (
        record.get("type") == "event_msg"
        and isinstance(payload, dict)
        and payload.get("type") in {"task_complete", "turn_aborted"}
    )


def find_turn_id(
    first: tuple[int, JsonObject],
    records: Iterator[tuple[int, JsonObject]],
    limit: int = 12,
) -> str | None:
    entries = [first]
    for _ in range(limit - 1):
        entry = next(records, None)
        if entry is None:
            break
        entries.append(entry)
    for _, record in entries:
        candidate = turn_id(record)
        if candidate is not None:
            return candidate
    return None


def native_boundary(
    child: Path,
    parent: Path | None,
    fork_id: str | None,
    explicit_start: int | None,
) -> tuple[int, int, str]:
    if explicit_start is not None:
        if explicit_start < 2:
            raise ValueError("--native-start-line must be at least 2")
        return explicit_start, max(0, explicit_start - 2), "explicit native boundary"
    if not fork_id:
        return 2, 0, "non-fork rollout"
    if parent is None:
        raise ValueError(
            "forked rollout requires --parent to avoid attributing imported history"
        )

    parent_meta = first_session_meta(parent)
    parent_id = parent_meta.get("id") or parent_meta.get("session_id")
    if parent_id != fork_id:
        raise ValueError(
            f"parent session {parent_id!r} does not match forked_from_id {fork_id!r}"
        )

    child_records = read_records(child)
    parent_records = read_records(parent)
    next(child_records, None)

    imported = 0
    previous_child: JsonObject | None = None
    after_terminal = False
    pending_start_line: int | None = None
    pending_matches = 0
    while True:
        child_entry = next(child_records, None)
        parent_entry = next(parent_records, None)

        if parent_entry is None:
            imported += pending_matches
            if child_entry is None:
                return imported + 2, imported, "normalized complete-parent comparison"
            return (
                child_entry[0],
                imported,
                "normalized complete-parent prefix",
            )
        if child_entry is None:
            raise ValueError(
                "fork ends inside parent history; supply --native-start-line "
                "after independent verification"
            )

        child_line, child_record = child_entry
        _, parent_record = parent_entry
        if without_transport_timestamp(child_record) == without_transport_timestamp(
            parent_record
        ):
            if pending_start_line is not None:
                child_turn = turn_id(child_record)
                parent_turn = turn_id(parent_record)
                if child_turn is None and parent_turn is None:
                    pending_matches += 1
                    continue
                if child_turn != parent_turn:
                    raise ValueError(
                        f"fork turn identity diverges ambiguously at child line "
                        f"{child_line}; supply --native-start-line after "
                        "independent verification"
                    )
                imported += pending_matches + 1
                pending_start_line = None
                pending_matches = 0
                after_terminal = False
                previous_child = child_record
                continue
            if after_terminal and turn_id(child_record) is None:
                pending_start_line = child_line
                pending_matches = 1
                continue
            imported += 1
            previous_child = child_record
            after_terminal = is_turn_terminal(child_record)
            continue

        if pending_start_line is not None or (
            previous_child is not None and is_turn_terminal(previous_child)
        ):
            child_turn = find_turn_id(child_entry, child_records)
            parent_turn = find_turn_id(parent_entry, parent_records)
            if child_turn is not None and parent_turn is not None:
                if child_turn != parent_turn:
                    return (
                        pending_start_line or child_line,
                        imported,
                        "normalized parent prefix with divergent native turn",
                    )

        raise ValueError(
            f"fork history diverges ambiguously at child line {child_line}; "
            "supply --native-start-line after independent verification"
        )


def message_text(payload: JsonObject) -> tuple[str | None, str | None]:
    payload_type = payload.get("type")
    if payload_type == "user_message":
        message = payload.get("message")
        return (message if isinstance(message, str) else None, None)

    if payload_type != "message" or payload.get("role") != "user":
        return None, None
    content = payload.get("content")
    if not isinstance(content, list):
        return None, None
    parts = [
        item.get("text")
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "input_text"
        and isinstance(item.get("text"), str)
    ]
    metadata = payload.get("internal_chat_message_metadata_passthrough")
    turn_id = metadata.get("turn_id") if isinstance(metadata, dict) else None
    return ("\n".join(parts) if parts else None, turn_id)


def provenance(text: str) -> str:
    stripped = text.lstrip()
    if (
        stripped.startswith("# AGENTS.md instructions")
        or stripped.startswith("<environment_context")
        or stripped.startswith("<INSTRUCTIONS")
    ):
        return "runtime-context"
    if stripped.startswith("<codex_internal_context"):
        return "internal-goal"
    if stripped.startswith("<turn_aborted"):
        return "runtime-control"
    if stripped.startswith("<subagent_notification") or stripped.startswith(
        "Message Type:"
    ):
        return "agent-relay"
    if stripped.startswith("<user_shell_command"):
        return "user-shell"
    return "native-user"


def excerpt(text: str, limit: int | None) -> str:
    compact = " ".join(text.split())
    if limit is None or len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def collect_native(
    path: Path, start_line: int, message_limit: int | None
) -> JsonObject:
    record_types: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    excluded_messages: Counter[str] = Counter()
    directives: list[JsonObject] = []
    recent_messages: dict[str, tuple[int, str]] = {}
    turns: dict[str, JsonObject] = {}
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    for line_number, record in read_records(path):
        if line_number < start_line:
            continue
        timestamp = record.get("timestamp")
        if isinstance(timestamp, str):
            first_timestamp = first_timestamp or timestamp
            last_timestamp = timestamp
        record_type = str(record.get("type", "unknown"))
        record_types[record_type] += 1
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue

        event_type: str | None = None
        turn_id: str | None = None
        if record_type == "event_msg":
            raw_event_type = payload.get("type")
            if isinstance(raw_event_type, str):
                event_type = raw_event_type
                event_types[event_type] += 1
            raw_turn_id = payload.get("turn_id")
            if isinstance(raw_turn_id, str):
                turn_id = raw_turn_id

        if event_type in {"task_started", "task_complete", "turn_aborted"} and turn_id:
            state = {
                "task_started": "active",
                "task_complete": "complete",
                "turn_aborted": "aborted",
            }[event_type]
            turns[turn_id] = {
                "turn_id": turn_id,
                "state": state,
                "line": line_number,
                "timestamp": timestamp,
            }

        text: str | None = None
        message_turn_id: str | None = turn_id
        if record_type == "event_msg" and event_type == "user_message":
            text, _ = message_text(payload)
        elif record_type == "response_item":
            text, message_turn_id = message_text(payload)
        if not text:
            continue

        source = provenance(text)
        previous = recent_messages.get(text)
        if (
            previous is not None
            and previous[1] != record_type
            and line_number - previous[0] <= 3
        ):
            continue
        recent_messages[text] = (line_number, record_type)
        if source not in {"native-user", "user-shell"}:
            excluded_messages[source] += 1
            continue
        directives.append(
            {
                "line": line_number,
                "timestamp": timestamp,
                "turn_id": message_turn_id,
                "source": source,
                "text": excerpt(text, message_limit),
            }
        )

    return {
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "record_types": dict(sorted(record_types.items())),
        "event_types": dict(sorted(event_types.items())),
        "excluded_messages": dict(sorted(excluded_messages.items())),
        "directives": directives,
        "turns": list(turns.values()),
    }


def render_markdown(report: JsonObject) -> str:
    session = report["session"]
    native = report["native"]
    lines = [
        "# Rollout Context",
        "",
        f"- Session: `{session['id']}`",
        f"- Forked from: `{session.get('forked_from_id') or 'none'}`",
        f"- Working directory: `{session.get('cwd') or 'unknown'}`",
        f"- Native boundary: line {report['native_start_line']} "
        f"({report['boundary_method']})",
        f"- Imported records: {report['imported_records']}",
        f"- Native interval: {native.get('first_timestamp') or 'unknown'} to "
        f"{native.get('last_timestamp') or 'unknown'}",
        "",
        "## Native User Directives",
        "",
    ]
    directives = native["directives"]
    if directives:
        for item in directives:
            lines.append(
                f"- `{item['timestamp'] or 'unknown'}` `{item['source']}`: "
                f"{item['text']}"
            )
    else:
        lines.append("- None found")

    lines.extend(["", "## Turn States", ""])
    turns = native["turns"]
    if turns:
        for item in turns:
            lines.append(
                f"- `{item['turn_id']}`: {item['state']} at "
                f"{item['timestamp'] or 'unknown'}"
            )
    else:
        lines.append("- None found")

    lines.extend(["", "## Provenance Summary", ""])
    lines.append(
        f"- Record types: `{json.dumps(native['record_types'], sort_keys=True)}`"
    )
    lines.append(
        f"- Event types: `{json.dumps(native['event_types'], sort_keys=True)}`"
    )
    lines.append(
        f"- Excluded messages: "
        f"`{json.dumps(native['excluded_messages'], sort_keys=True)}`"
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument(
        "--native-start-line",
        type=int,
        help="use an independently verified native boundary",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--full-messages",
        action="store_true",
        help="emit complete native user messages instead of bounded excerpts",
    )
    parser.add_argument("--max-message-chars", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        meta = first_session_meta(args.rollout)
        session_id = meta.get("id") or meta.get("session_id")
        fork_id = meta.get("forked_from_id")
        start_line, imported, method = native_boundary(
            args.rollout,
            args.parent,
            fork_id if isinstance(fork_id, str) else None,
            args.native_start_line,
        )
        limit = None if args.full_messages else args.max_message_chars
        if limit is not None and limit < 1:
            raise ValueError("--max-message-chars must be positive")
        native = collect_native(args.rollout, start_line, limit)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report: JsonObject = {
        "session": {
            "id": session_id,
            "forked_from_id": fork_id,
            "cwd": meta.get("cwd"),
            "thread_source": meta.get("thread_source"),
        },
        "native_start_line": start_line,
        "imported_records": imported,
        "boundary_method": method,
        "native": native,
    }
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
