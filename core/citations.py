from __future__ import annotations

from typing import Any, Mapping


def seconds_to_timestamp(seconds: float | int) -> str:
    """Convert seconds into MM:SS or HH:MM:SS."""
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_value = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"
    return f"{minutes:02d}:{seconds_value:02d}"


def format_timestamp_range(start_seconds: float | int | None, end_seconds: float | int | None) -> str:
    if start_seconds is None or end_seconds is None:
        return "Timestamp unavailable"
    return f"{seconds_to_timestamp(start_seconds)} - {seconds_to_timestamp(end_seconds)}"


def format_source_citation(metadata: Mapping[str, Any]) -> str:
    source_file = metadata.get("source_file") or metadata.get("source_name") or "Unknown source"
    start_seconds = metadata.get("start_seconds")
    end_seconds = metadata.get("end_seconds")
    timestamp_range = format_timestamp_range(start_seconds, end_seconds)
    return f"{source_file}\n{timestamp_range}"