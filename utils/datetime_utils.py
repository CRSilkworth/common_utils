from typing import Any, Union, Tuple
from utils.type_utils import TimeRanges
import datetime
from email.utils import parsedate_to_datetime
import calendar

EPOCH = datetime.datetime(1970, 1, 1)
MIN_TS = -1e18
MAX_TS = 1e18
datetime_min = datetime.datetime.min
datetime_max = datetime.datetime.max


def datetime_to_float(d: datetime.datetime) -> float:
    """Convert datetime to float seconds since epoch, works for min/max."""
    delta = d - EPOCH
    return delta.total_seconds()


def float_to_datetime(value: float) -> datetime.datetime:
    """Convert float seconds since epoch back to datetime."""
    return EPOCH + datetime.timedelta(seconds=value)


def to_micro(dt):
    return dt.replace(microsecond=(dt.microsecond // 1000) * 1000)


def _to_iso(dt: datetime.datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="microseconds")


def generate_time_ranges(
    start: datetime.datetime, end: datetime.datetime, period: str
) -> TimeRanges:
    """
    Generate consecutive time ranges between two datetime.datetimes.

    Args:
        start: Start datetime.datetime (inclusive).
        end: End datetime.datetime (inclusive).
        period: One of: "daily", "weekly", "monthly", "yearly".

    Returns:
        List of (range_start, range_end) datetime.datetime tuples.
    """

    if start > end:
        raise ValueError("start must be before end")

    period = period.lower()
    ranges = []

    current = start

    while current <= end:
        if period == "daily":
            next_dt = current + datetime.timedelta(days=1)

        elif period == "weekly":
            next_dt = current + datetime.timedelta(weeks=1)

        elif period == "monthly":
            year = current.year
            month = current.month

            # Move to first of next month
            if month == 12:
                next_year, next_month = year + 1, 1
            else:
                next_year, next_month = year, month + 1

            # Try to preserve day-of-month if possible
            day = min(current.day, calendar.monthrange(next_year, next_month)[1])
            next_dt = current.replace(year=next_year, month=next_month, day=day)

        elif period == "yearly":
            try:
                next_dt = current.replace(year=current.year + 1)
            except ValueError:
                # Handle Feb 29 -> Feb 28 for non-leap years
                next_dt = current.replace(month=2, day=28, year=current.year + 1)

        else:
            raise ValueError("period must be one of: daily, weekly, monthly, yearly")

        range_end = min(next_dt, end)
        ranges.append((to_micro(current), to_micro(range_end)))

        current = next_dt

    return ranges


def convert_timestamps(obj):
    """
    Recursively convert timestamps/datetimes and datetime-like strings to ISO8601.
    """

    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_timestamps(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_timestamps(v) for v in obj)

    # Already a datetime
    if isinstance(obj, datetime.datetime):
        return _to_iso(obj)

    # Try to detect datetime-like strings
    if isinstance(obj, str):
        dt = _maybe_parse_datetime(obj)
        if dt:
            return _to_iso(dt)
        return obj  # normal string → return untouched

    # Firestore / protobuf timestamps
    try:
        from google.protobuf.timestamp_pb2 import Timestamp as ProtoTimestamp

        if isinstance(obj, ProtoTimestamp):
            return _to_iso(obj.ToDatetime())
    except Exception:
        pass

    try:
        from google.cloud.firestore_v1._helpers import Timestamp as FSHelperTimestamp

        if isinstance(obj, FSHelperTimestamp):
            return _to_iso(obj.to_datetime())
    except Exception:
        pass

    # Any object with a .to_datetime() / .ToDatetime() method
    for fn in ("to_datetime", "ToDatetime"):
        if hasattr(obj, fn) and callable(getattr(obj, fn)):
            try:
                dt = getattr(obj, fn)()
                if isinstance(dt, datetime.datetime):
                    return _to_iso(dt)
            except Exception:
                pass

    return obj


def _maybe_parse_datetime(s: str):
    """Try ISO first, then RFC1123. Return datetime if parseable, else None."""
    if not isinstance(s, str):
        return None

    # Try ISO 8601
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass

    # Try RFC1123 / email dates (e.g., "Mon, 01 Jan 0001 00:00:00 GMT")
    try:
        return parsedate_to_datetime(s)
    except Exception:
        pass

    return None


def normalize_datetime(obj: Any) -> datetime.datetime:
    return datetime.datetime.fromisoformat(convert_timestamps(obj))


def _ensure_aware_utc(dt: datetime.datetime, assume_naive_as_utc: bool = True):
    """Return a timezone-aware UTC datetime. If naive and assume_naive_as_utc is True,
    treat it as UTC (replace tzinfo). If naive and assume_naive_as_utc is False,
    interpret it as local time and convert to UTC.
    """
    if dt.tzinfo is None:
        if assume_naive_as_utc:
            return dt.replace(tzinfo=datetime.timezone.utc)
        else:
            # interpret naive as local, convert to UTC
            return dt.astimezone(datetime.timezone.utc)
    # already aware -> convert to UTC
    return dt.astimezone(datetime.timezone.utc)


def datetimes_close(a: datetime.datetime, b: datetime.datetime) -> bool:
    return abs(a - b) <= datetime.timedelta(milliseconds=1)


def datetime_to_timestamp(
    dt: datetime.datetime,
    min_ts: float = MIN_TS,
    max_ts: float = MAX_TS,
) -> float:
    """
    Convert a datetime -> timestamp (seconds since epoch, float), safely handling
    datetime.min/max by mapping to numeric sentinels.

    - naive datetimes are treated as UTC if assume_naive_as_utc is True.
    - returns a finite float (clamped to min_ts/max_ts on overflow).
    """
    if dt == datetime.datetime.min:
        return float(min_ts)
    if dt == datetime.datetime.max:
        return float(max_ts)

    try:
        ts = float(dt.timestamp())
    except (OverflowError, OSError, ValueError):
        # fallback: map very-small/very-large datetimes to sentinels
        return float(min_ts) if dt < EPOCH else float(max_ts)

    # clamp to safe numeric bounds if necessary
    if ts <= min_ts:
        return float(min_ts)
    if ts >= max_ts:
        return float(max_ts)
    return ts


def _normalize_input_ts_to_seconds(ts: float) -> float:
    """
    Heuristic to convert an input numeric ts into seconds.
    Accepts seconds, milliseconds, microseconds (and negative equivalents).
    Rules:
      - if abs(ts) > 1e14 -> treat as microseconds (divide by 1e6)
      - elif abs(ts) > 1e11 -> treat as milliseconds (divide by 1e3)
      - else -> treat as seconds
    These cutoffs are conservative and work for typical epoch values.
    """
    abs_ts = abs(ts)
    if abs_ts > 1e14:
        return ts / 1e6  # microseconds -> seconds
    if abs_ts > 1e11:
        return ts / 1e3  # milliseconds -> seconds
    return ts  # assume seconds


def timestamp_to_datetime(
    ts: Union[int, float], min_ts: float = MIN_TS, max_ts: float = MAX_TS
) -> datetime.datetime:
    """
    Convert numeric timestamp -> timezone-aware UTC datetime.
    Handles sentinel bounds (min_ts/max_ts) and heuristically detects
    milliseconds/microseconds and converts them to seconds.
    """
    ts = float(ts)

    if ts <= min_ts:
        return datetime.datetime.min
    if ts >= max_ts:
        return datetime.datetime.max

    seconds = _normalize_input_ts_to_seconds(ts)

    # final guard: if the seconds value is still outside reasonable float range, clamp
    if seconds <= -1e18:
        return datetime.datetime.min
    if seconds >= 1e18:
        return datetime.datetime.max

    # fromtimestamp raises OSError on some platforms for out-of-range seconds
    try:
        return datetime.datetime.utcfromtimestamp(seconds)
    except (OverflowError, OSError, ValueError):
        # fallback to sentinels
        return datetime.datetime.min if seconds < 0 else datetime.datetime.max
