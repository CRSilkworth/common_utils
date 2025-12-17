from typing import Any, Union, Tuple
from utils.type_utils import TimeRanges
import datetime
import calendar
from email.utils import parsedate_to_datetime


UTC = datetime.timezone.utc
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=UTC)

MIN_TS = -1e18
MAX_TS = 1e18


def ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    """
    Ensure a datetime is timezone-aware UTC.
    - Naive datetimes are assumed to be UTC
    - Aware datetimes are converted to UTC
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_milli(dt: datetime.datetime) -> datetime.datetime:
    dt = ensure_utc(dt)
    return dt.replace(microsecond=(dt.microsecond // 1000) * 1000)


datetime_min = to_milli(datetime.datetime.min.replace(tzinfo=UTC))
datetime_max = to_milli(datetime.datetime.max.replace(tzinfo=UTC))


def _to_iso(dt: datetime.datetime) -> str:
    dt = ensure_utc(dt)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def generate_time_ranges(
    start: datetime.datetime, end: datetime.datetime, period: str
) -> TimeRanges:

    start = ensure_utc(start)
    end = ensure_utc(end)

    if start > end:
        raise ValueError("start must be before end")

    period = period.lower()
    ranges = []

    current = start

    while current < end:
        if period == "daily":
            next_dt = current + datetime.timedelta(days=1)

        elif period == "weekly":
            next_dt = current + datetime.timedelta(weeks=1)

        elif period == "monthly":
            year = current.year
            month = current.month

            if month == 12:
                next_year, next_month = year + 1, 1
            else:
                next_year, next_month = year, month + 1

            day = min(current.day, calendar.monthrange(next_year, next_month)[1])
            next_dt = current.replace(year=next_year, month=next_month, day=day)

        elif period == "yearly":
            try:
                next_dt = current.replace(year=current.year + 1)
            except ValueError:
                next_dt = current.replace(month=2, day=28, year=current.year + 1)

        else:
            raise ValueError("period must be one of: daily, weekly, monthly, yearly")

        range_end = min(next_dt, end)
        ranges.append((to_milli(current), to_milli(range_end)))

        current = next_dt

    return ranges


def to_isos(obj):
    """
    Recursively convert timestamps/datetimes and datetime-like strings to ISO8601.
    """

    if isinstance(obj, dict):
        return {k: to_isos(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_isos(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_isos(v) for v in obj)

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
    if not isinstance(s, str):
        return None

    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except Exception:
        pass

    try:
        dt = parsedate_to_datetime(s)
        return dt.astimezone(UTC)
    except Exception:
        pass

    return None


def datetimes_close(a: datetime.datetime, b: datetime.datetime) -> bool:
    return abs(a - b) <= datetime.timedelta(milliseconds=1)


def to_datetimes(obj: Any):
    """
    Recursively convert timestamps/datetimes and datetime-like strings
    into datetime.datetime objects (millisecond precision).
    """

    if isinstance(obj, dict):
        return {k: to_datetimes(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_datetimes(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_datetimes(v) for v in obj)

    if isinstance(obj, datetime.datetime):
        return to_milli(obj)

    # Datetime-like strings
    if isinstance(obj, str):
        dt = _maybe_parse_datetime(obj)
        if dt:
            return to_milli(dt)
        return obj

    # Firestore / protobuf timestamps
    try:
        from google.protobuf.timestamp_pb2 import Timestamp as ProtoTimestamp

        if isinstance(obj, ProtoTimestamp):
            return to_milli(obj.ToDatetime())
    except Exception:
        pass

    try:
        from google.cloud.firestore_v1._helpers import Timestamp as FSHelperTimestamp

        if isinstance(obj, FSHelperTimestamp):
            return to_milli(obj.to_datetime())
    except Exception:
        pass

    # Any object with a .to_datetime() / .ToDatetime() method
    for fn in ("to_datetime", "ToDatetime"):
        if hasattr(obj, fn) and callable(getattr(obj, fn)):
            try:
                dt = getattr(obj, fn)()
                if isinstance(dt, datetime.datetime):
                    return to_milli(dt)
            except Exception:
                pass

    return obj


def to_timestamps(
    obj: Any,
    *,
    min_ts: float = MIN_TS,
    max_ts: float = MAX_TS,
):
    """
    Recursively convert datetime-like objects into float timestamps
    (seconds since epoch), preserving container structure.
    """

    if isinstance(obj, dict):
        return {
            k: to_timestamps(v, min_ts=min_ts, max_ts=max_ts) for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [to_timestamps(v, min_ts=min_ts, max_ts=max_ts) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_timestamps(v, min_ts=min_ts, max_ts=max_ts) for v in obj)

    # Already a datetime
    if isinstance(obj, datetime.datetime):
        return datetime_to_timestamp(obj, min_ts=min_ts, max_ts=max_ts)

    # Datetime-like strings
    if isinstance(obj, str):
        dt = _maybe_parse_datetime(obj)
        if dt:
            return datetime_to_timestamp(dt, min_ts=min_ts, max_ts=max_ts)
        return obj

    # Firestore / protobuf timestamps
    try:
        from google.protobuf.timestamp_pb2 import Timestamp as ProtoTimestamp

        if isinstance(obj, ProtoTimestamp):
            return datetime_to_timestamp(obj.ToDatetime(), min_ts=min_ts, max_ts=max_ts)
    except Exception:
        pass

    try:
        from google.cloud.firestore_v1._helpers import Timestamp as FSHelperTimestamp

        if isinstance(obj, FSHelperTimestamp):
            return datetime_to_timestamp(
                obj.to_datetime(), min_ts=min_ts, max_ts=max_ts
            )
    except Exception:
        pass

    # Any object with a .to_datetime() / .ToDatetime() method
    for fn in ("to_datetime", "ToDatetime"):
        if hasattr(obj, fn) and callable(getattr(obj, fn)):
            try:
                dt = getattr(obj, fn)()
                if isinstance(dt, datetime.datetime):
                    return datetime_to_timestamp(dt, min_ts=min_ts, max_ts=max_ts)
            except Exception:
                pass

    return obj


def datetime_to_timestamp(
    dt: datetime.datetime,
    min_ts: float = MIN_TS,
    max_ts: float = MAX_TS,
) -> float:
    dt = ensure_utc(dt)

    if datetimes_close(dt, datetime_min):
        return float(min_ts)
    if datetimes_close(dt, datetime_max):
        return float(max_ts)

    try:
        ts = float(dt.timestamp())
    except Exception:
        return float(min_ts) if dt < EPOCH else float(max_ts)

    return max(min(ts, max_ts), min_ts)


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
    milliseconds/milliseconds and converts them to seconds.
    """
    ts = float(ts)

    if ts <= min_ts:
        return to_milli(datetime_min)
    if ts >= max_ts:
        return to_milli(datetime_max)

    seconds = _normalize_input_ts_to_seconds(ts)

    # final guard: if the seconds value is still outside reasonable float range, clamp
    if seconds <= -1e18:
        return to_milli(datetime_min)
    if seconds >= 1e18:
        return to_milli(datetime_max)

    # fromtimestamp raises OSError on some platforms for out-of-range seconds
    try:
        return to_milli(datetime.datetime.fromtimestamp(seconds, tz=UTC))
    except (OverflowError, OSError, ValueError):
        # fallback to sentinels
        return to_milli(datetime_min) if seconds < 0 else to_milli(datetime_max)
