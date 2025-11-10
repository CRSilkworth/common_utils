from typing import Any
import datetime
from email.utils import parsedate_to_datetime

EPOCH = datetime.datetime(1970, 1, 1)

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
    return dt.isoformat()


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
