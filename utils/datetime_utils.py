import datetime

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
