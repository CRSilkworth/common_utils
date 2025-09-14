import datetime as dt

EPOCH = dt.datetime(1970, 1, 1)


def datetime_to_float(d: dt.datetime) -> float:
    """Convert datetime to float seconds since epoch, works for min/max."""
    delta = d - EPOCH
    return delta.total_seconds()


def float_to_datetime(value: float) -> dt.datetime:
    """Convert float seconds since epoch back to datetime."""
    return EPOCH + dt.timedelta(seconds=value)
