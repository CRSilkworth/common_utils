from typing import Text, Any, List
from itertools import groupby
from operator import itemgetter
from utils.type_utils import Allowed
from utils.serialize_utils import attempt_deserialize
from utils.gcp_utils import read_from_gcs_signed_urls


def strict_zip(*gens):
    """
    Zip an arbitrary number of iterables together, raising ValueError
    if they don't all have the same length.
    """
    if len(gens) == 1:
        for val in gens[0]:
            yield val
    sentinel = object()
    generators = [iter(g) for g in gens]
    while True:
        values = [next(it, sentinel) for it in generators]
        if all(v is sentinel for v in values):
            break  # all done
        if any(v is sentinel for v in values):
            raise ValueError(
                "Iterables have different lengths. Did all the upstream nodes "
                "get run?"
            )

        yield tuple(values)


def generator_from_urls(signed_urls: List[Text], value_type: Any = Allowed):
    value = read_from_gcs_signed_urls(signed_urls) if signed_urls else []

    for file_content in value:
        file_content, output, _ = attempt_deserialize(file_content, value_type)
        if output:
            raise ValueError(
                "failed to deserialize file from generator: "
                f"{output['stderr_output']}"
            )
        for item in file_content:
            yield item


def sim_values_generator(sim_value_gen, chunked: bool = False):
    """
    Convert a generator with values of the form (sim_parms, value)  to a generator
    of a generator (if chunked) where the outer generator  goes over the sim_params
    and the inner over the values with the same sim_params
    """
    if chunked:
        for key, group in groupby(sim_value_gen, key=itemgetter(0)):
            yield (value for _, value in group)
    else:
        for key, value in sim_value_gen:
            yield value


def merge_generators(generators):
    """
    Merge several flat generators keyed by
    (sim_param_key, collection_name, time_range).

    Yields:
        (key, [val_from_iter1, val_from_iter2, ...])
    where missing values are None.
    """
    # Flatten each generator
    streams = []
    for i, it in enumerate(generators):
        try:
            key, val = next(it)
            streams.append((key, i, val, it))
        except StopIteration:
            continue

    while streams:
        # find smallest key across all active streams
        min_key = min(s[0] for s in streams)

        row = [None] * len(generators)
        new_streams = []
        for key, idx, val, stream in streams:
            if key == min_key:
                row[idx] = val
                try:
                    nxt_key, nxt_val = next(stream)
                    new_streams.append((nxt_key, idx, nxt_val, stream))
                except StopIteration:
                    pass
            else:
                new_streams.append((key, idx, val, stream))

        yield min_key, row
        streams = new_streams
