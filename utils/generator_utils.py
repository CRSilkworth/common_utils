from typing import Text, Any, List, Iterator, Tuple, Dict, Optional
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


def merge_key_and_data_iterators(
    key_iterator: Iterator[Tuple],
    data_iterator: Iterator[Tuple[Tuple, Dict[str, bytes]]],
    value_file_groups: List[List[str]],
) -> Iterator[Tuple[Tuple, int, Dict[str, Optional[bytes]]]]:
    """
    Merge streamed data with full key iterator, yielding one dict per VF group.
    Ensures every (key, group_idx) is output exactly once. Missing data is None.

    Args:
        key_iterator: iterator of all step keys
        data_iterator: iterator yielding (step, {vf_id: bytes})
        value_file_groups: list of VF id groups, e.g. [["vf_1","vf_2"], ["vf_1","vf_3",
        "vf_4"]]

    Yields:
        (step, group_idx, {vf_id: bytes or None})
    """
    data_key, data_dict = None, {}

    for key in key_iterator:
        if data_key is None or data_key < key:
            try:
                data_key, data_dict = next(data_iterator)
            except StopIteration:
                data_key, data_dict = None, {}

        if data_key == key:
            print("matched", data_key)
            current_data = data_dict
        else:
            current_data = {}

        group_idx = key[-1]
        group = value_file_groups[group_idx]
        out_dict = {vf_id: current_data.get(vf_id) for vf_id in group}
        yield key, out_dict
