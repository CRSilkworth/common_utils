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
        print("here")
        for val in gens[0]:
            print(val)
            yield val
    sentinel = object()
    iterators = [iter(g) for g in gens]

    while True:
        values = [next(it, sentinel) for it in iterators]

        if all(v is sentinel for v in values):
            break  # all done
        if any(v is sentinel for v in values):
            raise ValueError(
                "Iterables have different lengths. Did all the upstream nodes "
                "get run?"
            )

        yield tuple(values)


def generator_from_urls(signed_urls: List[Text], value_type: Any = Allowed):
    value = read_from_gcs_signed_urls(signed_urls) if signed_urls else None

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
        for value in sim_value_gen:
            yield value
