def strict_zip(*gens):
    """
    Zip an arbitrary number of iterables together, raising ValueError
    if they don't all have the same length.
    """
    sentinel = object()
    iterators = [iter(g) for g in gens]

    while True:
        values = [next(it, sentinel) for it in iterators]

        if all(v is sentinel for v in values):
            break  # all done
        if any(v is sentinel for v in values):
            raise ValueError("Iterables have different lengths")

        yield tuple(values)
