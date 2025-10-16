import io
import pstats
import cProfile


def profile_function(filename=None):
    def decorator(f):
        def wrapped(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                return f(*args, **kwargs)
            finally:
                profiler.disable()
                if filename:
                    profiler.dump_stats(filename)
                else:
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                    ps.print_stats(20)
                    print(s.getvalue())

        return wrapped

    return decorator
