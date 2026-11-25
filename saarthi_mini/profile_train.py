"""Report wall-clock time and Linux process peak RSS alongside training output."""
import json
import resource
import time

from .train import main


def profile():
    start = time.monotonic()
    try:
        main()
    finally:
        print(json.dumps({
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }), flush=True)


if __name__ == "__main__":
    profile()
