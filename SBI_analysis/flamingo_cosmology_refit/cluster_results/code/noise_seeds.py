"""Stable, domain-separated seeds for each row and each SO noise split.

The row ID, not the worker/chunk/order, controls its noise. Preflight IDs start
at 524288, outside the supported production range. SHA256 is specified explicitly
so Python's randomized hash() and NumPy RNG-version changes cannot alter seeds.
"""
import hashlib

ALGORITHM = "sha256-halfdome-so-v1-first64be-shift1"
PREFLIGHT_OFFSET = 524288


def split_seeds(row_id, master_seed=12345):
    if row_id < 0 or master_seed < 0:
        raise ValueError("Seeds and row IDs must be nonnegative")
    result = []
    for split in (1, 2):
        message = "halfdome-so-v1|%d|%d|%d" % (master_seed, row_id, split)
        result.append(int.from_bytes(hashlib.sha256(message.encode("ascii")).digest()[:8],
                                     "big") >> 1)
    return result
