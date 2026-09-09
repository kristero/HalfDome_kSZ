#!/usr/bin/env python3
"""Wait for cluster summary completion, then download the plots and tables.

Uses system ssh/scp, so the same idark credentials work on Windows and Linux.
A failed summary is downloaded as diagnostics and never labelled complete.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time


def remote_status(host, root):
    complete = shlex.quote(str(PurePosixPath(root) / "summary/summary_complete.json"))
    failed = shlex.quote(str(PurePosixPath(root) / "summary/summary_failure.json"))
    command = f"if test -s {complete}; then echo complete; elif test -s {failed}; then echo failed; else echo pending; fi"
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                             host, command], capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="idark")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=180)
    parser.add_argument("--timeout-hours", type=float, default=168)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.timeout_hours <= 0:
        raise ValueError("Polling interval and timeout must be positive")
    # Keep scp paths literal across different OpenSSH versions.
    if not args.remote_root.startswith("/") or any(c not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-." for c in args.remote_root):
        raise ValueError("Remote root must be an absolute path without shell metacharacters")
    args.local_root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_hours * 3600
    while True:
        try:
            status = remote_status(args.host, args.remote_root)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), status, flush=True)
            if status in ("complete", "failed"):
                remote = f"{args.host}:{args.remote_root}"
                for name in ("summary", "logs", "experiment.json", "rerun_provenance.json", "submission_jobs.tsv"):
                    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                    "-r", f"{remote}/{name}", str(args.local_root)], check=True, timeout=900)
                for method in ("bins40", "pca", "moped"):
                    local = args.local_root / method
                    local.mkdir(exist_ok=True)
                    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                    f"{remote}/{method}/training_complete.json", str(local)],
                                   check=(status == "complete"), timeout=120)
                if status == "complete":
                    marker = args.local_root / "summary/summary_complete.json"
                    config = json.loads((args.local_root / "experiment.json").read_text())
                    if json.loads(marker.read_text())["experiment_id"] != config["experiment_id"]:
                        raise ValueError("Downloaded summary belongs to a different experiment")
                    if not list((args.local_root / "summary").glob("*.png")):
                        raise ValueError("Summary completed but no PNG plots were downloaded")
                receipt = dict(status=status, remote_root=args.remote_root,
                               downloaded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                (args.local_root / "download_status.json").write_text(json.dumps(receipt, indent=2))
                print(f"Downloaded {status} summary to {args.local_root}", flush=True)
                return 0 if status == "complete" else 2
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(f"Connection/transfer issue; will retry: {error}", flush=True)
        if args.once:
            return 3
        if time.monotonic() >= deadline:
            raise TimeoutError("Summary was not downloaded before the deadline; rerun this downloader")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
