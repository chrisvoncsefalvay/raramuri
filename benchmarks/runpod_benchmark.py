#!/usr/bin/env python3
"""RunPod Serverless benchmark for Rarámuri phase-wise inference timing.

Runs inference on a target video via RunPod's sync API, collects
per-phase timing, warm-up data, and GPU utilisation, then prints a
formatted report and optionally writes Prometheus metrics to a file.

Usage:
    # Set env vars:
    export RUNPOD_API_KEY="rp_xxxxxxxx"
    export RUNPOD_ENDPOINT_ID="your-endpoint-id"

    # Single run (cold start):
    python benchmarks/runpod_benchmark.py

    # Multiple runs to capture warm-start deltas:
    python benchmarks/runpod_benchmark.py --runs 3

    # Custom video:
    python benchmarks/runpod_benchmark.py --video-url "https://youtu.be/..."

    # Write Prometheus metrics to file:
    python benchmarks/runpod_benchmark.py --metrics-out benchmark_metrics.prom
"""

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required.  pip install requests", file=sys.stderr)
    sys.exit(1)

DEFAULT_VIDEO_URL = "https://youtu.be/fU0uiGAm_SY"


def call_runsync(endpoint_id: str, api_key: str, payload: dict, timeout: int = 600) -> dict:
    """Call RunPod runsync endpoint and return the response."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"input": payload}

    resp = requests.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def call_run_async(endpoint_id: str, api_key: str, payload: dict) -> str:
    """Submit an async job and return the job ID."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"input": payload}

    resp = requests.post(url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["id"]


def poll_status(endpoint_id: str, api_key: str, job_id: str, poll_interval: float = 5.0, timeout: int = 600) -> dict:
    """Poll an async job until completion."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")

        if status == "COMPLETED":
            return data
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"Job {job_id} ended with status: {status}", file=sys.stderr)
            print(json.dumps(data, indent=2), file=sys.stderr)
            sys.exit(1)

        print(f"  [{job_id}] status={status} ...", flush=True)
        time.sleep(poll_interval)

    print(f"Timed out waiting for job {job_id}", file=sys.stderr)
    sys.exit(1)


def format_phase_table(phase_timings: dict, total_seconds: float) -> str:
    """Format phase timings as an ASCII table."""
    if not phase_timings:
        return "  (no phase timings available)"

    # Determine column widths.
    phases = [(k, v) for k, v in phase_timings.items() if isinstance(v, (int, float))]
    if not phases:
        return "  (no numeric phase timings)"

    max_name = max(len(p[0]) for p in phases)
    max_name = max(max_name, 5)  # "Phase"

    lines = []
    header = f"  {'Phase':<{max_name}}  {'Seconds':>8}  {'% Total':>8}  Bar"
    lines.append(header)
    lines.append("  " + "-" * (max_name + 30))

    for name, secs in phases:
        pct = (secs / total_seconds * 100) if total_seconds > 0 else 0
        bar_len = int(pct / 2)
        bar = "#" * bar_len
        lines.append(f"  {name:<{max_name}}  {secs:>8.3f}  {pct:>7.1f}%  {bar}")

    lines.append("  " + "-" * (max_name + 30))
    lines.append(f"  {'TOTAL':<{max_name}}  {total_seconds:>8.3f}  {100.0:>7.1f}%")

    return "\n".join(lines)


def format_warmup_table(warmup: dict) -> str:
    """Format warm-up timings."""
    if not warmup:
        return "  (no warm-up data)"

    total = warmup.get("total_seconds", 0)
    phases = warmup.get("phases", {})

    lines = []
    lines.append(f"  Total warm-up: {total:.3f}s")
    for name, secs in phases.items():
        lines.append(f"    {name}: {secs:.3f}s")

    return "\n".join(lines)


def build_prometheus_output(run_results: list[dict], video_url: str) -> str:
    """Build Prometheus text exposition from all benchmark runs."""
    lines = []
    lines.append("# HELP raramuri_benchmark_runs_total Number of benchmark runs completed.")
    lines.append("# TYPE raramuri_benchmark_runs_total counter")
    lines.append(f"raramuri_benchmark_runs_total {len(run_results)}")
    lines.append("")

    for i, run in enumerate(run_results):
        timing = run.get("timing", {})
        phase_timings = timing.get("phases", {})
        warmup = timing.get("warmup", {})
        warm_start = timing.get("warm_start", False)
        total = timing.get("total_seconds", 0)
        e2e = run.get("_e2e_seconds", 0)

        labels = f'run="{i}",warm_start="{warm_start}",video="{video_url}"'

        lines.append(f"# run {i}: warm_start={warm_start}")
        lines.append(f'raramuri_benchmark_e2e_seconds{{{labels}}} {e2e}')
        lines.append(f'raramuri_benchmark_inference_seconds{{{labels}}} {total}')

        for phase, secs in phase_timings.items():
            if isinstance(secs, (int, float)):
                lines.append(f'raramuri_benchmark_phase_seconds{{{labels},phase="{phase}"}} {secs}')

        warmup_total = warmup.get("total_seconds", 0)
        lines.append(f'raramuri_benchmark_warmup_total_seconds{{{labels}}} {warmup_total}')
        for phase, secs in warmup.get("phases", {}).items():
            lines.append(f'raramuri_benchmark_warmup_phase_seconds{{{labels},phase="{phase}"}} {secs}')

        lines.append("")

    # Aggregate: compute mean warm-start inference time (excluding run 0 if cold).
    warm_runs = [r for r in run_results if r.get("timing", {}).get("warm_start")]
    if warm_runs:
        mean_total = sum(r["timing"]["total_seconds"] for r in warm_runs) / len(warm_runs)
        mean_e2e = sum(r["_e2e_seconds"] for r in warm_runs) / len(warm_runs)
        lines.append("# HELP raramuri_benchmark_warm_mean_inference_seconds Mean inference time across warm-start runs.")
        lines.append("# TYPE raramuri_benchmark_warm_mean_inference_seconds gauge")
        lines.append(f"raramuri_benchmark_warm_mean_inference_seconds {mean_total:.3f}")
        lines.append(f"raramuri_benchmark_warm_mean_e2e_seconds {mean_e2e:.3f}")

        # Per-phase means across warm runs.
        phase_names = set()
        for r in warm_runs:
            phase_names.update(r.get("timing", {}).get("phases", {}).keys())
        lines.append("# HELP raramuri_benchmark_warm_mean_phase_seconds Mean phase time across warm-start runs.")
        lines.append("# TYPE raramuri_benchmark_warm_mean_phase_seconds gauge")
        for pname in sorted(phase_names):
            vals = [r["timing"]["phases"].get(pname, 0) for r in warm_runs if isinstance(r["timing"]["phases"].get(pname), (int, float))]
            if vals:
                mean_val = sum(vals) / len(vals)
                lines.append(f'raramuri_benchmark_warm_mean_phase_seconds{{phase="{pname}"}} {mean_val:.3f}')

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="RunPod Serverless benchmark for Rarámuri")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"),
                        help="RunPod endpoint ID (or RUNPOD_ENDPOINT_ID env)")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"),
                        help="RunPod API key (or RUNPOD_API_KEY env)")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL,
                        help=f"Video URL to benchmark (default: {DEFAULT_VIDEO_URL})")
    parser.add_argument("--start-time", default=None, help="Start time (HH:MM:SS)")
    parser.add_argument("--end-time", default=None, help="End time (HH:MM:SS)")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs (default: 1)")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout in seconds (default: 600)")
    parser.add_argument("--async-mode", action="store_true", help="Use async run+poll instead of runsync")
    parser.add_argument("--metrics-out", default=None, help="Write Prometheus metrics to this file")
    parser.add_argument("--json-out", default=None, help="Write raw results to this JSON file")
    parser.add_argument("--no-predictions", action="store_true",
                        help="Exclude raw predictions from response (lighter payload)")

    args = parser.parse_args()

    if not args.endpoint_id:
        print("ERROR: --endpoint-id or RUNPOD_ENDPOINT_ID required", file=sys.stderr)
        sys.exit(1)
    if not args.api_key:
        print("ERROR: --api-key or RUNPOD_API_KEY required", file=sys.stderr)
        sys.exit(1)

    payload = {
        "video_url": args.video_url,
        "include_predictions": not args.no_predictions,
    }
    if args.start_time:
        payload["start_time"] = args.start_time
    if args.end_time:
        payload["end_time"] = args.end_time

    print(f"Rarámuri RunPod Serverless Benchmark")
    print(f"====================================")
    print(f"Endpoint:  {args.endpoint_id}")
    print(f"Video:     {args.video_url}")
    print(f"Runs:      {args.runs}")
    print(f"Mode:      {'async' if args.async_mode else 'runsync'}")
    print()

    run_results = []

    for run_idx in range(args.runs):
        print(f"--- Run {run_idx + 1}/{args.runs} ---")
        e2e_start = time.monotonic()

        if args.async_mode:
            print("  Submitting async job...", flush=True)
            job_id = call_run_async(args.endpoint_id, args.api_key, payload)
            print(f"  Job ID: {job_id}", flush=True)
            response = poll_status(args.endpoint_id, args.api_key, job_id,
                                   timeout=args.timeout)
            output = response.get("output", {})
        else:
            print("  Calling runsync...", flush=True)
            response = call_runsync(args.endpoint_id, args.api_key, payload,
                                    timeout=args.timeout)
            output = response.get("output", {})

        e2e_seconds = round(time.monotonic() - e2e_start, 3)

        if "error" in output:
            print(f"  ERROR: {output['error']}", file=sys.stderr)
            continue

        timing = output.get("timing", {})
        phase_timings = timing.get("phases", {})
        total_seconds = timing.get("total_seconds", 0)
        warmup = timing.get("warmup", {})
        warm_start = timing.get("warm_start", False)
        shape = output.get("shape", [])

        output["_e2e_seconds"] = e2e_seconds
        run_results.append(output)

        print(f"  Status:     {'WARM start' if warm_start else 'COLD start'}")
        print(f"  Shape:      {shape}")
        print(f"  E2E time:   {e2e_seconds:.3f}s (includes network + RunPod overhead)")
        print(f"  Inference:  {total_seconds:.3f}s")
        print()

        print("  Warm-up:")
        print(format_warmup_table(warmup))
        print()

        print("  Phase Breakdown:")
        print(format_phase_table(phase_timings, total_seconds))
        print()

        # Print Prometheus metrics text if returned by handler.
        metrics_text = output.get("metrics_text")
        if metrics_text:
            print("  Prometheus Metrics (from handler):")
            for line in metrics_text.strip().split("\n"):
                print(f"    {line}")
            print()

    if not run_results:
        print("No successful runs.", file=sys.stderr)
        sys.exit(1)

    # Summary across runs.
    if len(run_results) > 1:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        cold_runs = [r for r in run_results if not r.get("timing", {}).get("warm_start")]
        warm_runs = [r for r in run_results if r.get("timing", {}).get("warm_start")]

        if cold_runs:
            cold_total = cold_runs[0]["timing"]["total_seconds"]
            cold_e2e = cold_runs[0]["_e2e_seconds"]
            print(f"  Cold start:  inference={cold_total:.3f}s  e2e={cold_e2e:.3f}s")

        if warm_runs:
            totals = [r["timing"]["total_seconds"] for r in warm_runs]
            e2es = [r["_e2e_seconds"] for r in warm_runs]
            mean_t = sum(totals) / len(totals)
            mean_e = sum(e2es) / len(e2es)
            min_t = min(totals)
            max_t = max(totals)
            print(f"  Warm starts: n={len(warm_runs)}  mean={mean_t:.3f}s  min={min_t:.3f}s  max={max_t:.3f}s  e2e_mean={mean_e:.3f}s")

            # Per-phase warm means.
            phase_names = set()
            for r in warm_runs:
                phase_names.update(r.get("timing", {}).get("phases", {}).keys())
            if phase_names:
                print("\n  Warm-start phase means:")
                for pname in sorted(phase_names):
                    vals = [r["timing"]["phases"].get(pname, 0) for r in warm_runs
                            if isinstance(r["timing"]["phases"].get(pname), (int, float))]
                    if vals:
                        mean_v = sum(vals) / len(vals)
                        print(f"    {pname}: {mean_v:.3f}s")

        print()

    # Write outputs.
    if args.metrics_out:
        prom_text = build_prometheus_output(run_results, args.video_url)
        with open(args.metrics_out, "w") as f:
            f.write(prom_text)
        print(f"Prometheus metrics written to: {args.metrics_out}")

    if args.json_out:
        # Strip large arrays for the JSON output.
        slim_results = []
        for r in run_results:
            slim = {k: v for k, v in r.items() if k not in ("predictions", "spectrum", "metrics")}
            slim_results.append(slim)
        with open(args.json_out, "w") as f:
            json.dump(slim_results, f, indent=2)
        print(f"JSON results written to: {args.json_out}")


if __name__ == "__main__":
    main()
