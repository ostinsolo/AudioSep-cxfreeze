import argparse
import json
import os
import shlex
import subprocess
import sys
import time


def _default_worker_cmd():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_python = os.path.join(base_dir, "runtime", "bin", "python")
    worker_script = os.path.join(base_dir, "audiosep_worker.py")
    return f"{runtime_python} {worker_script} --worker"


def _read_until(proc, wanted_statuses):
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Worker exited before response")
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = msg.get("status")
        if status in wanted_statuses:
            return msg


def _run_worker_session(cmd, load_job, separate_jobs):
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        _read_until(proc, {"ready"})
        proc.stdin.write(json.dumps(load_job) + "\n")
        proc.stdin.flush()
        _read_until(proc, {"model_loaded"})

        per_job = []
        for job in separate_jobs:
            t0 = time.time()
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
            _read_until(proc, {"done"})
            per_job.append(time.time() - t0)

        proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
        proc.stdin.flush()
        _read_until(proc, {"exiting"})
        return per_job
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark AudioSep worker persistence")
    parser.add_argument("--worker-cmd", type=str, default=None, help="Command to start worker")
    parser.add_argument("--config", required=True, help="AudioSep config path")
    parser.add_argument("--checkpoint", required=True, help="AudioSep checkpoint path")
    parser.add_argument("--clap", required=True, help="CLAP checkpoint path")
    parser.add_argument("--roberta", required=True, help="RoBERTa directory")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input wav files")
    parser.add_argument("--text", default="vocals", help="Text prompt")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--use-chunk", action="store_true", help="Enable chunked inference")
    parser.add_argument("--use-torch-stft", default="false", help="true|false|auto")
    parser.add_argument("--auto-stft-seconds", type=float, default=60.0, help="Auto STFT threshold")
    parser.add_argument("--mmap", action="store_true", help="Enable mmap model loading")
    args = parser.parse_args()

    cmd = shlex.split(args.worker_cmd) if args.worker_cmd else shlex.split(_default_worker_cmd())
    os.makedirs(args.out_dir, exist_ok=True)

    load_job = {
        "cmd": "load_model",
        "config_path": args.config,
        "checkpoint_path": args.checkpoint,
        "clap_checkpoint_path": args.clap,
        "roberta_dir": args.roberta,
        "mmap": bool(args.mmap),
        "use_torch_stft": args.use_torch_stft,
        "auto_stft_seconds": args.auto_stft_seconds,
    }

    separate_jobs = []
    for i, inp in enumerate(args.inputs):
        out_path = os.path.join(args.out_dir, f"out_{i+1}.wav")
        separate_jobs.append({
            "cmd": "separate",
            "input": inp,
            "output": out_path,
            "text": args.text,
            "use_chunk": bool(args.use_chunk),
        })

    # Mode A: spawn per job (load model each time)
    per_job_spawn = []
    t0 = time.time()
    for job in separate_jobs:
        per_job_spawn.extend(_run_worker_session(cmd, load_job, [job]))
    total_spawn = time.time() - t0

    # Mode B: one persistent worker (load once, multiple jobs)
    t1 = time.time()
    per_job_persist = _run_worker_session(cmd, load_job, separate_jobs)
    total_persist = time.time() - t1

    print("=== AudioSep Persistence Benchmark ===")
    print(f"Inputs: {len(separate_jobs)}")
    print(f"Spawn-per-job total: {total_spawn:.2f}s  per-job: {[round(x,2) for x in per_job_spawn]}")
    print(f"Persistent total:    {total_persist:.2f}s  per-job: {[round(x,2) for x in per_job_persist]}")


if __name__ == "__main__":
    main()
