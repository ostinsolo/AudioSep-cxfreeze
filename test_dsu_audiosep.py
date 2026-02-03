#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time

import soundfile as sf


def read_json_line(proc, timeout_s=120, log_lines=None):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return None
            continue
        line = line.strip()
        if log_lines is not None:
            log_lines.append(line)
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return None


def wait_for_status(proc, want, timeout_s=120, log_lines=None):
    want_set = set(want)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        obj = read_json_line(proc, timeout_s=5, log_lines=log_lines)
        if obj is None:
            if proc.poll() is not None:
                return None
            continue
        if obj.get("status") in want_set:
            return obj
    return None


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    exe = os.path.join(project_root, "dist", "dsu-audiosep", "dsu-audiosep")
    if sys.platform == "win32":
        exe += ".exe"

    input_wav = "/Users/ostino/Music/harmonica_audiosep.wav"
    output_wav = os.path.join(project_root, "test_output", "dsu_audiosep_out.wav")
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    if not os.path.exists(exe):
        print(f"ERROR: Executable not found: {exe}")
        return 1
    if not os.path.exists(input_wav):
        print(f"ERROR: Input audio not found: {input_wav}")
        return 1

    proc = subprocess.Popen(
        [exe, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(exe),
        bufsize=1,
    )

    log_lines = []
    try:
        t0 = time.time()
        ready = wait_for_status(proc, want=["ready", "error"], timeout_s=300, log_lines=log_lines)
        if not ready or ready.get("status") != "ready":
            print(f"ERROR: Worker not ready: {ready}")
            if log_lines:
                print("Last output lines:")
                for ln in log_lines[-10:]:
                    print(ln)
            return 1
        startup_s = round(time.time() - t0, 2)
        print(f"Startup time: {startup_s}s")

        load_cmd = {
            "cmd": "load_model",
            "config_path": os.path.join(project_root, "config", "audiosep_base.yaml"),
            "checkpoint_path": os.path.join(project_root, "checkpoint", "audiosep_base_4M_steps.ckpt"),
            "clap_checkpoint_path": os.path.join(project_root, "checkpoint", "music_speech_audioset_epoch_15_esc_89.98.pt"),
            "roberta_dir": os.path.join(project_root, "roberta-base"),
        }
        proc.stdin.write(json.dumps(load_cmd) + "\n")
        proc.stdin.flush()
        loaded = wait_for_status(proc, want=["model_loaded", "error"], timeout_s=300, log_lines=log_lines)
        if not loaded or loaded.get("status") != "model_loaded":
            print(f"ERROR: Model load failed: {loaded}")
            return 1

        sep_cmd = {
            "cmd": "separate",
            "input": input_wav,
            "output": output_wav,
            "text": "mouth harmonica",
            "use_chunk": True,
        }
        proc.stdin.write(json.dumps(sep_cmd) + "\n")
        proc.stdin.flush()
        done = wait_for_status(proc, want=["done", "error"], timeout_s=600, log_lines=log_lines)
        if not done or done.get("status") != "done":
            print(f"ERROR: Separation failed: {done}")
            return 1

        if not os.path.exists(output_wav):
            print(f"ERROR: Output not found: {output_wav}")
            return 1

        info = sf.info(output_wav)
        print(f"Output: {output_wav} ({info.samplerate} Hz, {info.frames} frames)")

        proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=10)
        return 0
    finally:
        try:
            proc.kill()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
