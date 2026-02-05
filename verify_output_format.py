#!/usr/bin/env python3
"""
Minimal regression test: run separation and verify output format.
Checks: stereo (2 channels), 44.1 kHz, valid WAV, duration matches input.

Usage:
  python verify_output_format.py --input /path/to/input.wav [--models /path/to/models]
  Or set env: AUDIOSEP_VERIFY_INPUT, AUDIOSEP_MODELS
"""
import argparse
import json
import os
import subprocess
import sys

try:
    import soundfile as sf
except ImportError:
    print("soundfile required: pip install soundfile")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.environ.get("AUDIOSEP_VERIFY_INPUT"), help="Input WAV path")
    ap.add_argument("--models", default=os.environ.get("AUDIOSEP_MODELS", ""), help="Models dir (for paths)")
    ap.add_argument("--worker", default="python audiosep_worker.py --worker", help="Worker command")
    args = ap.parse_args()

    if not args.input or not os.path.isfile(args.input):
        print("Error: --input required and must exist. Set AUDIOSEP_VERIFY_INPUT or pass --input")
        sys.exit(1)

    models = args.models.rstrip("/") or "/tmp/audiosep_models"
    config = f"{models}/config/audiosep_base.yaml"
    checkpoint = f"{models}/checkpoint/audiosep_base_4M_steps.ckpt"
    clap = f"{models}/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt"
    roberta = f"{models}/roberta-base"
    out_path = "/tmp/audiosep_verify_output.wav"

    if not os.path.isfile(checkpoint):
        print(f"Skip: checkpoint not found at {checkpoint}. Set AUDIOSEP_MODELS to models dir.")
        sys.exit(0)

    cmds = [
        json.dumps({
            "cmd": "load_model",
            "config_path": config,
            "checkpoint_path": checkpoint,
            "clap_checkpoint_path": clap,
            "roberta_dir": roberta,
            "use_torch_stft": "auto",
            "auto_stft_seconds": 60,
            "mmap": False,
        }),
        json.dumps({
            "cmd": "separate",
            "input": args.input,
            "output": out_path,
            "text": "vocals",
            "use_chunk": False,
        }),
        json.dumps({"cmd": "exit"}),
    ]
    proc = subprocess.run(
        args.worker.split(),
        input="\n".join(cmds) + "\n",
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        timeout=120,
    )
    if proc.returncode != 0:
        print("Worker failed:", proc.stderr[-500:] if proc.stderr else proc.stdout[-500:])
        sys.exit(1)

    if not os.path.isfile(out_path):
        print("Output file not created")
        sys.exit(1)

    d_in, sr_in = sf.read(args.input)
    d_out, sr_out = sf.read(out_path)
    dur_in = len(d_in) / sr_in
    dur_out = len(d_out) / sr_out

    ok = True
    if d_out.ndim != 2 or d_out.shape[1] != 2:
        print("FAIL: output not stereo (expected shape (N, 2))")
        ok = False
    else:
        print("PASS: stereo (2 channels)")

    if sr_out != 44100:
        print(f"FAIL: sample rate {sr_out}, expected 44100")
        ok = False
    else:
        print("PASS: 44.1 kHz")

    if abs(dur_in - dur_out) > 0.05:
        print(f"FAIL: duration mismatch in={dur_in:.2f}s out={dur_out:.2f}s")
        ok = False
    else:
        print("PASS: duration matches input")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
