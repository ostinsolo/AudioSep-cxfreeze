# AudioSep Benchmarks (Standalone cx_Freeze)

This file tracks **AudioSep-only** benchmarks for the standalone runtime in this repo.
It is independent from the parent DSU benchmarks.

## Test Inputs

Use consistent audio files for repeatable timing:

- Short: `tests/audio/test_4s.wav` (4 seconds)
- Long: `tests/audio/test_40s.wav` (40 seconds)

If you use different files, record them below with duration.

## Benchmark Method

We measure **cold** and **warm** timings.

- **Cold**: First run after starting the worker (includes model load).
- **Warm**: Second run with the same worker and cached model.

### Worker Mode (Frozen)

1. Start the worker:
   ```bash
   ./dist/dsu-audiosep/dsu-audiosep --worker
   ```

2. Send JSON commands (stdin), capture elapsed:
   ```bash
   printf '%s\n' \
     '{"cmd":"load_model","config_path":"config/audiosep_base.yaml","checkpoint_path":"/path/to/audiosep_base_4M_steps.ckpt","clap_checkpoint_path":"/path/to/music_speech_audioset_epoch_15_esc_89.98.pt","roberta_dir":"/path/to/roberta-base"}' \
     '{"cmd":"separate","input":"/path/to/test_4s.wav","output":"/tmp/audiosep_4s.wav","text":"vocals","use_chunk":true}' \
     '{"cmd":"separate","input":"/path/to/test_4s.wav","output":"/tmp/audiosep_4s_warm.wav","text":"vocals","use_chunk":true}' \
     '{"cmd":"exit"}' \
     | ./dist/dsu-audiosep/dsu-audiosep --worker
   ```

Use the `elapsed` field from the worker output as the timing.

### Python Environment (Pre-freeze)

Use the same model + inputs, run `pipeline.py` or a small script to measure:

```bash
python - <<'PY'
import time
import torch
from pipeline import build_audiosep, separate_audio

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = build_audiosep("config/audiosep_base.yaml", "checkpoint/audiosep_base_4M_steps.ckpt", device)

audio = "/path/to/test_4s.wav"
out = "/tmp/audiosep_4s_py.wav"
start = time.time()
separate_audio(model, audio, "vocals", out, device=device, use_chunk=True)
print("elapsed:", round(time.time() - start, 2))
PY
```

## Results (Fill In)

Record results in seconds. Include device, model, and file duration.

| Date | Device | Mode | Model | File | Duration | Cold (s) | Warm (s) | Notes |
|------|--------|------|-------|------|----------|----------|----------|-------|
| 2026-02-03 | mps | worker/frozen | audiosep_base | 15_1_29_2_2_2026_.wav | ? | 0.73 | 0.10 | Local frozen build, use_chunk=true |
| 2026-02-03 | mps | worker/frozen | audiosep_base | 15_1_29_2_2_2026_.wav | ? | 0.73 | 0.10 | Downloaded build failed: `NameError: os is not defined` in `open_clip/model.py` |
|      | cpu    | worker/frozen | audiosep_base | test_4s.wav | 4s |  |  |  |
|      | cuda   | worker/frozen | audiosep_base | test_4s.wav | 4s |  |  |  |

## MPS Optimizations (Ideas to Evaluate)

We are on **PyTorch 2.10**. After baseline benchmarks, evaluate:

- Native MPS STFT/ISTFT usage (reduce CPU fallbacks).
- Autocast behavior on MPS (verify no unintended slow path).
- Chunk sizes that balance memory vs throughput.

Document any change and re-run the table above.
