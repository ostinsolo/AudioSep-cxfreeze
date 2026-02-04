# AudioSep Benchmarks (Standalone cx_Freeze)

This file tracks **AudioSep-only** benchmarks for the standalone runtime in this repo.
It is independent from the parent DSU benchmarks.

## 🏆 Benchmark Summary

┌──────────────────────────────────────────────────────────────┐
│ 🎧 AudioSep Benchmarks — Snapshot                            │
├──────────────────────────────────────────────────────────────┤
│ 🧠 Load cost dominates cold starts (CLAP + model + imports).  │
│ ⚡ Warm runs are fast when cache hits are used.              │
│ 🔥 CUDA STFT on long files can OOM without chunking.         │
│ 🧩 Auto chunking fixes CUDA OOM on long files.               │
│ 🧪 UV vs pip inference speed is effectively the same.        │
└──────────────────────────────────────────────────────────────┘

### Quick Highlights

- 🧊 **Cold vs warm:** cold time is mostly model load; warm is much faster.
- 🚀 **CUDA:** best gains after warm-up; long files need chunking.
- 🧩 **Auto chunk:** avoids OOM on long CUDA STFT runs.
- 🧪 **UV vs pip:** install speed differs, runtime speed is similar.

### 🏅 Winning Methods (so far)

- 🥇 **Windows CUDA long file (43s), STFT true + auto chunk:** cold 3.52s, warm 1.15s
- 🥈 **Windows CUDA small file (0.98s), STFT true:** warm 0.06s (pip or uv)
- 🥉 **MPS runtime env, baseline + mmap:** lower load time than baseline (2.40s vs 2.79s)

### 📊 Architecture Overview (best known config)

| Arch | Best Mode | Key Settings | Status |
|------|-----------|--------------|--------|
| Windows CUDA | STFT true + auto chunk | `use_torch_stft=true`, `use_chunk="auto"`, `chunk_seconds=30` | ✅ Stable |
| macOS MPS | STFT auto + mmap | `use_torch_stft=auto`, `mmap=true`, `use_chunk=false` | ✅ Stable |
| Intel (CPU) | TBD | TBD | ⏳ Placeholder |

### 🧱 Cold / Warm Timing Matrix (by arch)

| Arch | Mode | File | pip cold (s) | pip warm (s) | uv cold (s) | uv warm (s) | Notes |
|------|------|------|--------------|--------------|-------------|-------------|-------|
| Windows CUDA | STFT true + auto chunk | `0_52_50_1_29_2026_.wav` (~43s) | **3.52** | **1.15** | — | — | Stable, no OOM (pip only so far) |
| Windows CUDA | STFT true | `22_9_4_1_28_2026_.wav` (~0.98s) | 6.78 | 0.06 | 7.10 | 0.06 | Small-file STFT |
| macOS MPS | worker/frozen | `15_1_29_2_2_2026_.wav` | 0.45 | 0.10 | — | — | Local frozen build, use_chunk=true |
| macOS MPS | worker/frozen | `harmonica_audiosep.wav` | 0.48 | 0.09 | — | — | Local frozen build, use_chunk=true |
| macOS MPS | worker/frozen | `1_28_46_1_31_2026_.wav` | 0.84 | 0.51 | — | — | Local frozen build, use_chunk=true |
| Intel (CPU) | TBD | TBD | TBD | TBD | TBD | TBD | Placeholder |

### ⚡ Speedups (Δ / ×)

| Scenario | From → To | Speedup |
|----------|-----------|---------|
| CUDA long file (43s) | no chunk → auto chunk (cold) | **~x9.1** (32.16s → 3.52s) |
| CUDA long file (43s) | cold → warm (auto chunk) | **~x3.1** (3.52s → 1.15s) |
| CUDA small file (0.98s) | cold → warm (pip) | **~x113** (6.78s → 0.06s) |
| CUDA small file (0.98s) | cold → warm (uv) | **~x118** (7.10s → 0.06s) |
| MPS load | baseline → mmap (load) | **~x1.16** (2.79s → 2.40s) |
| MPS long (78s) | baseline → MPS STFT | **~x1.4** (2.16s → 1.54s) |
| MPS long (160s) | baseline → MPS STFT | **~x3.2** (11.24s → 3.52s) |
| MPS persistence | spawn → persistent | **~x1.6** (12.04s → 7.47s) |

### 📏 Real‑time Factor (audio sec / processing sec)

Computed from entries with known audio duration only.

| Scenario | Duration | Time | RTF (x real‑time) |
|----------|----------|------|------------------|
| CUDA long, STFT true (pip, cold) | ~43.11s | 32.16s | **~1.34x** |
| CUDA long, STFT true (uv, cold) | ~43.11s | 115.87s | **~0.37x** |
| CUDA long, STFT true + auto chunk (cold) | ~43.11s | 3.52s | **~12.25x** |
| CUDA long, STFT true + auto chunk (warm) | ~43.11s | 1.15s | **~37.49x** |
| CUDA small, STFT true (pip, cold) | ~0.98s | 6.78s | **~0.14x** |
| CUDA small, STFT true (pip, warm) | ~0.98s | 0.06s | **~16.33x** |
| CUDA small, STFT true (uv, cold) | ~0.98s | 7.10s | **~0.14x** |
| CUDA small, STFT true (uv, warm) | ~0.98s | 0.06s | **~16.33x** |

### 📈 Quick Charts (lower is better)

Cold/Warm (CUDA, long file ~43s, STFT true + auto chunk)

```
Cold  ████████████ 3.52s
Warm  ████         1.15s
```

Persistent vs Spawn (CUDA, 3 inputs)

```
Spawn   ████████████████████████████████████████ 45.60s
Persist ███████████                              17.02s
```

## Test Inputs

Use consistent audio files for repeatable timing:

- Short: `tests/audio/test_4s.wav` (4 seconds)
- Long: `tests/audio/test_40s.wav` (40 seconds)

If you use different files, record them below with duration.

Windows CUDA inputs (size-based, Documents folder):
- `22_9_4_1_28_2026_.wav` (~169 KB)
- `17_24_54_1_27_2026_.wav` (~543 KB)
- `0_52_50_1_29_2026_.wav` (~7.25 MB)

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
| 2026-02-03 | mps | worker/frozen | audiosep_base | 15_1_29_2_2_2026_.wav | ? | 0.45 | 0.10 | Local frozen build, use_chunk=true |
| 2026-02-03 | mps | worker/frozen | audiosep_base | 15_1_29_2_2_2026_.wav | ? | 0.73 | 0.10 | Downloaded build failed: `NameError: os is not defined` in `open_clip/model.py` |
| 2026-02-03 | mps | worker/frozen | audiosep_base | harmonica_audiosep.wav | ? | 0.48 | 0.09 | Local frozen build, use_chunk=true |
| 2026-02-03 | mps | worker/frozen | audiosep_base | 1_28_46_1_31_2026_.wav | ? | 0.84 | 0.51 | Local frozen build, use_chunk=true |

### Windows CUDA (RTX 3070 Laptop GPU)

Worker: `audiosep_worker.py` (venv_cuda)  
Mode: `use_torch_stft=auto`, `auto_stft_seconds=60`, `use_chunk=false`  
Inputs (size-based): `22_9_4_1_28_2026_.wav`, `17_24_54_1_27_2026_.wav`, `0_52_50_1_29_2026_.wav`

| Variant | Total (s) | Per-job (s) | Notes |
|---------|-----------|-------------|-------|
| Spawn-per-job | 45.60 | 2.37, 2.40, 3.29 | Load model each run |
| Persistent worker | 17.02 | 2.45, 0.10, 1.01 | Single long-lived process |

#### CUDA STFT (small file, cold vs warm)

Input: `22_9_4_1_28_2026_.wav` (~0.98s)  
Mode: `use_torch_stft=true`, `use_chunk=false`, `device=cuda`

| Env | Cold (s) | Warm (s) | Notes |
|-----|----------|----------|-------|
| pip venv (`venv_cuda`) | 6.78 | 0.06 | `expandable_segments` unsupported |
| uv venv (`venv_uv_cuda`) | 7.10 | 0.06 | `expandable_segments` unsupported |

#### CUDA STFT (large file, warm OOM)

Input: `0_52_50_1_29_2026_.wav` (~43.11s)  
Mode: `use_torch_stft=true`, `use_chunk=false`, `device=cuda`

- Cold: 32.16s (pip venv) / 115.87s (uv venv)
- Warm: **OOM** on RTX 3070 Laptop (8 GB)

#### CUDA STFT (auto chunk fix, long file)

Input: `0_52_50_1_29_2026_.wav` (~43.11s)  
Mode: `use_torch_stft=true`, `use_chunk="auto"`, `chunk_seconds=30`, `device=cuda`

| Variant | Cold (s) | Warm (s) | Notes |
|---------|----------|----------|-------|
| Auto chunk | 3.52 | 1.15 | Avoided OOM on RTX 3070 Laptop |

## Optimization Tests (Runtime Env, pre-freeze)

Environment: `test_audiosep/runtime` (PyTorch 2.10, MPS)  
Inputs: `harmonica_audiosep.wav` (harmonica) + `1_28_46_1_31_2026_.wav` (vocals)  
Mode: `use_chunk=false`

| Variant | Model Load (s) | Harmonica (s) | Long (s) | Notes |
|---------|----------------|---------------|----------|-------|
| Baseline (torchlibrosa, no mmap) | 2.79 | 0.46 | 0.35 | |
| mmap only | 2.40 | 0.44 | 0.30 | Faster load + slight inference win |
| MPS STFT only | 2.39 | 0.45 | 0.30 | Warns about `torch.istft` resize |
| mmap + MPS STFT | 2.26 | 0.57 | 0.34 | Load fastest, harmonica slower |

Cache test (`--max-cached-models 2`):
- First `load_model`: 2.82s (cached=false)
- Second `load_model`: 0.00s (cached=true)

### Long file (1–2 min) MPS STFT check

Input: `1_28_46_1_31_2026_.wav` (vocals), `use_chunk=false`

| Variant | Model Load (s) | Long (s) | Notes |
|---------|----------------|----------|-------|
| Baseline (torchlibrosa) | 2.71 | 0.55 | |
| MPS STFT | 2.32 | 0.53 | `torch.istft` resize warning |

### Long files (≈1–3 min) MPS STFT check

Inputs:  
- `18_0_43_1_30_2026_.wav` (~78s)  
- `17_55_20_1_30_2026_.wav` (~160s)  
Query: `vocals`, `use_chunk=false`

| Variant | Model Load (s) | 78s (s) | 160s (s) | Notes |
|---------|----------------|---------|----------|-------|
| Baseline (torchlibrosa) | 2.83 | 2.16 | 11.24 | |
| MPS STFT | 3.08 | 1.54 | 3.52 | `torch.istft` resize warning |

### Persistence benchmark (spawn vs. persistent worker)

Worker: `audiosep_worker.py` (runtime env)  
Inputs: `harmonica_audiosep.wav` + `18_0_43_1_30_2026_.wav`  
Mode: `use_torch_stft=auto`, `auto_stft_seconds=60`, `use_chunk=false`

| Variant | Total (s) | Per-job (s) | Notes |
|---------|-----------|-------------|-------|
| Spawn-per-job | 12.04 | 0.46, 3.33 | Load model each run |
| Persistent worker | 7.47 | 0.44, 3.11 | Single long-lived process |

### Cold vs warm and sequence

Worker: `audiosep_worker.py` (runtime env)  
Mode: `use_torch_stft=auto`, `auto_stft_seconds=60`, `use_chunk=false`

Cold starts (spawn → load → separate → exit):
- Short (`harmonica_audiosep.wav`): startup 1.42s, load 2.73s, separate 0.44s, total 4.58s
- Long (`18_0_43_1_30_2026_.wav`): startup 1.25s, load 2.62s, separate 3.30s, total 7.17s

Persistent sequence (short → long → short → long):
- Startup 1.24s, load 2.65s
- Per-job: 0.44s, 3.07s, 2.24s, 3.05s
- Total: 12.69s
|      | cpu    | worker/frozen | audiosep_base | test_4s.wav | 4s |  |  |  |
|      | cuda   | worker/frozen | audiosep_base | test_4s.wav | 4s |  |  |  |

## MPS Optimizations (Ideas to Evaluate)

We are on **PyTorch 2.10**. After baseline benchmarks, evaluate:

- Native MPS STFT/ISTFT usage (reduce CPU fallbacks).
- Autocast behavior on MPS (verify no unintended slow path).
- Chunk sizes that balance memory vs throughput.

Document any change and re-run the table above.
