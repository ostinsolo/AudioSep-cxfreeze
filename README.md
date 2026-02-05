# AudioSep cx_Freeze

Standalone frozen runtime for AudioSep (language-queried audio separation).  
Repo: `github.com/ostinsolo/AudioSep-cxfreeze`

This repo is **not** the upstream project. It packages AudioSep into a minimal, inference-only worker + shared runtime.

## Documentation

- **[AUDIOSEP_DOCUMENTATION.md](AUDIOSEP_DOCUMENTATION.md)** — Architecture, prompt-to-operation flow, stereo/44.1 kHz improvements
- **[AUDIOSEP_TODO.md](AUDIOSEP_TODO.md)** — Professional verification checklist (evidence-based)

## Improvements over upstream

- **Stereo output:** Per-channel processing (L and R separated independently); mono input duplicated to L=R
- **44.1 kHz output:** Resampled from model rate (32 kHz) for DAW compatibility
- **mmap fix:** Correct `load_from_checkpoint` usage for frozen builds

## Releases

- Tag `v*` or run the **Release AudioSep cx_Freeze** workflow to create a release.
- Targets:
  - **Windows CUDA** (`requirements-inference-windows-cuda.txt`)
  - **Windows CPU** (`requirements-inference-windows-cpu.txt`)
  - **macOS ARM (MPS)** (`requirements-inference-mps.txt`)
  - **macOS Intel** (manual build via `build_runtime_mac_intel.sh`)

## Build (local)

```bash
python build_dsu_audiosep.py
```

Output: `dist/dsu-audiosep/`  
Runtime scripts: `build_runtime_mac_mps.sh`, `build_runtime_mac_intel.sh`

### macOS Intel (manual build, reused across releases)

GitHub Actions macos runners are ARM. The release workflow **copies the Intel build from the previous release** into each new release. To update it:

1. On an Intel Mac (or via SSH): `./build_runtime_mac_intel.sh` then `python build_dsu_audiosep.py`
2. Create `audiosep-mac-intel.tar.gz`: `tar -czf audiosep-mac-intel.tar.gz -C dist dsu-audiosep`
3. Upload to the **current** release on GitHub (replaces the copied one)

Until you do that, each new release includes the Intel build from the prior release.

## Models

Model files are **not** committed. See `MODEL_DOWNLOADS.md` for URLs and download notes.

## Worker Protocol (audiosep)

Worker starts in persistent mode:

```bash
./dist/dsu-audiosep/dsu-audiosep --worker
```

Send one JSON command per line via stdin. Responses are JSON on stdout.

### Commands

**1) load_model**

```json
{
  "cmd": "load_model",
  "config_path": "/abs/path/config/audiosep_base.yaml",
  "checkpoint_path": "/abs/path/checkpoint/audiosep_base_4M_steps.ckpt",
  "clap_checkpoint_path": "/abs/path/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt",
  "roberta_dir": "/abs/path/roberta-base",
  "mmap": false,
  "use_torch_stft": "auto",
  "auto_stft_seconds": 60
}
```

**2) separate**

```json
{
  "cmd": "separate",
  "input": "/abs/path/input.wav",
  "output": "/abs/path/output.wav",
  "text": "harmonica",
  "use_chunk": false
}
```

### Parameters

- `config_path` (load_model): AudioSep YAML config.
- `checkpoint_path` (load_model): AudioSep main checkpoint.
- `clap_checkpoint_path` (load_model): CLAP checkpoint.
- `roberta_dir` (load_model): Local RoBERTa directory.
- `mmap` (load_model): `true` enables `torch.load(..., mmap=True)` when supported (faster load on SSD).
- `use_torch_stft` (load_model): `true` uses native `torch.stft/torch.istft` (MPS path in PyTorch 2.10). Use `"auto"` to switch based on input duration.
- `auto_stft_seconds` (load_model): Threshold (seconds) for `"auto"` mode. Default `60`.
- `input` (separate): Input WAV path.
- `output` (separate): Output WAV path.
- `text` (separate): Query text (e.g. `"harmonica"`, `"vocals"`).
- `use_chunk` (separate): `true` uses chunked inference (faster on long audio, can add artifacts on short clips). `false` uses full pass (best quality on short clips). `"auto"` enables duration-based chunking (see `audiosep.chunk_seconds`).

### Worker flags

- `--max-cached-models N`: keep up to `N` models in memory (LRU). Default `1`.

### Runtime policy (optional)

Workers can read a shared policy file to keep cache and MPS/STFT settings
consistent across runtimes without coupling processes.

Default path: `~/.dsu/runtime_policy.json`  
Override: `DSU_RUNTIME_POLICY_PATH=/abs/path/runtime_policy.json`

AudioSep keys (optional):
- `audiosep.max_cached_models`
- `audiosep.use_torch_stft` (`true`, `false`, or `"auto"`)
- `audiosep.auto_stft_seconds`
- `audiosep.mmap`
- `audiosep.chunk_seconds` (default 30)

### Example

```bash
printf '%s\n' \
  '{"cmd":"load_model","config_path":"/abs/path/config/audiosep_base.yaml","checkpoint_path":"/abs/path/checkpoint/audiosep_base_4M_steps.ckpt","clap_checkpoint_path":"/abs/path/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt","roberta_dir":"/abs/path/roberta-base"}' \
  '{"cmd":"separate","input":"/abs/path/harmonica_audiosep.wav","output":"/abs/path/out_nochunk.wav","text":"harmonica","use_chunk":false}' \
  '{"cmd":"separate","input":"/abs/path/harmonica_audiosep.wav","output":"/abs/path/out_chunk.wav","text":"harmonica","use_chunk":true}' \
  '{"cmd":"exit"}' \
  | ./dist/dsu-audiosep/dsu-audiosep --worker
```

### Benchmark persistence (spawn vs. persistent)

This compares spawning a worker per job vs a single long-lived worker.

Latest Windows CUDA results are recorded in `BENCHMARKS.md`.

```bash
python benchmark_worker_persistence.py \
  --config /abs/path/config/audiosep_base.yaml \
  --checkpoint /abs/path/checkpoint/audiosep_base_4M_steps.ckpt \
  --clap /abs/path/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt \
  --roberta /abs/path/roberta-base \
  --inputs /abs/path/a.wav /abs/path/b.wav \
  --out-dir /tmp/audiosep_bench \
  --text vocals \
  --use-torch-stft auto \
  --auto-stft-seconds 60
```

## Required repo layout (for GitHub Actions)

These paths must exist in the repo root:

- `audiosep_worker.py`
- `build_dsu_audiosep.py`
- `pipeline.py`
- `utils.py`
- `models/` (includes `models/CLAP/open_clip/` and `model_configs/`)
- `config/`
- `requirements-inference-*.txt`
- `.github/workflows/release.yml`

## Regression test

```bash
python verify_output_format.py --input /path/to/input.wav --models /path/to/models
```

Checks output is stereo, 44.1 kHz, and duration matches input.

## TODO

- [ ] Implement seeds in the AudioSep source code (for reproducible separation).

## Attribution

AudioSep is from the original authors and repository. This repo packages it into a frozen runtime.

Original project: https://github.com/Audio-AGI/AudioSep  
Paper: https://arxiv.org/abs/2308.05037

## Cite this work

If you use AudioSep, please cite the original authors:
```bibtex
@article{liu2023separate,
  title={Separate Anything You Describe},
  author={Liu, Xubo and Kong, Qiuqiang and Zhao, Yan and Liu, Haohe and Yuan, Yi, and Liu, Yuzhuo, and Xia, Rui and Wang, Yuxuan, and Plumbley, Mark D and Wang, Wenwu},
  journal={arXiv preprint arXiv:2308.05037},
  year={2023}
}
```
```bibtex
@inproceedings{liu22w_interspeech,
  title={Separate What You Describe: Language-Queried Audio Source Separation},
  author={Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Zhao, Jinzheng and Huang, Qiushi, and Plumbley, Mark D and Wang, Wenwu},
  year=2022,
  booktitle={Proc. Interspeech},
  pages={1801--1805},
}
```
