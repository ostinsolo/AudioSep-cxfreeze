# AudioSep cx_Freeze

Standalone frozen runtime for AudioSep (language-queried audio separation).  
Repo: `github.com/ostinsolo/AudioSep-cxfreeze`

This repo is **not** the upstream project. It packages AudioSep into a minimal, inference-only worker + shared runtime.

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

## Models

Model files are **not** committed. See `MODEL_DOWNLOADS.md` for URLs and download notes.

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
