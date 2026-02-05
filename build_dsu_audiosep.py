#!/usr/bin/env python3
import os
import sys
import shutil

from cx_Freeze import Executable, setup

sys.setrecursionlimit(5000)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

APP_NAME = "DSU-Audiosep"
APP_VERSION = "1.0.3"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dist", "dsu-audiosep")
if os.path.exists(OUTPUT_DIR):
    print(f"Removing previous build: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

include_files = []

config_dir = os.path.join(PROJECT_ROOT, "config")
if os.path.exists(config_dir):
    include_files.append((config_dir, "config"))

checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoint")
if os.path.exists(checkpoint_dir):
    pass

clap_configs = os.path.join(PROJECT_ROOT, "models", "CLAP", "open_clip", "model_configs")
if os.path.exists(clap_configs):
    include_files.append((clap_configs, os.path.join("models", "CLAP", "open_clip", "model_configs")))

clap_vocab = os.path.join(PROJECT_ROOT, "models", "CLAP", "open_clip", "bpe_simple_vocab_16e6.txt.gz")
if os.path.exists(clap_vocab):
    include_files.append((clap_vocab, os.path.join("models", "CLAP", "open_clip", "bpe_simple_vocab_16e6.txt.gz")))

packages = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch._C",
    "torch.cuda",
    "torch.amp",
    "torch.backends",
    "torch.fft",
    "torch.linalg",
    "torch.distributions",
    "torchaudio",
    "torchvision",
    "torchlibrosa",
    "numpy",
    "scipy",
    "soundfile",
    "librosa",
    "yaml",
    "transformers",
    "huggingface_hub",
    "ftfy",
    "regex",
    "tqdm",
    "h5py",
    "PIL",
    "models",
    "models.CLAP",
    "models.CLAP.open_clip",
]

excludes = [
    # AudioSep project-specific (training, colab, etc.)
    "models.CLAP.training",
    "data",
    "evaluation",
    "callbacks",
    "optimizers",
    "losses",
    "train",
    "benchmark",
    "predict",
    "run_test",
    "AudioSep_Colab",
    "lightning",
    "lightning.pytorch",
    "pytorch_lightning",
    "torchmetrics",
    "webdataset",
    "pandas",
    # NOTE: sklearn is NOT excluded - required by librosa and CLAP loss.py
    # Standard exclusions (reduce size, not needed at runtime)
    "tkinter",
    "test",
    "tests",
    "distutils",
    "setuptools",
    "pip",
    "wheel",
    "pydoc_data",
    "curses",
    "IPython",
    "jupyter",
    "notebook",
    "matplotlib.backends.backend_qt5agg",
    "PyQt5",
    "PySide2",
    "cx_Freeze",
]

build_exe_options = {
    "packages": packages,
    "excludes": excludes,
    "include_files": include_files,
    "zip_include_packages": [],
    "zip_exclude_packages": "*",
    "build_exe": OUTPUT_DIR,
}

exe_suffix = ".exe" if sys.platform == "win32" else ""
base = "Console" if sys.platform == "win32" else None

executables = [
    Executable(
        script=os.path.join(PROJECT_ROOT, "audiosep_worker.py"),
        target_name=f"dsu-audiosep{exe_suffix}",
        base=base,
    ),
]

if len(sys.argv) == 1:
    sys.argv.append("build_exe")


def _post_build_cleanup(output_dir):
    """Remove unnecessary files (tests, Cython sources, docs) to reduce size."""
    lib_dir = os.path.join(output_dir, "lib")
    total_removed = 0
    if not os.path.isdir(lib_dir):
        return
    # 1. Remove test directories
    for root, dirs, _ in os.walk(lib_dir, topdown=False):
        for d in dirs:
            if d in ("tests", "test"):
                path = os.path.join(root, d)
                if os.path.isdir(path):
                    try:
                        size = sum(
                            os.path.getsize(os.path.join(r, f))
                            for r, _, files in os.walk(path) for f in files
                        )
                        shutil.rmtree(path)
                        total_removed += size
                    except OSError:
                        pass
    # 2. Remove Cython/C source (.pyx, .pxd, .c, .h). Keep .pyi - librosa needs them
    for root, dirs, files in os.walk(lib_dir, topdown=False):
        for f in files:
            if f.endswith((".pyx", ".pxd", ".c", ".h")):
                path = os.path.join(root, f)
                try:
                    total_removed += os.path.getsize(path)
                    os.remove(path)
                except OSError:
                    pass
    # 3. Remove examples, doc, docs, include
    for root, dirs, _ in os.walk(lib_dir, topdown=False):
        for d in dirs:
            if d in ("examples", "example", "doc", "docs", "include"):
                path = os.path.join(root, d)
                if os.path.isdir(path):
                    try:
                        size = sum(
                            os.path.getsize(os.path.join(r, f))
                            for r, _, files in os.walk(path) for f in files
                        )
                        shutil.rmtree(path)
                        total_removed += size
                    except OSError:
                        pass
    if total_removed > 0:
        print(f"\nPost-build: Removed ~{total_removed / (1024*1024):.1f} MB (tests, .pyx/.c/.h, examples, doc, include)")


setup(
    name=APP_NAME,
    version=APP_VERSION,
    description="DSU Audiosep (inference-only)",
    options={"build_exe": build_exe_options},
    executables=executables,
)

# Post-build cleanup (runs after cx_Freeze completes)
_post_build_cleanup(OUTPUT_DIR)
