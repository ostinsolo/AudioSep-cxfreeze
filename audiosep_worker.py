#!/usr/bin/env python3
import json
import os
import sys
import time
import traceback

# Fix for PyTorch inspect issues in frozen builds
if getattr(sys, "frozen", False):
    import inspect
    _original_getsourcelines = inspect.getsourcelines
    _original_getsource = inspect.getsource
    _original_findsource = inspect.findsource

    def _safe_getsourcelines(obj):
        try:
            return _original_getsourcelines(obj)
        except OSError:
            return ([""], 0)

    def _safe_getsource(obj):
        try:
            return _original_getsource(obj)
        except OSError:
            return ""

    def _safe_findsource(obj):
        try:
            return _original_findsource(obj)
        except OSError:
            return ([""], 0)

    inspect.getsourcelines = _safe_getsourcelines
    inspect.getsource = _safe_getsource
    inspect.findsource = _safe_findsource

# Dummy modules to avoid heavy training deps at runtime
import torch
import numpy as np
import torchaudio
import torchvision
import types

def _install_dummy_lightning():
    lightning_mod = types.ModuleType("lightning")
    pytorch_mod = types.ModuleType("lightning.pytorch")

    class _LightningModule(torch.nn.Module):
        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, strict=False, map_location=None, **kwargs):
            mmap_flag = bool(kwargs.pop("mmap", False))
            instance = cls(**kwargs)
            if mmap_flag:
                try:
                    state = torch.load(checkpoint_path, map_location=map_location, mmap=True, weights_only=False)
                except TypeError:
                    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            else:
                state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            state_dict = state.get("state_dict", state)
            instance.load_state_dict(state_dict, strict=strict)
            return instance

    pytorch_mod.LightningModule = _LightningModule
    lightning_mod.pytorch = pytorch_mod
    sys.modules["lightning"] = lightning_mod
    sys.modules["lightning.pytorch"] = pytorch_mod


def _install_dummy_clap_data():
    data_mod = types.ModuleType("models.CLAP.training.data")

    def get_mel(audio_data, audio_cfg):
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg["sample_rate"],
            n_fft=audio_cfg["window_size"],
            win_length=audio_cfg["window_size"],
            hop_length=audio_cfg["hop_size"],
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=64,
            f_min=audio_cfg["fmin"],
            f_max=audio_cfg["fmax"],
        ).to(audio_data.device)
        mel = mel(audio_data)
        mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
        return mel.T

    def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg):
        with torch.no_grad():
            if len(audio_data) > max_len:
                if data_truncating == "rand_trunc":
                    longer = torch.tensor([True])
                elif data_truncating == "fusion":
                    mel = get_mel(audio_data, audio_cfg)
                    chunk_frames = max_len // audio_cfg["hop_size"] + 1
                    total_frames = mel.shape[0]
                    if chunk_frames == total_frames:
                        mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                        sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([False])
                    else:
                        ranges = np.array_split(
                            list(range(0, total_frames - chunk_frames + 1)), 3
                        )
                        if len(ranges[1]) == 0:
                            ranges[1] = [0]
                        if len(ranges[2]) == 0:
                            ranges[2] = [0]
                        idx_front = np.random.choice(ranges[0])
                        idx_middle = np.random.choice(ranges[1])
                        idx_back = np.random.choice(ranges[2])
                        mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
                        mel_chunk_middle = mel[idx_middle : idx_middle + chunk_frames, :]
                        mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]
                        mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, 64])(
                            mel[None]
                        )[0]
                        mel_fusion = torch.stack(
                            [mel_chunk_front, mel_chunk_middle, mel_chunk_back, mel_shrink],
                            dim=0,
                        )
                        sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([True])
                else:
                    raise NotImplementedError(
                        f"data_truncating {data_truncating} not implemented"
                    )
                overflow = len(audio_data) - max_len
                idx = np.random.randint(0, overflow + 1)
                audio_data = audio_data[idx : idx + max_len]
            else:
                if len(audio_data) < max_len:
                    if data_filling == "repeatpad":
                        n_repeat = int(max_len / len(audio_data))
                        audio_data = audio_data.repeat(n_repeat)
                        audio_data = torch.nn.functional.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "pad":
                        audio_data = torch.nn.functional.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "repeat":
                        n_repeat = int(max_len / len(audio_data))
                        audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                    else:
                        raise NotImplementedError(
                            f"data_filling {data_filling} not implemented"
                        )
                longer = torch.tensor([False])

            sample["audio"] = audio_data
            sample["longer"] = longer

        return sample

    data_mod.get_audio_features = get_audio_features
    sys.modules["models.CLAP.training.data"] = data_mod


def _install_dummy_clap_loss():
    loss_mod = types.ModuleType("models.CLAP.open_clip.loss")

    class _LossPlaceholder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("CLAP loss is not available in inference build")

    def _unavailable(*args, **kwargs):
        raise RuntimeError("CLAP loss is not available in inference build")

    loss_mod.ClipLoss = _LossPlaceholder
    loss_mod.LPLoss = _LossPlaceholder
    loss_mod.LPMetrics = _LossPlaceholder
    loss_mod.gather_features = _unavailable
    loss_mod.lp_gather_features = _unavailable
    sys.modules["models.CLAP.open_clip.loss"] = loss_mod


_install_dummy_lightning()
_install_dummy_clap_data()
_install_dummy_clap_loss()



def send_json(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def worker_mode():
    def _parse_max_cached_models(argv):
        if "--max-cached-models" in argv:
            idx = argv.index("--max-cached-models")
            if idx + 1 < len(argv):
                try:
                    return max(0, int(argv[idx + 1]))
                except ValueError:
                    pass
        env_val = os.environ.get("AUDIOSEP_MAX_CACHED_MODELS")
        if env_val:
            try:
                return max(0, int(env_val))
            except ValueError:
                pass
        return 1

    model = None
    model_cache = {}
    cache_order = []
    max_cached_models = _parse_max_cached_models(sys.argv)
    device = None
    config_path = None
    checkpoint_path = None
    base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
    os.environ["AUDIOSEP_BASE_DIR"] = base_dir
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    from pipeline import build_audiosep, separate_audio

    send_json({"status": "loading", "message": "Initializing DSU-Audiosep worker..."})

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    send_json({"status": "ready", "device": str(device)})

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            job = json.loads(line)
        except Exception:
            send_json({"status": "error", "message": "Invalid JSON"})
            continue

        cmd = job.get("cmd")
        if cmd == "exit":
            send_json({"status": "exiting"})
            break

        if cmd == "get_status":
            send_json({
                "status": "status",
                "ready": True,
                "device": str(device),
                "model_loaded": model is not None,
                "checkpoint_path": checkpoint_path,
                "config_path": config_path,
            })
            continue

        if cmd == "load_model":
            config_path = job.get("config_path", "config/audiosep_base.yaml")
            checkpoint_path = job.get("checkpoint_path", "checkpoint/audiosep_base_4M_steps.ckpt")
            clap_checkpoint_path = job.get("clap_checkpoint_path")
            roberta_dir = job.get("roberta_dir")
            device_str = job.get("device")
            use_torch_stft = bool(job.get("use_torch_stft", False))
            mmap_load = bool(job.get("mmap", False))
            if device_str:
                device = torch.device(device_str)

            if not os.path.isabs(config_path):
                config_path = os.path.join(base_dir, config_path)
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(base_dir, checkpoint_path)
            if clap_checkpoint_path and not os.path.isabs(clap_checkpoint_path):
                clap_checkpoint_path = os.path.join(base_dir, clap_checkpoint_path)
            if not clap_checkpoint_path:
                default_clap = os.path.join(base_dir, "checkpoint", "music_speech_audioset_epoch_15_esc_89.98.pt")
                clap_checkpoint_path = default_clap
            if roberta_dir and not os.path.isabs(roberta_dir):
                roberta_dir = os.path.join(base_dir, roberta_dir)
            if not roberta_dir:
                roberta_dir = os.path.join(base_dir, "roberta-base")

            os.environ["AUDIOSEP_BASE_DIR"] = base_dir
            if clap_checkpoint_path:
                os.environ["AUDIOSEP_CLAP_CKPT"] = clap_checkpoint_path
            if roberta_dir:
                os.environ["AUDIOSEP_ROBERTA_DIR"] = roberta_dir
            os.environ["AUDIOSEP_USE_TORCH_STFT"] = "1" if use_torch_stft else "0"
            os.environ["AUDIOSEP_MMAP_LOAD"] = "1" if mmap_load else "0"

            if not os.path.exists(config_path):
                send_json({"status": "error", "message": f"Config not found: {config_path}"})
                continue
            if not os.path.exists(checkpoint_path):
                send_json({"status": "error", "message": f"Checkpoint not found: {checkpoint_path}"})
                continue
            if clap_checkpoint_path and not os.path.exists(clap_checkpoint_path):
                send_json({"status": "error", "message": f"CLAP checkpoint not found: {clap_checkpoint_path}"})
                continue
            if roberta_dir and not os.path.exists(roberta_dir):
                send_json({"status": "error", "message": f"RoBERTa dir not found: {roberta_dir}"})
                continue

            cache_key = (
                os.path.abspath(config_path),
                os.path.abspath(checkpoint_path),
                os.path.abspath(clap_checkpoint_path) if clap_checkpoint_path else "",
                os.path.abspath(roberta_dir) if roberta_dir else "",
                str(device),
                bool(use_torch_stft),
                bool(mmap_load),
            )

            if cache_key in model_cache:
                model = model_cache[cache_key]
                if cache_key in cache_order:
                    cache_order.remove(cache_key)
                cache_order.append(cache_key)
                send_json({
                    "status": "model_loaded",
                    "elapsed": 0.0,
                    "checkpoint_path": checkpoint_path,
                    "clap_checkpoint_path": clap_checkpoint_path,
                    "roberta_dir": roberta_dir,
                    "config_path": config_path,
                    "device": str(device),
                    "cached": True,
                    "cache_size": len(model_cache),
                    "max_cached_models": max_cached_models,
                })
                continue

            try:
                t0 = time.time()
                model = build_audiosep(
                    config_yaml=config_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    mmap=mmap_load,
                )
                elapsed = round(time.time() - t0, 2)
                if max_cached_models > 0:
                    model_cache[cache_key] = model
                    cache_order.append(cache_key)
                    while len(cache_order) > max_cached_models:
                        evict_key = cache_order.pop(0)
                        model_cache.pop(evict_key, None)
                send_json({
                    "status": "model_loaded",
                    "elapsed": elapsed,
                    "checkpoint_path": checkpoint_path,
                    "clap_checkpoint_path": clap_checkpoint_path,
                    "roberta_dir": roberta_dir,
                    "config_path": config_path,
                    "device": str(device),
                    "cached": False,
                    "cache_size": len(model_cache),
                    "max_cached_models": max_cached_models,
                })
            except Exception as e:
                send_json({
                    "status": "error",
                    "message": f"Failed to load model: {str(e)}",
                    "traceback": traceback.format_exc(),
                })
            continue

        if cmd == "separate":
            if model is None:
                send_json({"status": "error", "message": "Model not loaded. Use load_model first."})
                continue

            input_path = job.get("input")
            output_path = job.get("output")
            text = job.get("text", "speech")
            use_chunk = bool(job.get("use_chunk", False))

            if not input_path or not os.path.exists(input_path):
                send_json({"status": "error", "message": f"Input not found: {input_path}"})
                continue
            if not output_path:
                send_json({"status": "error", "message": "Output path not specified"})
                continue

            try:
                t0 = time.time()
                separate_audio(
                    model=model,
                    audio_file=input_path,
                    text=text,
                    output_file=output_path,
                    device=str(device),
                    use_chunk=use_chunk,
                )
                elapsed = round(time.time() - t0, 2)
                send_json({
                    "status": "done",
                    "elapsed": elapsed,
                    "files": [output_path],
                })
            except Exception as e:
                send_json({
                    "status": "error",
                    "message": f"Separation failed: {str(e)}",
                    "traceback": traceback.format_exc(),
                })
            continue

        send_json({"status": "error", "message": f"Unknown command: {cmd}"})


def main():
    if "--worker" in sys.argv:
        worker_mode()
    else:
        print("Usage: audiosep_worker.py --worker", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
