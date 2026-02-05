import yaml
from typing import List
import torch
import numpy as np
import librosa
from scipy.io.wavfile import write
from utils import ignore_warnings, parse_yaml, load_ss_model
from models.clap_encoder import CLAP_Encoder

def build_audiosep(config_yaml, checkpoint_path, device, mmap=False):
    ignore_warnings()
    configs = parse_yaml(config_yaml)
    
    query_encoder = CLAP_Encoder().eval()
    model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder,
        mmap=mmap,
    ).eval().to(device)

    print(f'Loaded AudioSep model from [{checkpoint_path}]')
    return model

SR = 32000  # Model native rate
OUT_SR = 44100  # Output sample rate (match typical DAW/project rate)


def _run_separation(model, mixture_mono, conditions, device, use_chunk):
    """Run separation on a single mono channel. Returns (samples,) float32."""
    input_dict = {
        "mixture": torch.Tensor(mixture_mono)[None, None, :].to(device),
        "condition": conditions,
    }
    if use_chunk:
        window = (1.0 + 3.0 + 1.0) * SR
        if input_dict["mixture"].shape[2] <= window * 2:
            sep = model.ss_model(input_dict)["waveform"]
            sep = sep.squeeze(0).squeeze(0).data.cpu().numpy()
        else:
            sep = model.ss_model.chunk_inference(input_dict)
            sep = np.squeeze(sep)
    else:
        sep = model.ss_model(input_dict)["waveform"]
        sep = sep.squeeze(0).squeeze(0).data.cpu().numpy()
    return sep


def separate_audio(model, audio_file, text, output_file, device='cuda', use_chunk=False):
    print(f'Separating audio from [{audio_file}] with textual query: [{text}]')
    # Load preserving channels: stereo (2, N) or mono (N,)
    mixture, fs = librosa.load(audio_file, sr=SR, mono=False)
    if mixture.ndim == 1:
        mixture = mixture[np.newaxis, :]  # (1, N)

    with torch.no_grad():
        text_list = [text]
        conditions = model.query_encoder.get_query_embed(
            modality='text',
            text=text_list,
            device=device
        )

        # Process each channel separately for real stereo (L and R get independent separation)
        sep_channels = []
        for ch in range(mixture.shape[0]):
            sep_ch = _run_separation(model, mixture[ch], conditions, device, use_chunk)
            sep_channels.append(sep_ch)

        # Stack to stereo (N, 2) or mono input -> duplicate to (N, 2)
        sep_stereo = np.stack(sep_channels, axis=1)
        if sep_stereo.shape[1] == 1:
            sep_stereo = np.repeat(sep_stereo, 2, axis=1)

        # Resample from model rate (32 kHz) to output rate (44.1 kHz)
        if OUT_SR != SR:
            sep_stereo = librosa.resample(
                sep_stereo.T, orig_sr=SR, target_sr=OUT_SR, res_type="polyphase"
            ).T

        out_int16 = np.round(sep_stereo * 32767).astype(np.int16)
        write(output_file, OUT_SR, out_int16)
        print(f'Separated audio written to [{output_file}]')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_audiosep(
        config_yaml='config/audiosep_base.yaml', 
        checkpoint_path='checkpoint/step=3920000.ckpt', 
        device=device)

    audio_file = '/mnt/bn/data-xubo/project/AudioShop/YT_audios/Y3VHpLxtd498.wav'
    text = 'pigeons are cooing in the background'
    output_file = 'separated_audio.wav'
    
    separate_audio(model, audio_file, text, output_file, device)
