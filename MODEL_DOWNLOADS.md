## AudioSep Model Downloads

Primary checkpoints used for inference:

1. **AudioSep base separation model**
   - File: `audiosep_base_4M_steps.ckpt`
   - URL: `https://huggingface.co/spaces/badayvedat/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt`

2. **CLAP audio-text encoder**
   - File: `music_speech_audioset_epoch_15_esc_89.98.pt`
   - URL: `https://huggingface.co/spaces/badayvedat/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt`

3. **RoBERTa base (local, no hub needed)**
   - Directory: `roberta-base/`
   - Files:
     - `config.json`
     - `pytorch_model.bin`
     - `vocab.json`
     - `merges.txt`
     - `tokenizer.json`
     - `tokenizer_config.json`
     - `special_tokens_map.json` (optional; 404 for roberta-base)
   - URLs:
     - `https://huggingface.co/roberta-base/resolve/main/config.json`
     - `https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin`
     - `https://huggingface.co/roberta-base/resolve/main/vocab.json`
     - `https://huggingface.co/roberta-base/resolve/main/merges.txt`
     - `https://huggingface.co/roberta-base/resolve/main/tokenizer.json`
     - `https://huggingface.co/roberta-base/resolve/main/tokenizer_config.json`

Hugging Face checkpoint directory (browse for additional or larger variants):

- `https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint`

Notes:
- These files are **not bundled** in the frozen build.
- The Node downloader should save:
  - AudioSep/CLAP checkpoints under `test_audiosep/checkpoint/`
  - RoBERTa files under `test_audiosep/roberta-base/`
- The worker accepts `clap_checkpoint_path`, `checkpoint_path`, and `roberta_dir` in `load_model`.
