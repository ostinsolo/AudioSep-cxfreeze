#!/bin/bash
# Test AudioSep executable directly - extract vocals from 19_26_2_2_4_2026_.wav
# Same paths as Max/DSU setup

AUDIOSEP_EXE="/Users/ostino/Documents/Max 9/DSU_VSTOPIA/ThirdPartyApps/audiosep/dsu-audiosep/dsu-audiosep"
MODELS="/Users/ostino/Documents/Max 9/DSU_VSTOPIA/ThirdPartyApps/Models/audiosep"
INPUT="/Users/ostino/Documents/19_26_2_2_4_2026_.wav"
OUTPUT="/Users/ostino/Documents/Max 9/DSU_VSTOPIA/Output_DSU/19_26_2_2_4_2026__audiosep_CLI_test.wav"

if [ ! -f "$AUDIOSEP_EXE" ]; then
  echo "Executable not found: $AUDIOSEP_EXE"
  exit 1
fi
if [ ! -f "$INPUT" ]; then
  echo "Input not found: $INPUT"
  exit 1
fi

echo "=== AudioSep CLI test: vocal extraction ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Text:   vocals"
echo ""

# Worker protocol: JSON lines over stdin
# Use "vocals" (plural) - matches benchmark/README; try "singing" or "speech" if needed
printf '%s\n' \
  "{\"cmd\":\"load_model\",\"config_path\":\"$MODELS/config/audiosep_base.yaml\",\"checkpoint_path\":\"$MODELS/checkpoint/audiosep_base_4M_steps.ckpt\",\"clap_checkpoint_path\":\"$MODELS/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt\",\"roberta_dir\":\"$MODELS/roberta-base\",\"use_torch_stft\":\"auto\",\"auto_stft_seconds\":60,\"mmap\":false}" \
  "{\"cmd\":\"separate\",\"input\":\"$INPUT\",\"output\":\"$OUTPUT\",\"text\":\"vocals\",\"use_chunk\":false}" \
  '{"cmd":"exit"}' \
  | "$AUDIOSEP_EXE" --worker 2>&1

echo ""
echo "Done. Output: $OUTPUT"
