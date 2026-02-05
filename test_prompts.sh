#!/bin/bash
# Test multiple AudioSep prompts - compare outputs to debug separation
# Run from test_audiosep with: bash test_prompts.sh

MODELS="/Users/ostino/Documents/Max 9/DSU_VSTOPIA/ThirdPartyApps/Models/audiosep"
INPUT="/Users/ostino/Documents/19_26_2_2_4_2026_.wav"
OUT_DIR="/tmp/audiosep_prompts"
WORKER="python audiosep_worker.py --worker"

mkdir -p "$OUT_DIR"

PROMPTS=(
  "vocals"
  "singing"
  "speech"
  "human voice"
  "bass"
  "drums"
  "music"
  "isolate vocals"
  "remove background"
  "voice"
  "vocal"
)

echo "=== AudioSep prompt comparison ==="
echo "Input: $INPUT"
echo "Output dir: $OUT_DIR"
echo ""

# Load model once, then separate with each prompt
{
  echo "{\"cmd\":\"load_model\",\"config_path\":\"$MODELS/config/audiosep_base.yaml\",\"checkpoint_path\":\"$MODELS/checkpoint/audiosep_base_4M_steps.ckpt\",\"clap_checkpoint_path\":\"$MODELS/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt\",\"roberta_dir\":\"$MODELS/roberta-base\",\"use_torch_stft\":\"auto\",\"auto_stft_seconds\":60,\"mmap\":false}"
  for p in "${PROMPTS[@]}"; do
    safe=$(echo "$p" | tr ' ' '_')
    echo "{\"cmd\":\"separate\",\"input\":\"$INPUT\",\"output\":\"$OUT_DIR/out_${safe}.wav\",\"text\":\"$p\",\"use_chunk\":false}"
  done
  echo '{"cmd":"exit"}'
} | $WORKER 2>&1 | grep -E 'status|Separating|Loaded|error'

echo ""
echo "Outputs in $OUT_DIR"
ls -la "$OUT_DIR"
