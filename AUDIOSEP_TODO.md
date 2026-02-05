# AudioSep Professional TODO

Evidence-based checklist. Mark items with verification proof (command output, file checks).

## Verification Checklist

### Verify stereo

- [x] Compare MD5 of L vs R for stereo input; confirm different
- **Evidence (2025-02-04):** `L == R (identical): False`, `Stereo verified: True (L and R differ)`. Shape (152577, 2), SR 44100.

### Verify 44.1 kHz

- [x] Check output with `soundfile`/`ffprobe`; duration matches input
- **Evidence (2025-02-04):** Input 3.46s @ 44100 Hz, Output 3.46s @ 44100 Hz. Duration match: True.

### Verify prompt sensitivity

- [x] Run `test_prompts.sh`; confirm outputs differ by prompt
- **Evidence (2025-02-04):** `md5 /tmp/audiosep_prompts/out_*.wav` — all hashes differ (e.g. bass=1c2dcdef..., drums=eca26a5d..., vocals=4911c79e...).

### Document original upstream

- [ ] Diff our pipeline vs Audio-AGI/AudioSep inference (when accessible)
- **Evidence:** Compare `pipeline.py` with upstream `separate.py` or equivalent

### Regression tests

- [x] Add minimal pytest/script for load → separate → check output format
- **Evidence:** `verify_output_format.py` added; checks stereo, 44.1 kHz, valid WAV

## Optional

### "remove X" operation

- [ ] Design and implement `mixture - extract(X)` if needed
- **Evidence:** New worker command or pipeline option

### Configurable OUT_SR

- [ ] Support 48 kHz or match input sample rate
- **Evidence:** `OUT_SR` from config or job parameter
