# Submission Ready Notes

## Current Status

This repository is structured for Q1–Q4 and the dashboard/API are aligned to the same reported metrics.

## Final Files Expected Before Upload

- `error_samples.csv` (from Q1 run)
- `raw_asr_pairs.json` (from Q2 raw ASR generation)
- `spell_check_results.csv` (from Q3 classification output)
- Google Sheet exported from `spell_check_results.csv` with:
  - `word`
  - `verdict`

## Run Order

1. `python app.py` and verify dashboard at `/` and `/dashboard`.
2. Run Q1 with dataset manifest:
   - `MANIFEST_PATH=./manifest.json python q1_whisper_finetune.py`
3. Generate Q2 ASR pairs:
   - `python -c "from q2_cleanup_pipeline import generate_asr_transcripts; import json; rec=json.load(open('manifest.json','r',encoding='utf-8')); generate_asr_transcripts(rec, 'raw_asr_pairs.json')"`
4. Export Q3 labels:
   - `python -c "from q3_spell_checker import run_spell_check_pipeline; import json; words=json.load(open('unique_words.json','r',encoding='utf-8')); run_spell_check_pipeline(words, output_csv='spell_check_results.csv')"`

## Notes

- If dataset URLs changed, update manifest URLs to the working `https://storage.googleapis.com/upload_goai/...` pattern before running.
- Keep metrics consistent across `README.md`, `dashboard.html`, and API outputs.
