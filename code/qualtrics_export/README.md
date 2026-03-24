# Qualtrics Focus Export

Structure:

- `input/`: put the JSON array files you want to convert here
- `build_survey_subset_from_clips.py`: builds a JSON subset from clip filenames like `f1_item0.wav`
- `qualtrics_focus_export.py`: converts JSON items into Qualtrics Advanced TXT
- `qualtrics_inference_export.py`: converts JSON items into the inference-survey Qualtrics Advanced TXT
- `output/`: generated `.txt` files for Qualtrics import

Example:

```bash
python3 code/qualtrics_export/qualtrics_focus_export.py \
  code/qualtrics_export/input/my_focus_items.json \
  code/qualtrics_export/output/my_focus_survey.txt
```

This writes:

- `my_focus_survey.txt`: Qualtrics Advanced TXT import file
- `my_focus_survey.audio_map.csv`: question-to-audio mapping for manual upload and attachment

Audio mapping behavior:

- By default, clips are looked up in `data/clips/set1/`.
- If the input file is `f1.json`, the script maps question 1 to `f1_item0.wav`, question 2 to `f1_item1.wav`, and so on.
- If you build a custom subset from multiple source JSON files, add `source_json`, `source_item_index`, or `audio_file` fields to each item so the audio map stays correct.

Recommended workflow for the focus survey from `data/clips/set1/`:

```bash
python3 code/qualtrics_export/build_survey_subset_from_clips.py \
  data/clips/set1 \
  code/qualtrics_export/input/set1_focus_items.json

python3 code/qualtrics_export/qualtrics_focus_export.py \
  code/qualtrics_export/input/set1_focus_items.json \
  code/qualtrics_export/output/set1_focus_survey.txt
```

That gives you:

- `set1_focus_survey.txt` for the Qualtrics question import
- `set1_focus_survey.audio_map.csv` so you can manually attach the right uploaded audio to each question

Recommended workflow for the inference survey from `data/clips/set2/`:

```bash
python3 code/qualtrics_export/build_survey_subset_from_clips.py \
  data/clips/set2 \
  code/qualtrics_export/input/set2_inference_items.json

python3 code/qualtrics_export/qualtrics_inference_export.py \
  code/qualtrics_export/input/set2_inference_items.json \
  code/qualtrics_export/output/set2_inference_survey.txt
```

That gives you:

- `set2_inference_survey.txt` for the Qualtrics question import
- `set2_inference_survey.audio_map.csv` so you can manually attach the right uploaded audio to each question
