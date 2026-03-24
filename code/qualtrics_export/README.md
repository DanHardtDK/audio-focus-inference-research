# Qualtrics Survey Export Workflow

This folder contains the scripts for building Qualtrics import files from the experiment JSON items and the selected audio clips.

## Folder layout

- `build_survey_subset_from_clips.py`
  Builds a JSON subset from clip filenames like `f1_item0.wav`.
- `focus_text_utils.py`
  Normalizes sentence text so the exported survey does not reveal focus through all-caps words.
- `qualtrics_export_common.py`
  Shared helpers for loading JSON and creating audio mapping CSV files.
- `qualtrics_focus_export.py`
  Creates the Qualtrics Advanced TXT import for the focus survey.
- `qualtrics_inference_export.py`
  Creates the Qualtrics Advanced TXT import for the inference survey.
- `input/`
  Place generated or hand-edited survey JSON files here.
- `output/`
  Generated Qualtrics TXT files and audio mapping CSV files.

## Source data

- Source JSON items live in `data/input/f*.json`
- Focus-survey audio clips are selected from `data/clips/set1/`
- Inference-survey audio clips are selected from `data/clips/set2/`

Clip names determine which source item is selected:

- `f1_item0.wav` means item `0` from `data/input/f1.json`
- `f8_item5.wav` means item `5` from `data/input/f8.json`

## What the subset builder adds

When you run `build_survey_subset_from_clips.py`, it copies the matching source JSON items into one smaller JSON array and adds:

- `source_json`
- `source_item_index`
- `audio_file`
- `S1_normalized`
- `S2_normalized`

The normalized fields are used for survey display so all-caps emphasis cues are removed from the visible text.

## Focus survey workflow

Use this for the audio-perception survey based on `data/clips/set1/`.

### Step 1: build the focus subset JSON

```bash
python3 code/qualtrics_export/build_survey_subset_from_clips.py \
  data/clips/set1 \
  code/qualtrics_export/input/set1_focus_items.json
```

This creates:

- `code/qualtrics_export/input/set1_focus_items.json`

### Step 2: create the Qualtrics focus import

```bash
python3 code/qualtrics_export/qualtrics_focus_export.py \
  code/qualtrics_export/input/set1_focus_items.json \
  code/qualtrics_export/output/set1_focus_survey.txt
```

This creates:

- `code/qualtrics_export/output/set1_focus_survey.txt`
- `code/qualtrics_export/output/set1_focus_survey.audio_map.csv`

### Focus survey behavior

- The Qualtrics block is `[[Block:Focus Survey]]`
- The question asks:
  `In Sentence 1, which word was said with stronger emphasis?`
- Only normalized `Sentence 1` is shown in the survey
- `Sentence 2` is not shown
- The answer choices are derived from `Sentence 1`
- The answer choices are normalized, so no all-caps emphasis is visible

Example exported focus question:

```txt
[[Question:MC:SingleAnswer:Vertical]]
In Sentence 1, which word was said with stronger emphasis?<br><br>
Sentence 1: Sam only gave Rob oranges.<br>
[[Choices]]
oranges
Rob
```

## Inference survey workflow

Use this for the reasoning survey based on `data/clips/set2/`.

### Step 1: build the inference subset JSON

```bash
python3 code/qualtrics_export/build_survey_subset_from_clips.py \
  data/clips/set2 \
  code/qualtrics_export/input/set2_inference_items.json
```

This creates:

- `code/qualtrics_export/input/set2_inference_items.json`

### Step 2: create the Qualtrics inference import

```bash
python3 code/qualtrics_export/qualtrics_inference_export.py \
  code/qualtrics_export/input/set2_inference_items.json \
  code/qualtrics_export/output/set2_inference_survey.txt
```

This creates:

- `code/qualtrics_export/output/set2_inference_survey.txt`
- `code/qualtrics_export/output/set2_inference_survey.audio_map.csv`

### Inference survey behavior

- The Qualtrics block is `[[Block:Inference Survey]]`
- The question asks:
  `Given Sentence 1, what can we say about Sentence 2?`
- Both displayed sentences are normalized, so no all-caps emphasis is visible
- The answer choices are fixed:
  - `Sentence 2 must be true.`
  - `Sentence 2 might be true.`
  - `Sentence 2 must be false.`
- The JSON label is mapped as:
  - `A -> Sentence 2 must be true.`
  - `B -> Sentence 2 might be true.`
  - `C -> Sentence 2 must be false.`

Example exported inference question:

```txt
[[Question:MC:SingleAnswer:Vertical]]
Given Sentence 1, what can we say about Sentence 2?<br><br>
Sentence 1: Sam only gave Rob oranges.<br>
Sentence 2: Sam also gave Rob apples.<br>
[[Choices]]
Sentence 2 must be true.
Sentence 2 might be true.
Sentence 2 must be false.
```

## Audio mapping CSV

Each export script also writes a sidecar CSV for manual Qualtrics attachment:

- `*.audio_map.csv`

The CSV includes:

- `question_number`
- `item_id`
- `label`
- `focus`
- `logic`
- `alternative`
- `s1`
- `s2`
- `audio_filename`
- `audio_path`
- `audio_exists`

Use this CSV as your checklist when manually attaching uploaded audio to the Qualtrics questions.

## Manual Qualtrics workflow

Recommended order:

1. Generate the subset JSON from the clip folder.
2. Generate the Qualtrics TXT import file.
3. Import the TXT file into Qualtrics.
4. Upload the corresponding audio files in Qualtrics.
5. Use the `*.audio_map.csv` file to attach the right audio file to each question.

## Custom subsets

If you want to build a survey from a hand-made JSON file instead of from a clip folder, keep these fields on each item so audio mapping still works cleanly:

- `source_json`
- `source_item_index`
- or `audio_file`

If those fields are missing, the exporter can still build the survey TXT, but the audio mapping CSV may not be able to resolve the right clip for mixed-source subsets.
