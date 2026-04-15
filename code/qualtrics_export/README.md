# Qualtrics Survey Export Workflow

This folder contains the scripts for building Qualtrics import files from the experiment JSON items and the selected audio clips.

The current exporters are designed around stable question IDs and simple Qualtrics TXT imports:

- No `Embedded Data` is written into the Survey Flow.
- Each question ID is derived from the matching audio filename stem, e.g. `f10_item3`.
- Question order is shuffled at build time by default.
- Focus-survey answer choices are randomized in Qualtrics.

## Folder layout

- `build_survey_subset_from_clips.py`
  Builds a JSON subset from clip filenames like `f1_item0.wav`, `ns1_item0.wav`, or
  `speaker0_f1_item0.wav`.
- `build_focus24_survey.py`
  Builds the current 24-item multi-speaker focus survey and prepares its Qualtrics TXT import.
- `build_multispeaker_inference_survey.py`
  Builds a multi-speaker inference survey and prepares its Qualtrics TXT import.
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

- Source JSON items live in `data/stimuli/f*.json` and `data/stimuli/ns*.json`
- Focus-survey audio clips are selected from `data/clips/set1/`
- Inference-survey audio clips are selected from `data/clips/set2/`

Clip names determine which source item is selected:

- `f1_item0.wav` means item `0` from `data/stimuli/f1.json`
- `ns1_item0.wav` means item `0` from `data/stimuli/ns1.json`
- `speaker0_f1_item0.wav` means item `0` from `data/stimuli/f1.json`, with
  `speaker0` stored in the generated JSON metadata.

## What the subset builder adds

When you run `build_survey_subset_from_clips.py`, it copies the matching source JSON items into one smaller JSON array and adds:

- `source_json`
- `source_item_index`
- `audio_file`
- `S1_normalized`
- `S2_normalized`

The normalized fields are used for survey display so all-caps emphasis cues are removed from the visible text.

## Shared exporter behavior

Both exporters now do the following:

- Set the Qualtrics question ID from the audio clip stem, e.g. `f8_item5`.
- Write no `[[ED:...]]` tags.
- Write an `*.audio_map.csv` file for matching questions to uploaded audio.
- Shuffle the question order at export time by default.

You can disable build-time shuffling with:

```bash
--no-shuffle-questions
```

You can make a shuffled export reproducible with:

```bash
--seed 42
```

Qtip: The Qualtrics Advanced TXT format documents choice randomization, but does not document a tag for per-respondent question randomization. The exporters therefore shuffle the question list when building the import file.

## Focus survey workflow

Use this for the audio-perception survey based on `data/clips/set1/`.

### Current 24-item multi-speaker survey

Use this one-command workflow for the current focus survey:

```bash
python3 code/qualtrics_export/build_focus24_survey.py --seed 42
```

By default this builds:

- `speaker0:f1:0-7`
- `speaker1:ns1:0-7`
- `speaker2:ns2:0-7`

It creates:

- `code/qualtrics_export/input/focus_24/focus_24_items.json`
- `code/qualtrics_export/output/focus_24/focus_24_focus_survey.txt`
- `code/qualtrics_export/output/focus_24/focus_24_focus_survey.audio_map.csv`
- copied audio clips in `data/clips/focus_24/`

To keep questions in the selected block order instead of shuffling at build time:

```bash
python3 code/qualtrics_export/build_focus24_survey.py --no-shuffle-questions
```

To override the item selection, pass one or more `--selection` values:

```bash
python3 code/qualtrics_export/build_focus24_survey.py \
  --selection speaker0:f1:0-7 \
  --selection speaker1:ns1:0-7 \
  --selection speaker2:ns2:0-7
```

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
- The question ID is the audio filename stem, e.g. `f10_item3`
- The answer choices are the two content words from `Sentence 1` in sentence order
- The answer choices are normalized, so no all-caps emphasis is visible
- The answer choices are randomized in Qualtrics with `[[Randomize]]`

Example exported focus question:

```txt
[[Question:MC:SingleAnswer:Vertical]]
[[ID:f3_item6]]
In Sentence 1, which word was said with stronger emphasis?<br><br>
Sentence 1: Sam only gave Rob oranges.<br>
[[Randomize]]
[[Choices]]
Rob
oranges
```

## Inference survey workflow

Use this for the reasoning survey based on `data/clips/set2/`.

### Current multi-speaker inference survey

Use this one-command workflow for the currently available inference survey:

```bash
python3 code/qualtrics_export/build_multispeaker_inference_survey.py --seed 42
```

By default this builds the available clips for:

- `speaker0:f5:0-9`
- `speaker1:ns2:0-7`
- `speaker2:ns3:0-7`

It creates:

- `code/qualtrics_export/input/inference_multispeaker/items.json`
- `code/qualtrics_export/output/inference_multispeaker/inference_multispeaker_survey.txt`
- `code/qualtrics_export/output/inference_multispeaker/inference_multispeaker_survey.audio_map.csv`
- copied audio clips in `data/clips/inference_multispeaker/`

To build a 30-item version after `ns2` and `ns3` each have items 8 and 9
available for the selected speakers:

```bash
python3 code/qualtrics_export/build_multispeaker_inference_survey.py \
  --selection speaker0:f5:0-9 \
  --selection speaker1:ns2:0-9 \
  --selection speaker2:ns3:0-9 \
  --seed 42
```

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
- The question ID is the audio filename stem, e.g. `f8_item5`
- The answer choices are fixed:
  - `Sentence 2 must be true.`
  - `Sentence 2 might be true.`
  - `Sentence 2 must be false.`

Example exported inference question:

```txt
[[Question:MC:SingleAnswer:Vertical]]
[[ID:f1_item4]]
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
- `question_id`
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
