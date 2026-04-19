# Qualtrics Export Workflow

This folder contains the production scripts for building Qualtrics Advanced TXT
imports from the experiment stimulus JSON files.

There are two supported production outputs:

- Survey 1: one small focus import and one small inference import.
- 24-item speaker surveys: one focus import and one inference import per speaker
  for the shared `ns1`-`ns3` stimulus set.

Run the full production build from the repository root:

```bash
python3 code/qualtrics_export/build_production_surveys.py
```

## Generated Files

Survey 1:

- `code/qualtrics_export/output/survey1/set1_focus_survey.txt`
- `code/qualtrics_export/output/survey1/set1_focus_survey.audio_map.csv`
- `code/qualtrics_export/output/survey1/set2_inference_survey.txt`
- `code/qualtrics_export/output/survey1/set2_inference_survey.audio_map.csv`

24-item speaker surveys:

- `code/qualtrics_export/output/focus_24_by_speaker/speaker0_ns1-ns3_focus_survey.txt`
- `code/qualtrics_export/output/focus_24_by_speaker/speaker1_ns1-ns3_focus_survey.txt`
- `code/qualtrics_export/output/focus_24_by_speaker/speaker2_ns1-ns3_focus_survey.txt`
- `code/qualtrics_export/output/inference_24_by_speaker/speaker0_ns1-ns3_inference_survey.txt`
- `code/qualtrics_export/output/inference_24_by_speaker/speaker1_ns1-ns3_inference_survey.txt`
- `code/qualtrics_export/output/inference_24_by_speaker/speaker2_ns1-ns3_inference_survey.txt`

## Source Data

Survey 1 uses curated item lists:

- `code/qualtrics_export/input/survey1/set1_focus_items.json`
- `code/qualtrics_export/input/survey1/set2_inference_items.json`

Survey 1 audio is resolved from:

- `data/speakers/speaker0/clips`

The 24-item speaker surveys are generated directly from:

- `data/stimuli/ns1.json`
- `data/stimuli/ns2.json`
- `data/stimuli/ns3.json`

Their question IDs are speaker-specific audio stems such as
`speaker1_ns2_item4`, so the imported Qualtrics questions can be matched to
`data/speakers/{speaker}/clips/ns{n}_item{i}.wav`.

## Build Options

Build only one output group:

```bash
python3 code/qualtrics_export/build_production_surveys.py --only survey1
python3 code/qualtrics_export/build_production_surveys.py --only focus24
python3 code/qualtrics_export/build_production_surveys.py --only inference24
```

The production question order is shuffled at build time with fixed seeds:

- Survey 1: `--survey1-seed 1`
- 24-item speaker surveys: `--speaker24-seed 42`

Use `--no-shuffle-questions` only for inspection builds.

## Export Behavior

All imports use Qualtrics Advanced TXT format.

Focus surveys:

- Block: `[[Block:Focus Survey]]`
- Question: `In Sentence 1, which word was said with stronger emphasis?`
- Displayed text: normalized Sentence 1 only, without uppercase focus cues.
- Choices: the two content words from Sentence 1, randomized in Qualtrics with
  `[[Randomize]]`.

Inference surveys:

- Block: `[[Block:Inference Survey]]`
- Question: `Given Sentence 1, what can we say about Sentence 2?`
- Displayed text: normalized Sentence 1 and Sentence 2, without uppercase focus
  cues.
- Choices:
  - `Sentence 2 must be true.`
  - `Sentence 2 might be true.`
  - `Sentence 2 must be false.`

The Survey 1 audio map CSV files are sidecars for manual audio attachment in
Qualtrics. They include question IDs, item metadata, source sentences, resolved
audio paths, and whether each audio file exists.
