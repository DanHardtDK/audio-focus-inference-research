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

## Scoring Raw Inference Response CSVs

To score a raw Qualtrics response CSV for the 24-item inference survey, run:

```bash
python3 code/qualtrics_export/score_inference_24_csv.py /path/to/Inference_Survey.csv
```

By default this writes two CSVs to
`code/qualtrics_export/output/inference_scoring/`:

- `<input>.scored_long.csv`: one row per answered question with the raw numeric
  response, mapped `A/B/C` response, solution letter, and correctness flag.
- `<input>.scored_summary.csv`: one row per retained response with total correct,
  incorrect, accuracy, and whether the row contains all 24 survey answers.

The scorer is specific to the production 24-item inference survey. It derives
the solution key from `data/stimuli/ns1.json`, `ns2.json`, and `ns3.json`, and
derives the survey order from
`code/qualtrics_export/output/inference_24_by_speaker/speaker0_ns1-ns3_inference_survey.txt`.

It expects Qualtrics recode values where:

- `1` maps to `A`
- `2` maps to `B`
- `3` maps to `C`

The scorer ignores Qualtrics label/import rows that may appear below the header
in some raw CSV exports and scores only actual response rows whose `ResponseId`
looks like a Qualtrics record ID.

## Scoring Raw Focus Response CSVs

To score a raw Qualtrics response CSV for the 24-item focus survey, run:

```bash
python3 code/qualtrics_export/score_focus_24_csv.py /path/to/Focus_Survey.csv
```

By default this writes two CSVs to
`code/qualtrics_export/output/focus_scoring/`:

- `<input>.scored_long.csv`: one row per answered question with the response
  code, the chosen word, the correct word, and a correctness flag.
- `<input>.scored_summary.csv`: one row per retained response with totals,
  accuracy, missed question IDs, and missed-item details.

The scorer is specific to the production 24-item focus survey. It derives the
shared focus key from `data/stimuli/ns1.json`, `ns2.json`, and `ns3.json`, and
derives the survey order from
`code/qualtrics_export/output/focus_24_by_speaker/speaker0_ns1-ns3_focus_survey.txt`.

It expects focus responses where:

- source JSON `focus=1` maps to CSV response `1`
- source JSON `focus=2` maps to CSV response `2`

The scorer ignores Qualtrics label/import rows that may appear below the header
in some raw CSV exports and scores only actual response rows whose `ResponseId`
looks like a Qualtrics record ID.
