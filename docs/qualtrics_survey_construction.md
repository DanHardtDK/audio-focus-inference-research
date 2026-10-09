# Qualtrics Survey Construction

This document describes how the Qualtrics survey import files are constructed
and how each question or audio clip can be traced back to the original stimulus
example.

## Production Build

Build all production Qualtrics files from the repository root:

```bash
python3 code/qualtrics_export/build_production_surveys.py
```

The production builder creates two groups of survey imports:

1. Survey 1, the small focus survey and small inference survey.
2. The 24-item speaker surveys, one focus and one inference import for each
   speaker.

The generated Qualtrics imports are Advanced TXT files in
`code/qualtrics_export/output/`.

## Survey Groups

### Survey 1

Survey 1 is built from two curated item JSON files:

- Focus: `code/qualtrics_export/input/survey1/set1_focus_items.json`
- Inference: `code/qualtrics_export/input/survey1/set2_inference_items.json`

The generated files are:

- `code/qualtrics_export/output/survey1/set1_focus_survey.txt`
- `code/qualtrics_export/output/survey1/set1_focus_survey.audio_map.csv`
- `code/qualtrics_export/output/survey1/set2_inference_survey.txt`
- `code/qualtrics_export/output/survey1/set2_inference_survey.audio_map.csv`

Survey 1 uses original speaker audio from:

- `data/speakers/speaker0/clips/`

Survey 1 question IDs are the original clip stems, for example `f10_item3`.
The matching audio file is `data/speakers/speaker0/clips/f10_item3.wav`.

### 24-Item Speaker Surveys

The 24-item surveys are built directly from:

- `data/stimuli/ns1.json`
- `data/stimuli/ns2.json`
- `data/stimuli/ns3.json`

These three files contain the same 24 stimulus items for every speaker. The
builder creates one focus import and one inference import for each speaker:

- `code/qualtrics_export/output/focus_24_by_speaker/speaker0_ns1-ns3_focus_survey.txt`
- `code/qualtrics_export/output/focus_24_by_speaker/speaker1_ns1-ns3_focus_survey.txt`
- `code/qualtrics_export/output/focus_24_by_speaker/speaker2_ns1-ns3_focus_survey.txt`
- `code/qualtrics_export/output/inference_24_by_speaker/speaker0_ns1-ns3_inference_survey.txt`
- `code/qualtrics_export/output/inference_24_by_speaker/speaker1_ns1-ns3_inference_survey.txt`
- `code/qualtrics_export/output/inference_24_by_speaker/speaker2_ns1-ns3_inference_survey.txt`

The speaker-specific audio files are stored in:

- `data/speakers/speaker0/clips/`
- `data/speakers/speaker1/clips/`
- `data/speakers/speaker2/clips/`

## Clip and Question Naming

All item indices are zero-based.

### Original Speaker Clips

Original speaker clips use:

```text
f{file_number}_item{item_index}.wav
```

Example:

```text
data/speakers/speaker0/clips/f10_item3.wav
```

This means:

- Source stimulus file: `data/stimuli/f10.json`
- Source item index: `3`
- Qualtrics question ID / clip stem: `f10_item3`

### New-Speaker 24-Item Clips

The shared 24-item stimulus files use:

```text
ns{set_number}_item{item_index}.wav
```

Example:

```text
data/speakers/speaker2/clips/ns3_item7.wav
```

This means:

- Speaker: `speaker2`
- Source stimulus file: `data/stimuli/ns3.json`
- Source item index: `7`

In the 24-item Qualtrics imports, the question ID adds the speaker prefix:

```text
speaker{speaker_number}_ns{set_number}_item{item_index}
```

Example:

```text
[[ID:speaker2_ns3_item7]]
```

This question uses:

```text
data/speakers/speaker2/clips/ns3_item7.wav
```

and stimulus item:

```text
data/stimuli/ns3.json, item index 7
```

### Tracing `ns` Items Back to Original Examples

The `ns1.json`, `ns2.json`, and `ns3.json` files include `original_file` and
`original_index` fields when an item was copied from the original 100-item set.
For example:

```json
{
  "source": "original",
  "original_file": "f4",
  "original_index": 1
}
```

This means the `ns` item came from:

```text
data/stimuli/f4.json, item index 1
```

or, using the clip-stem convention:

```text
f4_item1
```

## Survey Text Construction

The exported survey text removes the all-caps focus cue from displayed
sentences. The uppercase focus remains in the source JSON files for analysis
and traceability.

Focus survey questions show only normalized Sentence 1 and ask:

```text
In Sentence 1, which word was said with stronger emphasis?
```

The answer choices are the two content words from Sentence 1. Qualtrics
randomizes these answer choices with `[[Randomize]]`.

Inference survey questions show normalized Sentence 1 and Sentence 2 and ask:

```text
Given Sentence 1, what can we say about Sentence 2?
```

The answer choices are fixed:

- `Sentence 2 must be true.`
- `Sentence 2 might be true.`
- `Sentence 2 must be false.`

The label field `A` in the source JSON gives the correct answer key:

- `A`: Sentence 2 must be true.
- `B`: Sentence 2 might be true.
- `C`: Sentence 2 must be false.

## Question Order

The production import files are shuffled at build time with fixed seeds so the
output is reproducible:

- Survey 1: seed `1`
- 24-item speaker surveys: seed `42`

The sidecar `*.audio_map.csv` files for Survey 1 are written in the same order
as the Qualtrics import file. Use these CSVs as the attachment checklist when
adding uploaded audio to Qualtrics.

## Survey 1 Trace-Back Table

| Survey | Question ID / clip stem | Source JSON | Source item | S1 | S2 | Label |
|---|---|---|---:|---|---|---|
| Survey 1 focus | `f10_item3` | `f10.json` | 3 | Sam only gave Sue apples. | Sam didn't give Rob apples. | A |
| Survey 1 focus | `f1_item0` | `f1.json` | 0 | Sam only gave Bill grapes. | Sam also gave Sue grapes. | C |
| Survey 1 focus | `f2_item5` | `f2.json` | 5 | Sam only gave Sue apples. | Sam didn't give Sue oranges. | B |
| Survey 1 focus | `f2_item6` | `f2.json` | 6 | Sam only gave Sue grapes. | Sam didn't give Sue apples. | A |
| Survey 1 focus | `f3_item4` | `f3.json` | 4 | Sam only gave Ellen apples. | Sam didn't give Bill apples. | B |
| Survey 1 focus | `f3_item6` | `f3.json` | 6 | Sam only gave Rob oranges. | Sam also gave Rob apples. | B |
| Survey 1 focus | `f6_item4` | `f6.json` | 4 | Sam only gave Rob oranges. | Sam also gave Rob bananas. | C |
| Survey 1 focus | `f9_item8` | `f9.json` | 8 | Sam only gave Bill grapes. | Sam also gave Rob grapes. | B |
| Survey 1 inference | `f10_item4` | `f10.json` | 4 | Sam only gave Bill bananas. | Sam also gave Bill grapes. | B |
| Survey 1 inference | `f1_item4` | `f1.json` | 4 | Sam only gave Rob oranges. | Sam also gave Rob apples. | C |
| Survey 1 inference | `f1_item5` | `f1.json` | 5 | Sam only gave Sue grapes. | Sam didn't give Tom grapes. | A |
| Survey 1 inference | `f4_item2` | `f4.json` | 2 | Sam only gave Mary oranges. | Sam also gave Ellen oranges. | C |
| Survey 1 inference | `f8_item4` | `f8.json` | 4 | Sam only gave Ellen oranges. | Sam also gave Sue oranges. | B |
| Survey 1 inference | `f8_item5` | `f8.json` | 5 | Sam only gave Rob oranges. | Sam didn't give Rob apples. | A |
| Survey 1 inference | `f8_item9` | `f8.json` | 9 | Sam only gave Ellen grapes. | Sam didn't give Rob grapes. | B |
| Survey 1 inference | `f9_item3` | `f9.json` | 3 | Sam only gave Mary apples. | Sam didn't give Mary grapes. | B |

## 24-Item `ns1`-`ns3` Trace-Back Table

This table applies to all three speakers. Add the speaker prefix to get the
Qualtrics question ID, for example `ns2_item7` becomes
`speaker1_ns2_item7` in the speaker1 imports.

| ns item | Original source | S1 | S2 | Label | Design |
|---|---|---|---|---|---|
| `ns1_item0` | `f1_item0` | Sam only gave BILL grapes. | Sam also gave Sue grapes. | C | focus=1, alternative=1, logic=POS |
| `ns1_item1` | `f1_item9` | Sam only gave Sue ORANGES. | Sam didn't give Rob oranges. | B | focus=2, alternative=1, logic=NEG |
| `ns1_item2` | `f8_item1` | Sam only gave ROB grapes. | Sam also gave Rob oranges. | B | focus=1, alternative=2, logic=POS |
| `ns1_item3` | `f2_item4` | Sam only gave Tom ORANGES. | Sam didn't give Tom bananas. | A | focus=2, alternative=2, logic=NEG |
| `ns1_item4` | `f1_item6` | Sam only gave Ellen BANANAS. | Sam also gave Mary bananas. | B | focus=2, alternative=1, logic=POS |
| `ns1_item5` | `f3_item0` | Sam only gave ELLEN grapes. | Sam didn't give Tom grapes. | A | focus=1, alternative=1, logic=NEG |
| `ns1_item6` | `f1_item4` | Sam only gave Rob ORANGES. | Sam also gave Rob apples. | C | focus=2, alternative=2, logic=POS |
| `ns1_item7` | `f1_item3` | Sam only gave TOM apples. | Sam didn't give Tom grapes. | B | focus=1, alternative=2, logic=NEG |
| `ns2_item0` | `f7_item3` | Sam only gave BILL grapes. | Sam didn't give Bill oranges. | B | focus=1, alternative=2, logic=NEG |
| `ns2_item1` | `f5_item0` | Sam only gave Ellen GRAPES. | Sam also gave Ellen oranges. | C | focus=2, alternative=2, logic=POS |
| `ns2_item2` | `f4_item8` | Sam only gave BILL bananas. | Sam didn't give Tom bananas. | A | focus=1, alternative=1, logic=NEG |
| `ns2_item3` | `f8_item4` | Sam only gave Ellen ORANGES. | Sam also gave Sue oranges. | B | focus=2, alternative=1, logic=POS |
| `ns2_item4` | `f8_item2` | Sam only gave TOM apples. | Sam also gave Bill apples. | C | focus=1, alternative=1, logic=POS |
| `ns2_item5` | `f4_item5` | Sam only gave Bill BANANAS. | Sam didn't give Bill grapes. | A | focus=2, alternative=2, logic=NEG |
| `ns2_item6` | `f9_item0` | Sam only gave ELLEN bananas. | Sam also gave Ellen grapes. | B | focus=1, alternative=2, logic=POS |
| `ns2_item7` | `f8_item9` | Sam only gave Ellen GRAPES. | Sam didn't give Rob grapes. | B | focus=2, alternative=1, logic=NEG |
| `ns3_item0` | `f4_item4` | Sam only gave ROB bananas. | Sam also gave Rob oranges. | B | focus=1, alternative=2, logic=POS |
| `ns3_item1` | `f8_item3` | Sam only gave Mary APPLES. | Sam also gave Mary bananas. | C | focus=2, alternative=2, logic=POS |
| `ns3_item2` | `f2_item3` | Sam only gave SUE bananas. | Sam didn't give Rob bananas. | A | focus=1, alternative=1, logic=NEG |
| `ns3_item3` | `f6_item3` | Sam only gave Mary GRAPES. | Sam also gave Ellen grapes. | B | focus=2, alternative=1, logic=POS |
| `ns3_item4` | `f5_item1` | Sam only gave ELLEN apples. | Sam also gave Rob apples. | C | focus=1, alternative=1, logic=POS |
| `ns3_item5` | `f5_item9` | Sam only gave Tom BANANAS. | Sam didn't give Tom oranges. | A | focus=2, alternative=2, logic=NEG |
| `ns3_item6` | `f2_item5` | Sam only gave SUE apples. | Sam didn't give Sue oranges. | B | focus=1, alternative=2, logic=NEG |
| `ns3_item7` | `f4_item1` | Sam only gave Bill APPLES. | Sam didn't give Mary apples. | B | focus=2, alternative=1, logic=NEG |
