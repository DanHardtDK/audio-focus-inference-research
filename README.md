# Focus-Sensitive Inference from Speech

Code and data for "Can Audio LLMs Understand Prosodic Focus? An
Alternative-Semantics Inference Test with *Only*" — an inference test for
audio LLMs where the correct response depends on the location of a focal
accent.

## Contents

-   `code/` --- experimental scripts, including:
    -   `audioInput.py` --- main inference/focus-ID pipeline (OpenAI and
        Gemini backends)
    -   `analyze_accuracy.py`, `inference_stats.py` --- statistical
        significance tests (McNemar, exact binomial)
    -   `judge_claude.py` --- LLM-as-judge scoring of model explanations
    -   `speaker_acoustics.py` --- Praat/parselmouth acoustic analysis of
        speaker recordings
    -   `qwen/` --- layer-wise logistic-regression probing of open-weight
        Qwen audio models
    -   `qualtrics_export/` --- human survey export/scoring tools
-   `data/speakers/` --- speaker recordings (raw and clipped)
-   `data/stimuli/` --- sentence-pair stimuli (JSON) and survey definitions
-   `data/output/` --- per-run and master CSV/log outputs used in the paper
-   `data/humanSurvey/` --- anonymized human survey results (raw exports,
    which contain respondent PII, are gitignored and not included)
-   `results/` --- statistical analysis writeups
-   `docs/` --- data organization, recording protocol, and probing
    methodology notes

## Requirements

-   Python **3.10+**
-   Dependencies listed in `requirements.txt`

Install dependencies:

``` bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API Keys

Before running the scripts, set the required API keys as environment
variables:

``` bash
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_API_KEY="your_gemini_key"
export ANTHROPIC_API_KEY="your_claude_key"   # for judge_claude.py
```

## Example Run

``` bash
python code/audioInput.py f1 f2 --backend openai --model gpt-audio --mode audio
```

`input_paths` are file IDs (e.g. `f1 f2 f11`), resolved against `--wav-dir`
(default `data/speakers/speaker0/raw`) and `--json-dir` (default
`data/stimuli`). Run `python code/audioInput.py --help` for the full set of
options (few-shot, cross-validation, Gemini thinking budget/level, etc.).

See `commands.txt` for example commands for the Qwen probing pipeline.

## Notes

This repository was originally prepared for anonymous review purposes.
