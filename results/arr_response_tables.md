## Statistical analysis tables — for ARR/OpenReview author response

Note: OpenReview comment boxes render standard Markdown, including pipe tables
below. If a particular field strips Markdown, paste the plain-text fallback
version included at the end of each block instead.

---

### Table A. Model-vs-model comparisons (McNemar's exact test, Speaker 0, FS=0, N=128 matched items)

| Comparison | Acc. 1 | Acc. 2 | p (McNemar) |
|---|---|---|---|
| gemini-3.1-pro vs gemini-2.5-flash | 87.5% | 68.8% | <.0001 |
| gemini-3.1-pro vs gpt-4o-audio | 87.5% | 46.1% | <.0001 |
| gemini-3.1-pro vs gpt-audio | 87.5% | 39.1% | <.0001 |
| gemini-2.5-flash vs gpt-4o-audio | 68.8% | 46.1% | .0003 |
| gemini-2.5-flash vs gpt-audio | 68.8% | 39.1% | <.0001 |
| gpt-4o-audio vs gpt-audio | 46.1% | 39.1% | .2529 (n.s.) |

Plain-text fallback:
```
Comparison                              Acc.1   Acc.2   p (McNemar)
gemini-3.1-pro vs gemini-2.5-flash      87.5%   68.8%   <.0001
gemini-3.1-pro vs gpt-4o-audio          87.5%   46.1%   <.0001
gemini-3.1-pro vs gpt-audio             87.5%   39.1%   <.0001
gemini-2.5-flash vs gpt-4o-audio        68.8%   46.1%   .0003
gemini-2.5-flash vs gpt-audio           68.8%   39.1%   <.0001
gpt-4o-audio vs gpt-audio               46.1%   39.1%   .2529 (n.s.)
```

---

### Table B. Speaker 2 vs. other speakers (inference accuracy, FS=0)

| Model | Sp0 | Sp1 | Sp2 | p: Sp0 vs Sp2 | p: Sp1 vs Sp2 |
|---|---|---|---|---|---|
| gemini-3.1-pro-preview | 87.5% | 91.7% | 54.2% | .0004 (Fisher) | .0039 (McNemar) |
| gemini-2.5-flash | 68.8% | 58.3% | 41.7% | .0186 (Fisher) | .388 (McNemar, n.s.) |
| gpt-4o-audio-preview | 46.1% | 20.8% | 41.7% | .824 (n.s.) | .0625 (McNemar, marginal) |
| gpt-audio | 39.1% | 29.2% | 37.5% | 1.00 (n.s.) | .753 (n.s.) |

Sp0-vs-Sp2 uses an independent-samples test (different item pool from Sp1/Sp2); Sp1-vs-Sp2 uses McNemar's exact test (paired — Speakers 1 & 2 share the same 24 items).

Plain-text fallback:
```
Model                     Sp0     Sp1     Sp2     p(Sp0 vs Sp2)   p(Sp1 vs Sp2)
gemini-3.1-pro-preview    87.5%   91.7%   54.2%   .0004           .0039
gemini-2.5-flash          68.8%   58.3%   41.7%   .0186           .388 (n.s.)
gpt-4o-audio-preview      46.1%   20.8%   41.7%   .824 (n.s.)     .0625 (marginal)
gpt-audio                 39.1%   29.2%   37.5%   1.00 (n.s.)     .753 (n.s.)
```

---

### Table C. Few-shot trend (GEE logistic regression clustered by item, Speaker 0)

| Model | FS0 | FS2 | FS5 | p (trend) |
|---|---|---|---|---|
| gemini-3.1-pro-preview | 87.5% | 82.5% | 90.0% | .226 (n.s.) |
| gemini-2.5-flash | 68.8% | 53.6% | 60.0% | .240 (n.s.) |
| gpt-4o-audio-preview | 46.1% | 42.3% | 49.2% | .422 (n.s.) |
| gpt-audio | 39.1% | 39.1% | 45.0% | .256 (n.s.) |

Plain-text fallback:
```
Model                     FS0     FS2     FS5     p (trend)
gemini-3.1-pro-preview    87.5%   82.5%   90.0%   .226 (n.s.)
gemini-2.5-flash          68.8%   53.6%   60.0%   .240 (n.s.)
gpt-4o-audio-preview      46.1%   42.3%   49.2%   .422 (n.s.)
gpt-audio                 39.1%   39.1%   45.0%   .256 (n.s.)
```

---

### Table D. Human survey vs. gemini-3.1-pro-preview (one-sample t-test, real per-respondent scores)

| Speaker | n respondents | Human mean (SD) | gemini-3.1-pro acc. | p |
|---|---|---|---|---|
| Speaker 0 | 3 | 86.1% (6.4) | 87.5% | .742 (n.s.) |
| Speaker 1 | 4 | 63.5% (18.1) | 91.7% | .053 (marginal) |
| Speaker 2 | 5 | 60.0% (33.3) | 54.2% | .715 (n.s.) |

Plain-text fallback:
```
Speaker     n   Human mean(SD)   gemini-3.1-pro   p
Speaker 0   3   86.1% (6.4)      87.5%            .742 (n.s.)
Speaker 1   4   63.5% (18.1)     91.7%            .053 (marginal)
Speaker 2   5   60.0% (33.3)     54.2%            .715 (n.s.)
```

---

### Table E. Acoustic comparison across speakers (descriptive; ns1-ns3 shared clips)

| Speaker | N | F0 mean | F0 range | F0 std | Intensity range | Intensity std |
|---|---|---|---|---|---|---|
| Speaker 0 | 24 | 117.6 Hz | 366.5 Hz | 65.1 Hz | 57.2 dB | 16.5 dB |
| Speaker 1 | 24 | 149.1 Hz | 190.9 Hz | 30.7 Hz | 46.0 dB | 14.0 dB |
| Speaker 2 | 24 | 193.0 Hz | 451.2 Hz | 81.3 Hz | 25.3 dB | 6.6 dB |

Plain-text fallback:
```
Speaker     N    F0 mean   F0 range   F0 std   Int range   Int std
Speaker 0   24   117.6Hz   366.5Hz    65.1Hz   57.2dB      16.5dB
Speaker 1   24   149.1Hz   190.9Hz    30.7Hz   46.0dB      14.0dB
Speaker 2   24   193.0Hz   451.2Hz    81.3Hz   25.3dB       6.6dB
```
