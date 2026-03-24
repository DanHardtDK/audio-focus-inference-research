# Qualtrics Focus Export

Structure:

- `input/`: put the JSON array files you want to convert here
- `qualtrics_focus_export.py`: converts JSON items into Qualtrics Advanced TXT
- `output/`: generated `.txt` files for Qualtrics import

Example:

```bash
python3 code/qualtrics_export/qualtrics_focus_export.py \
  code/qualtrics_export/input/my_focus_items.json \
  code/qualtrics_export/output/my_focus_survey.txt
```
