"""
qwen_inspect.py — Inspect Qwen2-Audio-7B internals for focus inference.

Two analyses:
  1. LOGITS  — softmax probability over A / B / C at the first generated token,
               for every item clip.  Saved to {output-dir}/qwen_logits_{timestamp}.csv

  2. ENCODER — mean-pooled audio-encoder output vector (1280-dim) for every clip.
               Saved to {output-dir}/qwen_encoder_{timestamp}.csv
               Also prints cosine-similarity between every pair and saves a
               PNG heatmap to {output-dir}/qwen_encoder_heatmap_{timestamp}.png

Usage (from repo root):
    python code/qwen/qwen_inspect.py --file f1 [--file f2 ...] [--speaker speaker0]

The clips directory defaults to data/speakers/{speaker}/clips/
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

# Shared loader (handles precision + Qwen2-Audio / Qwen2.5-Omni families)
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_loader import load_qwen, forward_for_logits, encode_audio  # noqa: E402

MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
TARGET_SR  = 16000   # Whisper encoder sampling rate


# ---------------------------------------------------------------------------
# Model loading (cached on module level so multiple calls reuse it)
# ---------------------------------------------------------------------------
_model = None
_processor = None
_family = None

def load_model(model_name: str = MODEL_NAME, precision: str = "4bit"):
    global _model, _processor, _family
    if _model is None:
        _model, _processor, _sr, _family = load_qwen(model_name, precision)
        # qwen_inspect uses the module-level TARGET_SR; honour the processor's value if different.
        global TARGET_SR
        TARGET_SR = _sr
    return _model, _processor


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(path):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:          # stereo / multi-channel → mix down to mono
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    return arr


def build_inputs(processor, audio_arr, prompt_text):
    """Build model inputs for a single audio + text prompt."""
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a speech classification assistant. "
                "Listen to the audio clip which contains one sentence pair (S1 and S2). "
                "Classify S2 relative to S1: A = entailed, B = independent, C = contradicted. "
                "Reply with exactly one character: A, B, or C."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "item.wav"},
                {"type": "text",  "text": prompt_text},
            ],
        },
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[text],
        audio=[audio_arr],
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(_model.device) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Analysis 1 — logit probabilities for A / B / C
# ---------------------------------------------------------------------------

def get_abc_probs(inputs, processor):
    """
    Run one forward pass and return softmax probabilities for tokens A, B, C
    at the first generated position.
    Returns dict with keys 'p_A', 'p_B', 'p_C', 'pred'.
    """
    tok = processor.tokenizer
    # Token IDs for single-char labels
    id_A = tok.encode("A", add_special_tokens=False)[0]
    id_B = tok.encode("B", add_special_tokens=False)[0]
    id_C = tok.encode("C", add_special_tokens=False)[0]

    with torch.no_grad():
        logits = forward_for_logits(_model, _family, inputs)  # (1, seq_len, vocab)

    # last position = where the model would predict the next token
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits.float(), dim=-1)

    p_A = probs[id_A].item()
    p_B = probs[id_B].item()
    p_C = probs[id_C].item()

    # Renormalise over {A,B,C} only
    total = p_A + p_B + p_C
    p_A /= total; p_B /= total; p_C /= total

    pred = max(zip([p_A, p_B, p_C], ["A", "B", "C"]))[1]
    return {"p_A": p_A, "p_B": p_B, "p_C": p_C, "pred": pred}


# ---------------------------------------------------------------------------
# Analysis 2 — audio encoder representations
# ---------------------------------------------------------------------------

def _pool_hidden(t):
    """
    Mean-pool a hidden-state tensor over the time dimension and return a
    1-D numpy vector of shape (hidden_size,).
    Handles both shapes returned by different audio towers:
      Qwen2-Audio (batched):  (B, T, H) -> take batch 0 -> (T, H) -> mean(0)
      Qwen2.5-Omni (flat):    (T, H)                            -> mean(0)
    """
    if t.dim() == 3:
        t = t[0]            # (T, H)
    # t is now (T, H)
    return t.mean(dim=0).float().cpu().numpy()


def get_encoder_repr(inputs):
    """
    Run only the audio tower and return a mean-pooled 1-D numpy vector.
    Shape: (hidden_size,) — 1280 for Qwen2-Audio.
    """
    enc_out = encode_audio(_model, _family, inputs, output_hidden_states=False)
    return _pool_hidden(enc_out.last_hidden_state)


def get_all_layer_reprs(inputs):
    """
    Run the audio tower with output_hidden_states=True and return a list of
    mean-pooled vectors for every hidden state.
    Index 0 = conv/embedding output, indices 1–N = transformer layers.
    Each element is a numpy array of shape (hidden_size,).
    """
    enc_out = encode_audio(_model, _family, inputs, output_hidden_states=True)
    return [_pool_hidden(hs) for hs in enc_out.hidden_states]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    action="append", required=True,
                        dest="files", metavar="FILEID",
                        help="File ID to process (e.g. f1). Repeat for multiple.")
    parser.add_argument("--model",     default=MODEL_NAME,
                        help="HuggingFace model id (default: Qwen/Qwen2-Audio-7B-Instruct)")
    parser.add_argument("--precision", default="4bit",
                        choices=["4bit", "8bit", "fp16", "bf16"],
                        help="Model precision (default: 4bit, matches original runs)")
    parser.add_argument("--speaker", default="speaker0",
                        help="Speaker folder (default: speaker0)")
    parser.add_argument("--json-dir", default="data/stimuli",
                        help="Directory containing .json stimulus files")
    parser.add_argument("--wav-dir",  default=None,
                        help="Clips directory (default: data/speakers/{speaker}/clips)")
    parser.add_argument("--output-dir", default=os.path.join("data", "output", "qwen"),
                        help="Directory for output CSVs and heatmap (default: data/output/qwen)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Extract encoder representations from ALL hidden layers "
                             "(index 0 = embedding, 1-32 = transformer); "
                             "saves qwen_encoder_layers_*.csv in addition to the final-layer CSV")
    args = parser.parse_args()

    clips_dir = args.wav_dir or os.path.join(
        "data", "speakers", args.speaker, "clips"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    ts      = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sp_tag  = args.speaker                          # e.g. "speaker0"
    fl_tag  = "-".join(args.files)                  # e.g. "f1-f2-f3" or "ns1-ns2-ns3"
    tag     = f"{sp_tag}_{fl_tag}_{ts}"

    logits_path  = os.path.join(args.output_dir, f"qwen_logits_{tag}.csv")
    encoder_path = os.path.join(args.output_dir, f"qwen_encoder_{tag}.csv")
    heatmap_path = os.path.join(args.output_dir, f"qwen_encoder_heatmap_{tag}.png")
    layers_path  = os.path.join(args.output_dir, f"qwen_encoder_layers_{tag}.csv")

    model, processor = load_model(args.model, args.precision)

    logit_rows   = []
    encoder_rows = []   # each row: label info + flat vector
    layer_rows   = []   # stacked all-layer rows (only populated with --all-layers)
    clip_labels  = []   # for heatmap axis labels
    encoder_vecs = []

    for file_id in args.files:
        json_path = os.path.join(args.json_dir, file_id + ".json")
        if not os.path.exists(json_path):
            print(f"WARNING: {json_path} not found, skipping {file_id}")
            continue

        with open(json_path) as f:
            items = json.load(f)

        for idx, item in enumerate(items):
            clip_path = os.path.join(clips_dir, f"{file_id}_item{idx}.wav")
            if not os.path.exists(clip_path):
                print(f"  WARNING: clip not found: {clip_path}")
                continue

            print(f"  {file_id} item {idx} …", end=" ", flush=True)
            audio_arr = load_audio(clip_path)
            prompt = "Classify S2 relative to S1. Reply with A, B, or C only."
            inputs = build_inputs(processor, audio_arr, prompt)

            # --- Logits ---
            abc = get_abc_probs(inputs, processor)
            row = {
                "file_id":     file_id,
                "item_idx":    idx,
                "true_A":      item.get("A", ""),
                "focus":       item.get("focus", ""),
                "logic":       item.get("logic", ""),
                "alternative": item.get("alternative", ""),
                "S1":          item.get("S1", ""),
                "S2":          item.get("S2", ""),
                **abc,
                "correct":     int(abc["pred"] == item.get("A", "")),
            }
            logit_rows.append(row)

            # --- Encoder ---
            if args.all_layers:
                layer_vecs = get_all_layer_reprs(inputs)
                vec = layer_vecs[-1]
                meta = {
                    "file_id":     file_id,
                    "item_idx":    idx,
                    "true_A":      item.get("A", ""),
                    "focus":       item.get("focus", ""),
                    "logic":       item.get("logic", ""),
                    "alternative": item.get("alternative", ""),
                    "S1":          item.get("S1", ""),
                }
                for layer_idx, lv in enumerate(layer_vecs):
                    layer_rows.append({**meta, "layer": layer_idx,
                                       **{f"d{i}": v for i, v in enumerate(lv)}})
            else:
                vec = get_encoder_repr(inputs)
            encoder_rows.append({
                "file_id":     file_id,
                "item_idx":    idx,
                "true_A":      item.get("A", ""),
                "focus":       item.get("focus", ""),
                "logic":       item.get("logic", ""),
                "alternative": item.get("alternative", ""),
                "S1":          item.get("S1", ""),
                **{f"d{i}": v for i, v in enumerate(vec)},
            })
            encoder_vecs.append(vec)
            clip_labels.append(f"{file_id}[{idx}] {item.get('A','')} f{item.get('focus','')}")

            print(f"pred={abc['pred']} (A={abc['p_A']:.2f} B={abc['p_B']:.2f} C={abc['p_C']:.2f})")

    # --- Write logits CSV ---
    if logit_rows:
        fieldnames = list(logit_rows[0].keys())
        with open(logits_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(logit_rows)
        print(f"\n✓ Logits CSV: {logits_path}")

        # Print summary table
        print(f"\n{'Item':<12} {'S1':<40} {'True':>5} {'Pred':>5} {'p_A':>6} {'p_B':>6} {'p_C':>6}")
        print("-" * 80)
        for r in logit_rows:
            mark = "✓" if r["correct"] else "✗"
            print(f"{r['file_id']}[{r['item_idx']}] {mark}  "
                  f"{r['S1'][:36]:<36}  {r['true_A']:>4}  {r['pred']:>4}  "
                  f"{r['p_A']:>5.2f}  {r['p_B']:>5.2f}  {r['p_C']:>5.2f}")

    # --- Write encoder CSV ---
    if encoder_rows:
        fieldnames = list(encoder_rows[0].keys())
        with open(encoder_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(encoder_rows)
        print(f"\n✓ Encoder CSV: {encoder_path}")

    # --- Cosine-similarity heatmap ---
    if len(encoder_vecs) >= 2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            vecs = np.stack(encoder_vecs)               # (N, D)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs_n = vecs / (norms + 1e-8)
            sim = vecs_n @ vecs_n.T                     # cosine similarity matrix

            fig, ax = plt.subplots(figsize=(max(6, len(clip_labels) * 0.6),
                                            max(5, len(clip_labels) * 0.55)))
            im = ax.imshow(sim, vmin=0.7, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(clip_labels)))
            ax.set_yticks(range(len(clip_labels)))
            ax.set_xticklabels(clip_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(clip_labels, fontsize=7)
            ax.set_title("Audio encoder cosine similarity (mean-pooled)\nQwen2-Audio-7B-Instruct")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=150)
            print(f"✓ Heatmap PNG: {heatmap_path}")
        except ImportError:
            print("matplotlib not installed — skipping heatmap. Run: pip install matplotlib")

    # --- Write all-layers encoder CSV ---
    if args.all_layers and layer_rows:
        fieldnames = list(layer_rows[0].keys())
        with open(layers_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(layer_rows)
        print(f"\n\u2713 All-layers encoder CSV: {layers_path}")


if __name__ == "__main__":
    main()
