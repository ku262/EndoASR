import os
import json

methods = [
    "stage1-outputs",
    "stage2-outputs",
]

method_dirs = [
    "stage1-outputs",
    "stage2-outputs",
]

centers = ["C1", "C2", "C3", "C4", "C5"]
center_labels = ["C1", "C2", "C3", "C4", "C5", "Overall"]

aspects = [
    "(A) Prior medical history & indication for the current examination",
    "(B) Bowel preparation quality & intra-procedural patient status",
    "(C) Colorectal polyps lesions & interventions",
    "(D) Colorectal malignancy tumors & associated management",
    "(E) Inflammatory bowel disease & related descriptive findings",
    "(F) Conclusions & post-procedural plans"
]

JSONL_NAME = "asr_eval_results.jsonl"
TERM_DICT_PATH = "data/肠镜术语字典_flat.jsonl"

# 输出文件
OUT_DIR = ""
os.makedirs(OUT_DIR, exist_ok=True)
TERM_CACHE_OUT = os.path.join(OUT_DIR, "term_cache_by_key.jsonl")


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_terms(term_jsonl_path):

    terms = []
    seen = set()
    with open(term_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("term", "")
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            terms.append(t)

    terms.sort(key=len, reverse=True)
    return terms


def extract_terms_by_substring(text, terms):

    if not isinstance(text, str) or not text:
        return set()
    hit = set()
    for t in terms:
        if t in text:
            hit.add(t)
    return hit


def aspect_idx_from_local_index(i):

    return i // 10  # 0..5


def safe_get_pred(obj):

    for k in ["prediction_norm", "prediction", "pred", "hyp", "asr_text"]:
        v = obj.get(k, None)
        if isinstance(v, str):
            return v
    return ""


def safe_get_ref(obj):
    for k in ["reference", "ref", "gt", "text", "target"]:
        v = obj.get(k, None)
        if isinstance(v, str):
            return v
    return ""


def main(include_pred_text=True):
    terms = load_terms(TERM_DICT_PATH)
    print(f"[INFO] Loaded terms: {len(terms)} from {TERM_DICT_PATH}")

    overall_jsonl = os.path.join(method_dirs[0], JSONL_NAME)
    if not os.path.exists(overall_jsonl):
        raise FileNotFoundError(f"Overall jsonl not found: {overall_jsonl}")

    overall_items = read_jsonl(overall_jsonl)
    key2gold = {}
    for obj in overall_items:
        k = obj.get("key", None)
        if k is None:
            continue
        ref = safe_get_ref(obj)
        gold_terms = sorted(extract_terms_by_substring(ref, terms))
        key2gold[k] = gold_terms

    print(f"[INFO] Built key->gold_terms cache: {len(key2gold)} keys")

    with open(TERM_CACHE_OUT, "w", encoding="utf-8") as f:
        for k, gold_terms in key2gold.items():
            f.write(json.dumps({"key": k, "gold_terms": gold_terms}, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote term cache to: {TERM_CACHE_OUT}")

    for method_name, method_dir in zip(methods, method_dirs):
        out_path = os.path.join(OUT_DIR, f"term_sample_results_{method_name}.jsonl")
        n_written = 0
        n_skipped_missing_key = 0

        with open(out_path, "w", encoding="utf-8") as fout:

            for center_name, center_label in zip(centers, center_labels[:-1]):
                center_jsonl = os.path.join(method_dir, center_name, JSONL_NAME)
                if not os.path.exists(center_jsonl):
                    print(f"[WARN] Missing center jsonl: {center_jsonl} (skip center)")
                    continue

                items = read_jsonl(center_jsonl)
                if len(items) != 60:
                    print(f"[WARN] {method_name}-{center_label} expected 60 items, got {len(items)}")

                for i, obj in enumerate(items):
                    k = obj.get("key", None)
                    pred = safe_get_pred(obj)

                    aidx = aspect_idx_from_local_index(i)
                    aspect_name = aspects[aidx] if 0 <= aidx < len(aspects) else None

                    gold_terms = key2gold.get(k, None)
                    if k is None or gold_terms is None:

                        n_skipped_missing_key += 1
                        row = {
                            "method": method_name,
                            "center_label": center_label,
                            "center_name": center_name,
                            "local_index": i,
                            "aspect_idx": aidx,
                            "aspect": aspect_name,
                            "key": k,
                            "gold_n": None,
                            "hit_n": None,
                            "sample_term_accuracy": None,
                            "gold_terms": [],
                            "pred_terms": [],
                            "hit_terms": [],
                        }
                        if include_pred_text:
                            row["pred_text"] = pred
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_written += 1
                        continue

                    gold_set = set(gold_terms)
                    gold_n = len(gold_set)

                    pred_terms = extract_terms_by_substring(pred, terms)
                    hit_terms = sorted(gold_set & pred_terms)
                    hit_n = len(hit_terms)

                    sample_acc = (hit_n / gold_n) if gold_n > 0 else None

                    row = {
                        "method": method_name,
                        "center_label": center_label,
                        "center_name": center_name,
                        "local_index": i,
                        "aspect_idx": aidx,
                        "aspect": aspect_name,
                        "key": k,
                        "gold_n": gold_n,
                        "hit_n": hit_n,
                        "sample_term_accuracy": sample_acc,
                        "gold_terms": sorted(gold_set),
                        "pred_terms": sorted(pred_terms),
                        "hit_terms": hit_terms,
                    }
                    if include_pred_text:
                        row["pred_text"] = pred

                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_written += 1

        print(f"[INFO] Wrote {n_written} rows -> {out_path} (missing_key_rows={n_skipped_missing_key})")

    print("[DONE]")


if __name__ == "__main__":

    main(include_pred_text=True)
