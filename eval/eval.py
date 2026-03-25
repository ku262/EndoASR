import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json
from tqdm import tqdm
from jiwer import cer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import torchaudio
from pathlib import Path
import numpy as np
import re
import time

from funasr import AutoModel

NUM_MAP = {str(i): c for i, c in enumerate("零一二三四五六七八九")}

def int_to_chinese(n: int) -> str:

    if n < 10:
        return NUM_MAP[str(n)]
    elif n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return NUM_MAP[str(tens)] + "十"
        else:
            return NUM_MAP[str(tens)] + "十" + NUM_MAP[str(ones)]
    else:
        return str(n)  

def number_str_to_chinese(num_str: str) -> str:

    if "-" in num_str:  
        parts = num_str.split("-")
        return "至".join(number_str_to_chinese(p) for p in parts)
    if "." in num_str:
        int_part, dec_part = num_str.split(".")
        return "".join(NUM_MAP.get(d, d) for d in int_part) + "点" + "".join(NUM_MAP.get(d, d) for d in dec_part)
    return int_to_chinese(int(num_str))

def normalize_chinese_text(text: str) -> str:

    text = text.replace("\n", "").replace("\r", "").replace(" ", "")
    
    pattern_range = r"\d+\.?\d*(?:-\d+\.?\d*)?"
    text = re.sub(pattern_range, lambda m: number_str_to_chinese(m.group()), text)
    
    text = text.replace("CM", "厘米").replace("MM", "毫米").replace("M", "米")

    text = text.replace("cm", "厘米").replace("mm", "毫米").replace("m", "米")
    
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    
    return text

def evaluate_asr(model, jsonl_file, output_dir, device="cuda", max_samples=None, compute_bertscore=True):

    preds, refs = [], []
    num_samples = 0
    results = []

    if isinstance(jsonl_file, (str, Path)):
        jsonl_files = [Path(jsonl_file)]
    elif isinstance(jsonl_file, (list, tuple)):
        jsonl_files = [Path(f) for f in jsonl_file]
    else:
        raise ValueError("jsonl_file 必须是路径或路径列表")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_lines = []
    for file in jsonl_files:
        if not file.exists():
            print(f"⚠️ Warning: 文件不存在 {file}")
            continue
        with open(file, "r", encoding="utf-8") as f:
            all_lines.extend(f.readlines())

    print(f"共加载 {len(all_lines)} 条样本（来自 {len(jsonl_files)} 个文件）")

    smoothie = SmoothingFunction().method1
    bleu_scores, ref_lens, pred_lens, cer_scores = [], [], [], []
    bert_scores = None

    total_audio_time = 0.0  
    total_infer_time = 0.0  

    for idx, line in enumerate(tqdm(all_lines, desc="Evaluating ASR")):
        if max_samples and idx >= max_samples:
            break
        item = json.loads(line)

        raw_path = item["source"].replace("\\", "/")
        if "data/" in raw_path:
            rel_path = raw_path[raw_path.index("data/"):]
        else:
            rel_path = raw_path
        wav_path = Path(rel_path)
        ref_text = item["target"]

        if not wav_path.exists():
            print(f"⚠️ Warning: wav not found {wav_path}")
            continue

        ref_text_norm = normalize_chinese_text(ref_text)

        waveform, sr = torchaudio.load(str(wav_path))

        audio_dur = waveform.shape[1] / sr
        total_audio_time += audio_dur

        waveform = waveform.mean(0).to(device)
        
        t0 = time.time()
        res = model.generate(input=[waveform], is_final=True)
        infer_time = time.time() - t0
        total_infer_time += infer_time

        pred_text = res[0]["text"] if isinstance(res[0], dict) and "text" in res[0] else str(res[0])
        pred_text_norm = normalize_chinese_text(pred_text)


        preds.append(pred_text_norm)
        refs.append(ref_text_norm)
        num_samples += 1

        bleu1 = sentence_bleu(
            [list(ref_text_norm)],
            list(pred_text_norm),
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie
        )
        bleu_scores.append(bleu1)

        ref_len = len(ref_text_norm)
        pred_len = len(pred_text_norm)
        ref_lens.append(ref_len)
        pred_lens.append(pred_len)

        sample_cer = cer(ref_text_norm, pred_text_norm)
        cer_scores.append(sample_cer)

        results.append({
            "key": item.get("key", wav_path.stem),
            "reference": ref_text,
            "prediction": pred_text,
            "reference_norm": ref_text_norm,
            "prediction_norm": pred_text_norm,
            "cer": sample_cer,
            "bleu1": bleu1,
            "ref_len": ref_len,
            "pred_len": pred_len,
            "audio_duration": audio_dur,
            "infer_time": infer_time
        })

    overall_cer = float(np.mean(cer_scores))
    overall_bleu1 = float(np.mean(bleu_scores)) if bleu_scores else None
    avg_ref_len = float(np.mean(ref_lens)) if ref_lens else None
    avg_pred_len = float(np.mean(pred_lens)) if pred_lens else None
    rtf = total_infer_time / total_audio_time if total_audio_time > 0 else None


    if compute_bertscore and len(preds) > 0:
        P, R, F1 = bert_score(preds, refs, lang="zh", model_type="bert-base-chinese")
        bert_scores = F1.cpu().numpy().tolist()
        for i, f1 in enumerate(bert_scores):
            results[i]["bertscore"] = float(f1)
        overall_bertscore_mean = float(np.mean(bert_scores))
        overall_bertscore_std = float(np.std(bert_scores))
    else:
        overall_bertscore_mean = None
        overall_bertscore_std = None


    print(f"\n✅ Evaluation done on {num_samples} samples")
    print(f"Total audio time: {total_audio_time:.2f}s")
    print(f"Total inference time: {total_infer_time:.2f}s")
    print(f"RTF (Real-Time Factor): {rtf:.4f}")
    print(f"CER: {overall_cer*100:.2f}%")
    print(f"BLEU-1: {overall_bleu1:.4f}")
    if overall_bertscore_mean is not None:
        print(f"BERTScore (F1): {overall_bertscore_mean:.4f} (±{overall_bertscore_std:.4f})")
    print(f"Avg ref_len: {avg_ref_len:.2f}, Avg pred_len: {avg_pred_len:.2f}")

    out_jsonl = output_dir / "asr_eval_results.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {out_jsonl}")

    eval_summary = {
        "CER_mean": overall_cer,
        "CER_std": float(np.std(cer_scores)),
        "BLEU-1_mean": overall_bleu1,
        "BLEU-1_std": float(np.std(bleu_scores)),
        "BERTScore_mean": overall_bertscore_mean,
        "BERTScore_std": overall_bertscore_std,
        "ref_len_mean": avg_ref_len,
        "pred_len_mean": avg_pred_len,
        "num_samples": num_samples,
        "total_audio_time": total_audio_time,
        "total_infer_time": total_infer_time,
        "RTF": rtf,
        "input_files": [str(f) for f in jsonl_files]
    }
    config_json = output_dir / "config.json"
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)
    print(f"Evaluation summary saved to {config_json}")

    return eval_summary

# -------------------------
# demo
# -------------------------


data_type = ""
model_version = ""
model_path = "stage1_outputs"

model = AutoModel(
    model="paraformer-zh",
    device="cuda",
    model_path = model_path
)

output_dir = Path(f"Eval_output/{data_type}/{model_version}") 
os.makedirs(output_dir, exist_ok=True)
if data_type in ["test", "val"]:

    eval_res = evaluate_asr(model, f"data/{data_type}.jsonl", output_dir=output_dir, device="cuda", max_samples=None)
    eval_res = evaluate_asr(model, f"data/{data_type}_noise.jsonl", output_dir=output_dir, device="cuda", max_samples=None)

elif data_type == "retrospective":

    jsonl_files = [f"data/retrospective/jsonl/P1.jsonl",
                    f"data/retrospective/jsonl/P2.jsonl",
                    f"data/retrospective/jsonl/P3.jsonl",
                    f"data/retrospective/jsonl/P4.jsonl",
                    f"data/retrospective/jsonl/P5.jsonl",
                    f"data/retrospective/jsonl/P6.jsonl"]
    eval_res = evaluate_asr(model, jsonl_files, output_dir=output_dir, device="cuda", max_samples=None)

elif data_type == "prospective":
    jsonl_files = [
        f"data/prospective/jsonl/C1.jsonl",
        f"data/prospective/jsonl/C2.jsonl",
        f"data/prospective/jsonl/C3.jsonl",
        f"data/prospective/jsonl/C4.jsonl",
        f"data/prospective/jsonl/C5.jsonl"
    ]
    eval_res = evaluate_asr(model, jsonl_files, output_dir=output_dir, device="cuda", max_samples=None)
