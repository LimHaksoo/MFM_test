import os
import re
import json
import glob
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from PIL import Image

from llava.utils import disable_torch_init
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# IMAGE_PLACEHOLDER는 버전에 따라 없을 수 있어서 안전하게 처리
try:
    from llava.constants import IMAGE_PLACEHOLDER
except Exception:
    IMAGE_PLACEHOLDER = "<image>"

from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        # dict로 감싸진 케이스면 흔한 key들 시도
        for k in ["data", "items", "annotations"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        raise ValueError("Unsupported JSON format: top-level dict but no known list key found.")


def normalize_label_text(s: str) -> Optional[int]:
    """문자열에서 normal/anomalous 라벨을 뽑아 0/1로 변환."""
    if s is None:
        return None
    t = s.strip().lower()

    # 첫 단어/첫 줄 위주로 보는 게 안정적
    t = t.splitlines()[0].strip()
    # 구두점 제거
    t = re.sub(r"[^a-zA-Z]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 엄격 매칭(가능하면 학습도 이 형태로 강제 추천)
    if t.startswith("anomal") or t.startswith("abnorm") or t.startswith("defect") or t.startswith("fail") or t == "ng":
        return 1
    if t.startswith("normal") or t.startswith("good") or t.startswith("pass") or t == "ok":
        return 0

    # 완화 매칭
    if "anomal" in t or "abnorm" in t or "defect" in t:
        return 1
    if "normal" in t or "good" in t:
        return 0

    return None


def extract_gt(item: Dict[str, Any]) -> Tuple[str, int, str]:
    """
    return: (id, y_true(0/1), image_rel_or_abs)
    지원 포맷:
      - {"id","image","label"}
      - {"id","image","conversations":[...,{"from":"gpt","value":"normal"}]}
    """
    _id = str(item.get("id", item.get("question_id", item.get("uid", ""))))

    image_path = item.get("image", item.get("image_path", item.get("img", None)))
    if image_path is None:
        raise ValueError(f"Missing image path in item id={_id}")

    # label이 숫자로 있으면 최우선
    if "label" in item:
        y = item["label"]
        if isinstance(y, str):
            y2 = normalize_label_text(y)
            if y2 is None:
                raise ValueError(f"Unrecognized label text: {y} (id={_id})")
            return _id, y2, image_path
        y = int(y)
        if y not in [0, 1]:
            raise ValueError(f"label must be 0/1 but got {y} (id={_id})")
        return _id, y, image_path

    # conversations에서 gpt 답을 GT로 사용
    conv = item.get("conversations", None)
    if isinstance(conv, list):
        gt_text = None
        for m in conv:
            frm = (m.get("from") or "").lower()
            if frm in ["gpt", "assistant"]:
                gt_text = m.get("value", "")
                break
        y2 = normalize_label_text(gt_text or "")
        if y2 is None:
            raise ValueError(f"Cannot infer GT from conversations (id={_id}) -> {gt_text}")
        return _id, y2, image_path

    raise ValueError(f"Cannot infer GT label for item id={_id}. Provide label or conversations.")


def build_query(model, question: str) -> str:
    """
    run_llava.py 패턴:
    - question 안에 <image> placeholder가 있으면 그걸 image token으로 치환
    - 없으면 앞에 image token + \n + question 붙임
    """
    qs = question
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in qs:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = re.sub(re.escape(IMAGE_PLACEHOLDER), image_token_se, qs)
        else:
            qs = re.sub(re.escape(IMAGE_PLACEHOLDER), DEFAULT_IMAGE_TOKEN, qs)
    else:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    return qs


@torch.inference_mode()
def predict_one(
    tokenizer,
    model,
    image_processor,
    image_abs_path: str,
    conv_mode: str,
    question: str,
    max_new_tokens: int = 8,
) -> str:
    # conversation prompt
    qs = build_query(model, question)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # load image
    image = Image.open(image_abs_path).convert("RGB")
    images = [image]
    image_sizes = [image.size]

    # preprocess image
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    # tokenize prompt with image token
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    # greedy decode (reproducible eval)
    output_ids = model.generate(
        input_ids=input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    # decode only newly generated tokens
    gen_ids = output_ids[0, input_ids.shape[-1]:]
    out = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return out


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / max(1, n)

    return {
        "total": n,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def load_model_for_eval(model_path: str, model_base: Optional[str], load_4bit: bool) -> Tuple[Any, Any, Any, int]:
    """
    LLaVA 버전별 load_pretrained_model 시그니처 차이를 최대한 흡수.
    (run_llava.py에서는 load_pretrained_model(model_path, model_base, model_name) 형태를 사용) :contentReference[oaicite:5]{index=5}
    """
    model_name = get_model_name_from_path(model_path)
    disable_torch_init()

    # 4bit 옵션은 버전에 따라 파라미터가 다를 수 있어서 try/except로 흡수
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name,
            load_4bit=load_4bit
        )
    except TypeError:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name
        )

    model.eval()
    return tokenizer, model, image_processor, context_len


def list_checkpoints(ckpt_root: str) -> List[str]:
    """
    - checkpoint-* 폴더들을 step 순서로 정렬
    - ckpt_root 자체도 (adapter가 있으면) "final"로 포함
    """
    ckpts = []

    # root 자체에 adapter_config가 있으면 포함
    if os.path.exists(os.path.join(ckpt_root, "adapter_config.json")) or \
       os.path.exists(os.path.join(ckpt_root, "non_lora_trainables.bin")):
        ckpts.append(ckpt_root)

    subs = glob.glob(os.path.join(ckpt_root, "checkpoint-*"))
    # step 기준 정렬
    def step_key(p: str) -> int:
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 10**18

    subs = sorted(subs, key=step_key)
    ckpts.extend(subs)
    return ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", type=str, required=True, help="output_dir (contains checkpoint-*)")
    ap.add_argument("--model_base", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--eval_data", type=str, required=True, help="json or jsonl containing eval samples")
    ap.add_argument("--image_folder", type=str, required=True, help="base folder for relative image paths")
    ap.add_argument("--output_dir", type=str, default="./eval_results_mfm")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--question", type=str, default="Is this product normal or anomalous? Answer with one word: normal or anomalous.")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    items = load_json_or_jsonl(args.eval_data)
    samples = []
    for it in items:
        _id, y, img = extract_gt(it)
        samples.append((_id, y, img))
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    ckpt_paths = list_checkpoints(args.ckpt_root)
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoints found under: {args.ckpt_root}")

    summary_rows = []

    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.basename(ckpt_path.rstrip("/"))
        out_dir = os.path.join(args.output_dir, ckpt_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Evaluating: {ckpt_path} ===")
        tokenizer, model, image_processor, _ = load_model_for_eval(
            model_path=ckpt_path,
            model_base=args.model_base,
            load_4bit=args.load_4bit,
        )

        y_true, y_pred = [], []
        pred_records = []
        unknown_count = 0

        for _id, gt, img_rel in tqdm(samples, desc=f"eval[{ckpt_name}]"):
            # abs path
            img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(args.image_folder, img_rel)
            pred_text = predict_one(
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                image_abs_path=img_abs,
                conv_mode=args.conv_mode,
                question=args.question,
                max_new_tokens=args.max_new_tokens,
            )
            pred = normalize_label_text(pred_text)
            if pred is None:
                # 파싱 실패는 일단 normal(0)로 두고 unknown 카운트
                pred = 0
                unknown_count += 1

            y_true.append(gt)
            y_pred.append(pred)

            pred_records.append({
                "id": _id,
                "image": img_rel,
                "y_true": gt,
                "y_pred": pred,
                "pred_text": pred_text,
            })

        metrics = compute_metrics(y_true, y_pred)
        metrics["unknown_parse_count"] = unknown_count
        metrics["checkpoint_path"] = ckpt_path

        # save per-ckpt
        with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
            for r in pred_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"[{ckpt_name}] acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
              f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  unknown={unknown_count}")

        summary_rows.append({
            "checkpoint": ckpt_name,
            "checkpoint_path": ckpt_path,
            **{k: metrics[k] for k in ["accuracy", "precision", "recall", "f1", "total", "unknown_parse_count"]}
        })

        # free gpu
        del model
        torch.cuda.empty_cache()

    # save summary
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] Summary saved to: {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
