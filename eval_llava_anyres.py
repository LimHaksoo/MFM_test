import argparse
import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from collections import defaultdict

# LLaVA 관련 모듈 임포트
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from PIL import Image
from pathlib import Path

# 기존 데이터 로더 (사용자 환경에 맞게 경로 확인 필요)
import sys
sys.path.append(os.getcwd()) # 현재 경로 추가
from dataset import make_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Anomaly Detection Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LoRA checkpoint or merged model")
    parser.add_argument("--model_base", type=str, default="liuhaotian/llava-v1.6-vicuna-7b", help="Base model (required for LoRA)")
    parser.add_argument("--visa_root", type=str, required=True, help="Path to ViSA dataset root")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--mm_patch_merge_type", type=str, default=None)  # optional override
    parser.add_argument("--image_grid_pinpoints", type=str, default=None,
                    help="Only needed if model config lacks image_grid_pinpoints. "
                         "Pass same value used in training/config.json.")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    return parser.parse_args()

def get_loss(model, input_ids, attention_mask, images, labels, image_sizes=None):
    """
    특정 텍스트 시퀀스(질문+정답)에 대한 모델의 Loss를 계산합니다.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=labels,
            image_sizes=image_sizes,  
        )
    return outputs.loss.item()

def prepare_input(tokenizer, model, image, meta, label, target_type, conv_mode):
    """
    [수정됨] 질문(Prompt) 부분을 -100으로 마스킹하여 답변(Response)의 Loss만 계산하도록 함
    """
    # 1. 질문 생성
    qs = "Inspect this image for manufacturing defects. Is this object normal or anomalous? If anomalous, describe the defects."
    if DEFAULT_IMAGE_TOKEN not in qs:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # 2. 답변 생성
    if target_type == 'normal':
        target_response = "The object is normal."
    else:
        target_response = "The object is anomalous."

    # 3. 전체 대화(질문+답변) 생성
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], target_response)
    prompt_full = conv.get_prompt()

    # 4. 질문 부분만 따로 생성 (길이 측정용)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_q = conv.get_prompt()

    # 5. 토크나이징
    input_ids = tokenizer_image_token(prompt_full, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    # 질문 길이 계산 (답변이 시작되는 위치 찾기)
    # 주의: tokenizer 설정에 따라 토큰화 방식이 다를 수 있어, 직접 두 번 토크나이징해서 비교합니다.
    tokenized_q = tokenizer_image_token(prompt_q, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    len_q = tokenized_q.shape[0]

    # 6. Loss 계산용 타겟 생성 및 마스킹
    targets = input_ids.clone()
    # 질문 구간(0 ~ len_q)은 -100으로 채워서 Loss 계산 제외 (Ignore Index)
    targets[0, :len_q] = -100 

    return input_ids, targets

def main():
    args = parse_args()
    disable_torch_init()
    
    # 1. 모델 로드 (LoRA 어댑터 포함)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        device_map=None,
        device="cuda",
        load_4bit=False  # 학습 시 4bit를 썼으므로 평가도 4bit 권장 (메모리 절약)
    )

    from accelerate import infer_auto_device_map, dispatch_model
    import torch

    # GPU 개수/메모리 잡기
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, "멀티 GPU 샤딩하려면 GPU가 2장 이상 필요합니다."

    # GPU0에는 CLIP이 올라갈 거니까 LLM용 메모리를 좀 비워둠 (예: 3~4GB 여유)
    max_memory = {i: "10GiB" for i in range(n_gpus)}     # <- 각 GPU VRAM에 맞게 조절
    max_memory[0] = "6GiB"                                # <- GPU0은 CLIP 자리 남기기 (중요)

    # LLaMA는 레이어를 쪼개면 안 되므로 no_split 지정
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype=torch.float16,
    )

    model = dispatch_model(model, device_map=device_map)
    print("LLM device_map =", getattr(model, "hf_device_map", None))
    vt = model.get_vision_tower()
    proj = model.get_model().mm_projector.to(vt.device)
    print("vision_tower device:", vt.device, "dtype:", vt.dtype)
    print("mm_projector device:", next(proj.parameters()).device, "dtype:", next(proj.parameters()).dtype)
    # transformers device_map 확인(있으면)
    print("hf_device_map:", getattr(model, "hf_device_map", None))

    model.config.image_aspect_ratio = args.image_aspect_ratio
    if args.mm_patch_merge_type is not None:
        model.config.mm_patch_merge_type = args.mm_patch_merge_type

    if args.image_aspect_ratio == "anyres":
        # process_images(anyres)가 이 값을 씁니다.
        if not hasattr(model.config, "image_grid_pinpoints") or model.config.image_grid_pinpoints is None:
            if args.image_grid_pinpoints is None:
                raise ValueError("anyres requires model.config.image_grid_pinpoints. "
                                "Pass --image_grid_pinpoints or ensure it's in config.json.")
            model.config.image_grid_pinpoints = args.image_grid_pinpoints

    
    # 2. 데이터 로더 준비
    test_loader = make_dataloader(
        root=args.visa_root,
        dataset_name="visa",
        split="test",
        batch_size=args.batch_size, # Loss 비교 방식은 batch 1이 안전함
        num_workers=4,
        image_size=336, # LLaVA v1.5 기본 해상도
        return_mask=False,
        shuffle=False
    )

    # 3. 평가 루프
    target_normal = "Prediction: normal. Rationale: no visible defects."
    target_anom = "Prediction: anomalous. Rationale: visible defect patterns present."

    gts = []
    preds = []
    anomaly_print_count = 0
    normal_print_count = 0
    target_count = 100
    normal_diffs = []    # 정상 이미지들의 (Loss_Anom - Loss_Norm) 값
    anomaly_diffs = []   # 불량 이미지들의 (Loss_Anom - Loss_Norm) 값
    category_results = defaultdict(list)
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("Start Evaluation...")
    def resolve_image_path(meta, visa_root):
        # meta에 들어있는 경로 키는 구현마다 달라서 여러 후보를 순서대로 봅니다.
        for k in ["img_path", "image_path", "path", "file_path", "filepath", "file_name", "filename", "image"]:
            if k in meta and meta[k]:
                p = str(meta[k])
                if os.path.isabs(p):
                    return p
                return str(Path(visa_root) / p)
        raise KeyError(f"Cannot find image path in meta keys={list(meta.keys())}")

    for batch in tqdm(test_loader):
        # 배치 사이즈가 1이라고 가정
        # image_tensor = batch["image"].to(model.device, dtype=torch.float16)
        image_tensor = batch["image"].to(vt.device, dtype=vt.dtype)
        # image_tensor_flipped = torch.flip(image_tensor, [3])
        label = int(batch["label"].item())
        meta = batch["meta"] # list of dict, but batch=1 -> dict 접근 필요
        if isinstance(meta, list): 
            meta = meta[0]

        if args.image_aspect_ratio == "anyres":
            # 1) PIL 원본 로드
            img_path = resolve_image_path(meta, args.visa_root)
            pil_img = Image.open(img_path).convert("RGB")

            # 2) anyres 패치 전처리 + 원본 사이즈 기록 (w,h)
            image_sizes = [pil_img.size]  # [(width, height)]
            images = process_images([pil_img], image_processor, model.config)

            # 3) vision_tower 디바이스/타입으로 이동
            if isinstance(images, torch.Tensor):
                image_tensor = images.to(vt.device, dtype=vt.dtype)
            else:
                # (패치 개수가 이미지마다 다르면 list로 올 수 있음)
                image_tensor = [x.to(vt.device, dtype=vt.dtype) for x in images]
        else:
            image_sizes = None
            image_tensor = batch["image"].to(vt.device, dtype=vt.dtype)
        
        # 4. Normal 가설 검증
        input_ids_n, targets_n = prepare_input(tokenizer, model, image_tensor, meta, label, 'normal', args.conv_mode)
        loss_normal = get_loss(model, input_ids_n, None, image_tensor, targets_n, image_sizes=image_sizes)

        # 5. Anomaly 가설 검증 ("The object is anomalous.")
        input_ids_a, targets_a = prepare_input(tokenizer, model, image_tensor, meta, label, 'anomalous', args.conv_mode)
        loss_anom = get_loss(model, input_ids_a, None, image_tensor, targets_a, image_sizes=image_sizes)
        # # ---------------------------------------------------------
        # # [2] 뒤집은 이미지 평가 (검증용)
        # # ---------------------------------------------------------
        # # 같은 질문으로 뒤집은 이미지도 넣어봅니다.
        # loss_normal_flip = get_loss(model, input_ids_n, None, image_tensor_flipped, targets_n)
        # loss_anom_flip = get_loss(model, input_ids_a, None, image_tensor_flipped, targets_a)

        # # ---------------------------------------------------------
        # # [3] 점수 합산 (앙상블)
        # # ---------------------------------------------------------
        # # 원본과 반전 이미지의 Loss를 평균 냅니다. (노이즈가 줄어듭니다!)
        # loss_normal = (loss_normal + loss_normal_flip) / 2
        # loss_anom = (loss_anom + loss_anom_flip) / 2


        # 6. 예측 (Loss가 낮은 쪽 선택)
        pred = 0 if loss_normal < loss_anom else 1
        diff = loss_anom - loss_normal
        
        anomaly_score = loss_normal - loss_anom 
        
        # 3. 결과 수집 (카테고리별로 저장)
        category = meta.get('category', 'unknown')
        category_results[category].append((label, anomaly_score))
        
        # ============ [디버깅 코드 시작] ============
        # 실제 정답이 '불량(1)'인데, 모델이 헷갈려하는지 확인
        
        if label == 0:  # 딱 10개만 찍어봅니다.
            print(f"\n======== [Anomaly Case Check #{normal_print_count+1}] ========")
            print(f"📸 Image Meta: {meta}")
            print(f"📉 Loss (Normal Sentence): {loss_normal:.4f}")
            print(f"📉 Loss (Anomaly Sentence): {loss_anom:.4f}")
            
            if diff > 0:
                print(f"✅ 결과: 성공! (정상 문장의 Loss가 {abs(diff):.4f}만큼 더 낮음)")
            else:
                print(f"❌ 결과: 실패... (불량 문장을 더 선호함, 차이: {diff:.4f})")
            
            print(f"🤖 모델 예측: {'Anomalous' if pred==1 else 'Normal'} (정답: Normal)")
            print("========================================================\n")
            normal_print_count += 1        
            normal_diffs.append(diff)
        
        if label == 1:  # 딱 10개만 찍어봅니다.
            print(f"\n======== [Anomaly Case Check #{anomaly_print_count+1}] ========")
            print(f"📸 Image Meta: {meta}")
            print(f"📉 Loss (Normal Sentence): {loss_normal:.4f}")
            print(f"📉 Loss (Anomaly Sentence): {loss_anom:.4f}")
            
            if diff < 0:
                print(f"✅ 결과: 성공! (불량 문장의 Loss가 {abs(diff):.4f}만큼 더 낮음)")
            else:
                print(f"❌ 결과: 실패... (정상 문장을 더 선호함, 차이: {diff:.4f})")
            
            print(f"🤖 모델 예측: {'Anomalous' if pred==1 else 'Normal'} (정답: Anomalous)")
            print("========================================================\n")
            anomaly_print_count += 1
            anomaly_diffs.append(diff)
        # ============ [디버깅 코드 끝] ============
        
        gts.append(label)
        preds.append(pred)

    # 7. 메트릭 계산
    gts = np.array(gts)
    preds = np.array(preds)

    tp = int(((preds==1)&(gts==1)).sum())
    fp = int(((preds==1)&(gts==0)).sum())
    tn = int(((preds==0)&(gts==0)).sum())
    fn = int(((preds==0)&(gts==1)).sum())

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-8, (precision + recall))
    acc = (tp + tn) / len(gts)

    print(f"Total: {len(gts)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 결과 저장
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
    
    norm_arr = np.array(normal_diffs)
    anom_arr = np.array(anomaly_diffs)

    print("\n======== [Calibration Result] ========")
    print(f"🟢 Normal Case Diffs (Mean): {norm_arr.mean():.4f} (Std: {norm_arr.std():.4f})")
    print(f"🔴 Anomaly Case Diffs (Mean): {anom_arr.mean():.4f} (Std: {anom_arr.std():.4f})")
    
    # 0을 기준으로 했을 때의 정확도
    # (원래 로직: diff > 0 이면 Normal 예측)
    acc_naive_norm = (norm_arr > 0).sum() / len(norm_arr) if len(norm_arr) > 0 else 0
    acc_naive_anom = (anom_arr <= 0).sum() / len(anom_arr) if len(anom_arr) > 0 else 0
    print(f"📉 기준점 0.0 일 때 Accuracy -> Normal: {acc_naive_norm*100:.1f}%, Anomaly: {acc_naive_anom*100:.1f}%")

    # 💡 최적 Threshold 찾기 (간단한 버전: 두 평균의 중간값)
    # 정교하게 하려면 ROC Curve를 그려서 Youden Index를 찾아야 합니다.
    # 여기서는 간단히 두 분포가 겹치는 구간을 봅니다.
    
    if len(norm_arr) > 0 and len(anom_arr) > 0:
        # 모델이 편향되어서 Normal 평균이 음수(-0.5)라면, 기준점을 -0.5 근처로 옮겨야 합니다.
        optimal_threshold = (norm_arr.mean() + anom_arr.mean()) / 2
        print(f"⚖️ 추천 최적 Threshold: {optimal_threshold:.4f}")
        
        # 보정된 Threshold로 다시 계산
        acc_calib_norm = (norm_arr > optimal_threshold).sum() / len(norm_arr)
        acc_calib_anom = (anom_arr <= optimal_threshold).sum() / len(anom_arr)
        print(f"📈 보정 후 예상 Accuracy -> Normal: {acc_calib_norm*100:.1f}%, Anomaly: {acc_calib_anom*100:.1f}%")
        
        # F1 Score 재계산 (TP, FP, TN, FN)
        # Threshold보다 크면 Normal(0), 작으면 Anomaly(1)
        TP = (anom_arr <= optimal_threshold).sum()
        FN = (anom_arr > optimal_threshold).sum()
        TN = (norm_arr > optimal_threshold).sum()
        FP = (norm_arr <= optimal_threshold).sum()
        
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        print(f"🏆 Final Calibrated F1 Score: {f1:.4f}")
    else:
        print("데이터가 부족하여 Threshold를 계산할 수 없습니다.")

    print("\n" + "="*40)
    print(f"{'Category':<15} | {'AP':<10} | {'AUROC':<10}")
    print("-" * 40)
    
    ap_list = []
    auroc_list = []
    
    for cat, data in category_results.items():
        # 데이터 분리
        y_true = [x[0] for x in data]      # 정답 (0:정상, 1:불량)
        y_scores = [x[1] for x in data]    # 예측 점수
        
        # 데이터가 섞여 있어야(정상/불량 둘 다 존재) 계산 가능
        if len(set(y_true)) < 2:
            print(f"{cat:<15} | {'N/A':<10} | {'N/A':<10} (Only one class present)")
            continue
            
        # AP (Average Precision) 계산
        ap = average_precision_score(y_true, y_scores)
        
        # AUROC (Area Under ROC) 계산 - 덤으로 같이 봅니다
        auroc = roc_auc_score(y_true, y_scores)
        
        ap_list.append(ap)
        auroc_list.append(auroc)
        
        print(f"{cat:<15} | {ap:.4f}     | {auroc:.4f}")

    print("-" * 40)
    
    # mAP (AP들의 평균)
    if len(ap_list) > 0:
        mAP = sum(ap_list) / len(ap_list)
        mAUROC = sum(auroc_list) / len(auroc_list)
        print(f"🥇 mAP (Mean Average Precision): {mAP:.4f}")
        print(f"🥈 mAUROC (Mean AUROC)       : {mAUROC:.4f}")
        
        # 파일 저장
        with open(os.path.join(args.output_dir, "map_results.txt"), "w") as f:
            f.write(f"mAP: {mAP:.4f}\n")
            f.write(f"mAUROC: {mAUROC:.4f}\n")
    else:
        print("계산 가능한 카테고리가 없습니다.")

    print("\n======== [Finding Optimal F1 Score] ========")
    from sklearn.metrics import precision_recall_curve

    all_labels = []
    all_scores = []

    # 모든 카테고리의 데이터를 한곳에 모읍니다.
    for cat, data in category_results.items():
        for label, score in data:
            all_labels.append(label)
            all_scores.append(score)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Precision-Recall Curve를 그려서 최고의 F1 지점을 찾습니다.
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    
    # F1 계산 (0으로 나누기 방지)
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    # 최대 F1 Score와 그때의 Threshold 찾기
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx]

    print(f"💎 Best Possible F1 Score: {best_f1:.4f}")
    print(f"⚖️ Optimal Threshold: {best_threshold:.4f}")
    print("--------------------------------------------")
    print("이 값을 사용하여 실제 예측을 하면 성능이 크게 향상됩니다.")

    print("\n" + "="*50)
    print("💎 Calculating Best F1 Score per Category")
    print("="*50)
    
    f1_list = []
    
    print(f"{'Category':<15} | {'Best F1':<10} | {'Threshold':<10}")
    print("-" * 50)

    # 1. 카테고리별로 따로따로 최적의 F1을 구합니다.
    for cat, data in category_results.items():
        y_true = np.array([x[0] for x in data])
        y_scores = np.array([x[1] for x in data])

        # 정상 또는 불량 데이터만 있는 경우 계산 불가
        if len(set(y_true)) < 2:
            print(f"{cat:<15} | {'N/A':<10} | {'N/A':<10} (Skipped)")
            continue

        # Precision-Recall Curve 계산
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # F1 Score 계산
        numerator = 2 * precision * recall
        denominator = precision + recall
        f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        
        # 해당 카테고리에서의 최고 점수 찾기
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        f1_list.append(best_f1)
        
        print(f"{cat:<15} | {best_f1:.4f}     | {best_threshold:.4f}")

    print("-" * 50)

    # 2. 최종 점수: 카테고리별 F1의 평균 (Macro Average)
    if len(f1_list) > 0:
        macro_f1 = sum(f1_list) / len(f1_list)
        print(f"🏆 Class-Average Best F1 Score: {macro_f1:.4f}")
        
        with open(os.path.join(args.output_dir, "class_wise_f1.txt"), "w") as f:
            f.write(f"Class-Average Best F1: {macro_f1:.4f}\n")
    else:
        print("계산된 F1 점수가 없습니다.")

if __name__ == "__main__":
    main()