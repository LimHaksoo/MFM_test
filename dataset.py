import io
import sys
import tarfile
import zipfile
import shutil
import urllib.request
from urllib.parse import urlparse
import subprocess
import argparse
import csv

from tqdm import tqdm

import os
import glob
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
NORMAL_KEYWORDS = ("good", "ok", "normal")

def _is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMG_EXTS

def _default_clip_preprocess(image_size: int = 224) -> transforms.Compose:
    
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

def _guess_label_from_path(path: str) -> int:
    low = path.lower()
    return 0 if any(k in low.split(os.sep) for k in NORMAL_KEYWORDS) else 1

def _guess_mask_path_mvtec(img_path: str) -> Optional[str]:
   
    parts = img_path.replace("\\", "/").split("/")
    if "test" not in parts:
        return None
    try:
        idx = parts.index("test")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    defect = parts[idx + 1]
    fname = os.path.splitext(parts[-1])[0]
    # build ground truth path
    parts[idx] = "ground_truth"
    parts[idx + 1] = defect
    gt_dir = "/".join(parts[:-1])
    candidates = [
        os.path.join(gt_dir, f"{fname}_mask.png"),
        os.path.join(gt_dir, f"{fname}_mask.jpg"),
        os.path.join(gt_dir, f"{fname}.png"),
        os.path.join(gt_dir, f"{fname}.jpg"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def _guess_mask_path_generic(img_path: str) -> Optional[str]:
   
    d, fn = os.path.split(img_path)
    parent = os.path.dirname(d)
    fname, _ = os.path.splitext(fn)
    for mask_dir_name in ["mask", "masks", "ground_truth", "gt", "annotations"]:
        mdir = os.path.join(parent, mask_dir_name)
        if not os.path.isdir(mdir):
            continue
        cands = glob.glob(os.path.join(mdir, fname + ".*"))
        cands = [c for c in cands if _is_image(c)]
        if cands:
            return cands[0]
    return None

@dataclass
class Sample:
    img_path: str
    mask_path: Optional[str]
    label: int  # 0 normal, 1 anomaly
    dataset: str
    category: Optional[str]
    split: str  # 'train' | 'test' | 'all'

class AnomalyDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str = "train",
        image_size: int = 224,
        return_mask: bool = False,
        use_clip_preprocess: bool = True,
        normal_only_train: bool = False,
        visa_split: Optional[str] = "2cls_highshot",
    ):
    
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.image_size = image_size
        self.return_mask = return_mask
        self.normal_only_train = normal_only_train
        self.visa_split = visa_split

        self.transform = _default_clip_preprocess(image_size) if use_clip_preprocess \
            else transforms.Compose([transforms.Resize(image_size),
                                     transforms.CenterCrop(image_size),
                                     transforms.ToTensor()])

        self.samples: List[Sample] = self._scan()

    def _scan(self) -> List[Sample]:
        root = self.root
        ds = self.dataset_name
        split = self.split
        samples: List[Sample] = []

        if ds == "visa" and getattr(self, "visa_split", None):
            return self._scan_visa_from_split_csv()

        def add_sample(img_path: str, dataset: str, category: Optional[str], split_: str):
            label = _guess_label_from_path(img_path)
      
            if split_ == "train" and self.normal_only_train and label != 0:
                return
          
            mask_path = None
            if ds in ("mvtec", "mvtec_loco"):
                mask_path = _guess_mask_path_mvtec(img_path) or _guess_mask_path_generic(img_path)
            else:
                mask_path = _guess_mask_path_generic(img_path)

            samples.append(Sample(
                img_path=img_path,
                mask_path=mask_path,
                label=label,
                dataset=dataset,
                category=category,
                split=split_,
            ))

        
        if split in ("train", "test"):
            split_dirs = [split]
        else:
            split_dirs = ["train", "test"]

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")

        
        categories = [d for d in sorted(os.listdir(root))
                    if os.path.isdir(os.path.join(root, d))]

        for cat in categories:
            cat_dir = os.path.join(root, cat)
            for sp in split_dirs:
                # ---------- VisA 전용 처리 ----------
                if ds == "visa":
                    images_dir = os.path.join(cat_dir, "Data", "Images")
                    if not os.path.isdir(images_dir):
                        # 구조가 예상과 다르면 그냥 generic 로직으로 fallback
                        base_dirs = [cat_dir]
                        for bdir in base_dirs:
                            if not os.path.isdir(bdir):
                                continue
                            for p in glob.glob(os.path.join(bdir, "**", "*"), recursive=True):
                                if not os.path.isfile(p) or not _is_image(p):
                                    continue
                                low = p.replace("\\", "/").lower()
                                if any(seg in low for seg in ["/ground_truth/", "/mask/", "/masks/", "/annotations/"]):
                                    continue
                                add_sample(p, ds, cat, sp)
                        continue  # VisA 처리 끝

                    normal_dir = os.path.join(images_dir, "Normal")
                    anomaly_dir = os.path.join(images_dir, "Anomaly")

                    normal_imgs: List[str] = []
                    anomaly_imgs: List[str] = []

                    if os.path.isdir(normal_dir):
                        normal_imgs = [
                            p for p in glob.glob(os.path.join(normal_dir, "**", "*"), recursive=True)
                            if os.path.isfile(p) and _is_image(p)
                        ]
                    # import pdb; pdb.set_trace()
                    if os.path.isdir(anomaly_dir):
                        anomaly_imgs = [
                            p for p in glob.glob(os.path.join(anomaly_dir, "**", "*"), recursive=True)
                            if os.path.isfile(p) and _is_image(p)
                        ]

                    # VisA: 공식 train/test split이 없으니 여기서 간단히 나눔 (예: 80/20)
                    split_ratio = 0.8
                    n_norm = len(normal_imgs)
                    n_anom = len(anomaly_imgs)
                    n_norm_train = int(n_norm * split_ratio)
                    n_anom_train = int(n_anom * split_ratio)

                    if sp == "train":
                        cur_normals = normal_imgs[:n_norm_train] if n_norm_train > 0 else normal_imgs
                        cur_anoms = anomaly_imgs[:n_anom_train] if n_anom_train > 0 else anomaly_imgs

                        # train normal 은 항상 포함
                        for p in cur_normals:
                            add_sample(p, ds, cat, "train")

                        # train anomaly 는 normal_only_train 플래그로 제어
                        if not self.normal_only_train:
                            for p in cur_anoms:
                                add_sample(p, ds, cat, "train")

                    elif sp == "test":
                        cur_normals = normal_imgs[n_norm_train:] if n_norm_train > 0 else []
                        cur_anoms = anomaly_imgs[n_anom_train:] if n_anom_train > 0 else []

                        for p in cur_normals + cur_anoms:
                            add_sample(p, ds, cat, "test")

                    else:  # self.split == "all" 등
                        for p in normal_imgs + anomaly_imgs:
                            add_sample(p, ds, cat, "all")

                    # VisA는 여기서 끝, 다음 (sp, cat)으로
                    continue

                # ---------- 그 외 데이터셋 (mvtec, mvtec_loco, goodsad 등) ----------
                sp_dir = os.path.join(cat_dir, sp)
                base_dirs = [sp_dir] if os.path.isdir(sp_dir) else [cat_dir]

                for bdir in base_dirs:
                    if not os.path.isdir(bdir):
                        continue
                    for p in glob.glob(os.path.join(bdir, "**", "*"), recursive=True):
                        if not os.path.isfile(p):
                            continue
                        if not _is_image(p):
                            continue
                        low = p.replace("\\", "/").lower()
                        if any(seg in low for seg in ["/ground_truth/", "/mask/", "/masks/", "/annotations/"]):
                            continue
                        add_sample(p, ds, cat, sp)

        if not samples:
            
            for ext in IMG_EXTS:
                for p in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True):
                    if "/ground_truth/" in p.replace("\\", "/"):
                        continue
                    add_sample(p, ds, None, split if split in ("train", "test") else "all")

        return samples

    def _scan_visa_from_split_csv(self) -> List[Sample]:
        """
        VisA 전용:
        split_csv/{1cls,2cls_fewshot,2cls_highshot}.csv (또는 직접 지정한 csv 경로)를 이용해
        train/test/label/mask를 전부 정의한다.
        """
        if not self.visa_split:
            return []

        # visa_split 값이 '2cls_highshot' 같은 이름인지, 아니면 직접 csv 경로인지 둘 다 허용
        # base = os.path.join(self.root, "VisA")
        base = self.root # JISU MODIFIED
        if os.path.isfile(self.visa_split):
            csv_path = self.visa_split
        else:
            csv_name = self.visa_split
            if not csv_name.endswith(".csv"):
                csv_name = f"{csv_name}.csv"
            csv_path = os.path.join(base, "split_csv", csv_name)

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"VisA split_csv not found: {csv_path}")

        ds = self.dataset_name
        split = self.split
        samples: List[Sample] = []

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # 헤더일 가능성
                if row[0].lower() in ("category", "cls", "class"):
                    continue
                if len(row) < 4:
                    continue

                category = row[0].strip()
                split_csv = row[1].strip()      # 'train' or 'test'
                label_str = row[2].strip()      # 'normal' or 'anomaly'
                img_rel = row[3].strip()
                mask_rel = row[4].strip() if len(row) > 4 else ""

                # DataLoader에서 요청한 split만 선택
                if split in ("train", "test"):
                    if split_csv != split:
                        continue
                elif split != "all":
                    # 'all' 이 아니고 다른 이름이면, 같은 이름만 사용
                    if split_csv != split:
                        continue

                img_path = os.path.join(base, img_rel)
                mask_path = os.path.join(base, mask_rel) if mask_rel else None

                # 혹시 csv에는 있는데 실제 파일이 없으면 스킵
                if not os.path.isfile(img_path):
                    continue
                if mask_path is not None and not os.path.isfile(mask_path):
                    mask_path = None

                # label 매핑
                if label_str.lower() in ("good", "ok", "normal", "0"):
                    label = 0
                else:
                    label = 1

                # 1-class / unsupervised 설정: train에서 anomaly 빼고 싶으면 여기에 걸림
                if split_csv == "train" and self.normal_only_train and label != 0:
                    continue

                samples.append(Sample(
                    img_path=img_path,
                    mask_path=mask_path,
                    label=label,
                    dataset=ds,
                    category=category,
                    split=split_csv,
                ))

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _load_mask(self, path: Optional[str], size: Tuple[int, int]) -> torch.Tensor:
        if not self.return_mask:
            return torch.tensor(0)  # placeholder
        if path is None or not os.path.exists(path):
            return torch.zeros((1, size[1], size[0]), dtype=torch.float32)
        m = Image.open(path).convert("L").resize(size, Image.NEAREST)
        t = transforms.ToTensor()(m)  # (1,H,W), [0,1]
        
        t = (t > 0.5).float()
        return t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img = self._load_image(s.img_path)
        w, h = img.size
        img_t = self.transform(img)
        mask_t = self._load_mask(s.mask_path, (self.image_size, self.image_size)) if self.return_mask else torch.tensor(0)
        return {
            "image": img_t,                  # (3,H,W)
            "label": torch.tensor(s.label),  # 0 or 1
            "mask": mask_t,                  # (1,H,W) or scalar 0
            "meta": {
                "img_path": s.img_path,
                "mask_path": s.mask_path,
                "dataset": s.dataset,
                "category": s.category,
                "split": s.split,
                "orig_size": (h, w),
            }
        }

def make_dataloader(
    root: str,
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    return_mask: bool = False,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    ds = AnomalyDataset(
        root=root,
        dataset_name=dataset_name,
        split=split,
        image_size=image_size,
        return_mask=return_mask,
        normal_only_train=False,
    )
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      collate_fn=_collate_with_meta,)   

def make_concat_dataloader(
    roots: Dict[str, str],
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    return_mask: bool = False,
) -> DataLoader:
    """
    Args:
        roots: {"mvtec": "/path/to/mvtec", "visa": "...", "mvtec_loco": "...", "goodsad": "..."}
    """
    datasets = []
    for name, path in roots.items():
        datasets.append(AnomalyDataset(
            root=path, dataset_name=name, split=split,
            image_size=image_size, return_mask=return_mask))
    concat = ConcatDataset(datasets)
    return DataLoader(concat, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, pin_memory=True,
                      collate_fn=_collate_with_meta,)   

# =========================
# AUTO-DOWNLOAD INTEGRATION
# =========================
AUTO_DATASET_SOURCES = {
    # MVTec AD
    "mvtec": {
        "type": "direct",
        "url": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        "needs_env_consent": "AGREE_MVTEC_LICENSE"
    },
    # MVTec LOCO AD
    "mvtec_loco": {
        "type": "direct",
        "url": "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz",
        "needs_env_consent": "AGREE_MVTEC_LICENSE"
    },
    # VisA — Amazon Research S3
    "visa": {
        "type": "direct",
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "needs_env_consent": None
    },
    # GoodsAD — Kaggle 
    "goodsad": {
        "type": "kaggle",
        "slug": "dtcrxs/goodsad",
        "needs_env_consent": None
    },
}

class _TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def _human_size(n):
    for u in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}EB"

def _download_file(url: str, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with _TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=f"Downloading {os.path.basename(out_path)}") as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)
    print(f"[download] saved: {out_path} ({_human_size(os.path.getsize(out_path))})")

def _extract_any(archive_path: str, dest_dir: str):
    print(f"[extract] {archive_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    low = archive_path.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        return
    
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
        return
    except tarfile.ReadError as e:
 
        print(f"[warn] Python tarfile failed ({e}). Trying system 'tar'...")
        cmd = ["tar", "-xf", archive_path, "-C", dest_dir]
        subprocess.check_call(cmd)

def _maybe_flatten_topdir(dst_dir: str):
    entries = [e for e in os.listdir(dst_dir) if not e.startswith(".")]
    if len(entries) == 1:
        only = os.path.join(dst_dir, entries[0])
        if os.path.isdir(only):
            tmp = dst_dir + "_tmp"
            os.makedirs(tmp, exist_ok=True)
            for it in os.listdir(only):
                shutil.move(os.path.join(only, it), os.path.join(tmp, it))
            shutil.rmtree(dst_dir)
            shutil.move(tmp, dst_dir)

def _has_enough_images(path: str, min_count: int = 50) -> bool:
    if not os.path.isdir(path):
        return False
    c = 0
    for ext in IMG_EXTS:
        c += len(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True))
        if c >= min_count:
            return True
    return c >= min_count

def _download_mvtec_like(name: str, dest_root: str):
    src = AUTO_DATASET_SOURCES[name]
    env_flag = src.get("needs_env_consent")
    if env_flag:
        if os.environ.get(env_flag, "").lower() not in ("1", "true", "yes", "y"):
            raise RuntimeError(
                f"[{name}] 라이선스 동의가 필요합니다. 환경변수 {env_flag}=1 을 설정한 뒤 다시 실행하세요.\n"
                f"공식 다운로드 페이지: https://www.mvtec.com/company/research/datasets/{'mvtec-ad' if name=='mvtec' else 'mvtec-loco'}/downloads"
            )
    url = src["url"]
    archive_path = os.path.join(os.path.dirname(os.path.abspath(dest_root)), f"{name}.tar.xz")
    _download_file(url, archive_path)
    _extract_any(archive_path, dest_root)
    _maybe_flatten_topdir(dest_root)
    try: os.remove(archive_path)
    except Exception: pass

def _download_visa(dest_root: str):
    url = AUTO_DATASET_SOURCES["visa"]["url"]
    archive_path = os.path.join(os.path.dirname(os.path.abspath(dest_root)), "VisA_20220922.tar")
    # _download_file(url, archive_path)
    _extract_any(archive_path, dest_root)
    _maybe_flatten_topdir(dest_root)
    # try: os.remove(archive_path)
    # except Exception: pass

def _download_goodsad_via_kaggle(dest_root: str):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        raise RuntimeError(
            "[goodsad] Kaggle API가 필요합니다. `pip install kaggle` 후 "
            "~/.kaggle/kaggle.json 또는 KAGGLE_USERNAME/KAGGLE_KEY 환경변수를 설정하세요."
        )
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise RuntimeError(
            "[goodsad] Kaggle 인증 실패. ~/.kaggle/kaggle.json 혹은 환경변수를 점검하세요."
        ) from e
    slug = AUTO_DATASET_SOURCES["goodsad"]["slug"]
    dl_dir = os.path.dirname(os.path.abspath(dest_root))
    print(f"[goodsad] downloading Kaggle dataset: {slug}")
    api.dataset_download_files(slug, path=dl_dir, unzip=True, quiet=False)
   
    cand = [p for p in glob.glob(os.path.join(dl_dir, "*")) if os.path.isdir(p) and "goods" in os.path.basename(p).lower()]
    if cand:
        src_dir = cand[0]
        os.makedirs(dest_root, exist_ok=True)
        for it in os.listdir(src_dir):
            shutil.move(os.path.join(src_dir, it), os.path.join(dest_root, it))
        shutil.rmtree(src_dir)
    _maybe_flatten_topdir(dest_root)

def ensure_dataset_ready(dataset_name: str, dest_root: str) -> str:
   
    ds = dataset_name.lower()
    if _has_enough_images(dest_root, min_count=10):
        print(f"[auto] '{ds}' already ready at: {dest_root} (skip download)")
        return dest_root

    print(f"[auto] '{ds}' not found at {dest_root}. Start downloading...")
    if ds == "visa":
        _download_visa(dest_root)
    elif ds in ("mvtec", "mvtec_loco"):
        _download_mvtec_like(ds, dest_root)
    elif ds == "goodsad":
        _download_goodsad_via_kaggle(dest_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not _has_enough_images(dest_root, min_count=10):
        raise RuntimeError(f"[auto] '{ds}' 다운로드/해제 후에도 이미지가 충분히 보이지 않습니다: {dest_root}")
    print(f"[auto] '{ds}' is ready at: {dest_root}")
    return dest_root


_ORIG_make_dataloader = make_dataloader
def make_dataloader(
    root: str,
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    return_mask: bool = False,
    shuffle: Optional[bool] = None,
):
   
    try:
        ensure_dataset_ready(dataset_name, root)
    except Exception as e:
        print(f"[warn] auto-download failed for {dataset_name}: {e}\n"
              f"→ 수동으로 내려받아 {root}에 두고 다시 시도하세요.")
    return _ORIG_make_dataloader(
        root=root, dataset_name=dataset_name, split=split, batch_size=batch_size,
        num_workers=num_workers, image_size=image_size, return_mask=return_mask, shuffle=shuffle
    )

_ORIG_make_concat_dataloader = make_concat_dataloader
def make_concat_dataloader(
    roots: Dict[str, str],
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    return_mask: bool = False,
):
   
    ready = {}
    for name, path in roots.items():
        try:
            ready[name] = ensure_dataset_ready(name, path)
        except Exception as e:
            print(f"[warn] auto-download failed for {name}: {e}")
            ready[name] = path  
    return _ORIG_make_concat_dataloader(
        roots=ready, split=split, batch_size=batch_size, num_workers=num_workers,
        image_size=image_size, return_mask=return_mask
    )

# --- CLI: python data/dataset.py auto --name mvtec --dest /data/MVTecAD ---
def _build_auto_parser(subparsers):
    sp = subparsers.add_parser("auto", help="Ensure dataset exists; download if missing")
    sp.add_argument("--name", required=True, help="mvtec | visa | mvtec_loco | goodsad")
    sp.add_argument("--dest", required=True, help="Destination root (unzipped)")
    sp.set_defaults(func=_cli_auto)

def _cli_auto(args):
    ensure_dataset_ready(args.name, args.dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset utilities")
    subs = parser.add_subparsers()
    _build_auto_parser(subs)
    if len(sys.argv) == 1:
        parser.print_help(); sys.exit(0)
    args = parser.parse_args(); args.func(args)
# ==== END AUTO-DOWNLOAD ===============================================


def _collate_with_meta(batch):
    
    # images
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)

  
    m0 = batch[0]["mask"]
    if isinstance(m0, torch.Tensor) and m0.dim() == 3:
        masks = torch.stack([b["mask"] for b in batch], dim=0)  # (B,1,H,W)
    else:
        masks = torch.tensor([0] * len(batch), dtype=torch.long)  # placeholder (B,)


    metas = [b["meta"] for b in batch]

    return {"image": images, "label": labels, "mask": masks, "meta": metas}