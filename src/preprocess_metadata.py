# -*- coding: utf-8 -*-
"""
把臨床 Excel/CSV 轉成訓練用 labels.csv
用法：
  python src/preprocess_metadata.py --meta data/metadata.xlsx --img-root data/raw --out data/labels.csv
若你的檔是 CSV：
  python src/preprocess_metadata.py --meta data/metadata.csv --img-root data/raw --out data/labels.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import glob

# === 依你的實際欄位名稱改這裡（中文也可） ===
COLUMN_MAP = {
    "id": "流水號",          # 受試者或測驗唯一編號
    "label": "診別",         # PD / Control（或其他）
    "task": "任務",          # 例如：螺旋 / 3 / three / spiral / 其他
    "stage": "PD 階段",      # Hoehn & Yahr 等級（可選）
    "date": "測驗日",        # 可選
    "mmse": "MMSE",          # 可選
    "moca": "MoCa",          # 可選
    "updrs3": "UPDRS III",   # 可選
}

# 任務名稱到資料夾名稱的對應與同義詞
TASK_DIR_MAP = {
    "spiral": ["spiral", "螺旋"],
    "three":  ["three", "3", "數字3", "three-task"],
    "other":  ["other", "wave", "文字", "波形", "text"],
}

IMG_EXTS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]


def normalize_task(task_value: str) -> str:
    if pd.isna(task_value):
        return "other"
    t = str(task_value).strip().lower()
    for key, synonyms in TASK_DIR_MAP.items():
        for s in synonyms:
            if str(s).lower() in t:
                return key
    return "other"


def normalize_label(lbl: str) -> str:
    if pd.isna(lbl):
        return "Unknown"
    s = str(lbl).strip().lower()
    if s in ["pd", "parkinson", "parkinson's disease", "帕金森氏症", "帕金森"]:
        return "PD"
    if s in ["control", "hc", "healthy", "對照組", "健康"]:
        return "Control"
    return lbl


def find_image_path(img_root: Path, task_dir: str, subject_id: str):
    """
    嘗試在 data/raw/<task_dir>/ 底下，用 subject_id 當前綴去找影像檔。
    你也可以改成用「檔名」欄位精準對應（若你的表有影像檔名）。
    """
    for ext in IMG_EXTS:
        pattern1 = str(img_root / task_dir / f"{subject_id}*.{ext}")
        pattern2 = str(img_root / task_dir / f"*{subject_id}*.{ext}")
        matches = glob.glob(pattern1) or glob.glob(pattern2)
        if matches:
            return Path(matches[0]).as_posix()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="臨床資料 Excel/CSV 路徑")
    ap.add_argument("--img-root", default="data/raw", help="原始影像根目錄")
    ap.add_argument("--out", default="data/labels.csv", help="輸出的 labels.csv")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    img_root = Path(args.img_root)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 讀 Excel 或 CSV
    if meta_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(meta_path)
    else:
        df = pd.read_csv(meta_path)

    # 檢查必要欄位是否存在
    for need in ["id", "label", "task"]:
        col = COLUMN_MAP.get(need)
        if col not in df.columns:
            raise ValueError(f"找不到必要欄位：{col}，請確認 COLUMN_MAP 與你的表頭一致。")

    rows = []
    miss_cnt = 0
    for _, r in df.iterrows():
        sid = str(r[COLUMN_MAP["id"]]).strip()
        label = normalize_label(r[COLUMN_MAP["label"]])
        task = normalize_task(r[COLUMN_MAP["task"]])

        # 對應影像檔路徑（若你有「影像檔名」欄位，可直接用它組路徑會更穩）
        img_path = find_image_path(img_root, task, sid)

        if not img_path:
            miss_cnt += 1
            # 仍然寫出一列，方便之後檢查補檔
        row = {
            "subject_id": sid,
            "image_path": img_path,
            "label": label,
            "task_type": task,
        }
        # 可選欄位：若有就帶出
        if COLUMN_MAP.get("stage") in df.columns:
            row["pd_stage"] = r[COLUMN_MAP["stage"]]
        if COLUMN_MAP.get("date") in df.columns:
            row["date"] = r[COLUMN_MAP["date"]]
        if COLUMN_MAP.get("mmse") in df.columns:
            row["mmse"] = r[COLUMN_MAP["mmse"]]
        if COLUMN_MAP.get("moca") in df.columns:
            row["moca"] = r[COLUMN_MAP["moca"]]
        if COLUMN_MAP.get("updrs3") in df.columns:
            row["updrs_iii"] = r[COLUMN_MAP["updrs3"]]

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"✅ 產生完成：{out_csv.as_posix()}  （共 {len(out_df)} 筆）")
    if miss_cnt:
        print(f"⚠️ 未找到影像路徑的筆數：{miss_cnt}（請檢查 subject_id 與檔名對應或調整 find_image_path 邏輯）")


if __name__ == "__main__":
    main()
