# 🧠 PD Multimodal AI

Parkinson's Disease handwriting detection using **multimodal learning** — integrating handwriting, voice, and gait features to assist in early diagnosis and digital biomarkers research.

---

## 📂 Project Structure
pd_multimodal_ai/
├─ data/
│  ├─ raw/              # 原始手寫資料
│  ├─ processed/        # 預處理後的資料
│  └─ labels.csv        # 標註檔（如 subject_id, class, path）
│
├─ notebooks/           # Jupyter 實驗筆記
│  ├─ EDA_handwriting.ipynb     # 資料探索與視覺化
│  ├─ model_training.ipynb      # 模型訓練流程
│  └─ evaluation.ipynb          # 評估與可視化
│
├─ src/                 # 程式模組
│  ├─ preprocess.py     # 資料前處理
│  ├─ dataset.py        # Dataset & DataLoader
│  ├─ model.py          # CNN / ViT / Multimodal 模型架構
│  ├─ train.py          # 訓練邏輯
│  ├─ eval.py           # 評估邏輯
│  └─ utils.py          # 工具函式與視覺化
│
├─ outputs/             # 模型與結果輸出
│  ├─ checkpoints/      # 模型權重
│  ├─ logs/             # 訓練日誌
│  └─ predictions/      # 推論結果
│
├─ requirements.txt
├─ .gitignore
└─ README.md
