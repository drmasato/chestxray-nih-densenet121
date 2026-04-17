# 胸部X線画像診断AI（NIH ChestX-ray14）

DenseNet-121による胸部X線画像の多疾患分類モデル。NIH ChestX-ray14データセット（112,120枚）を用いて14種類の胸部疾患を検出する。

---

## 動作環境

| 項目 | バージョン |
|------|----------|
| OS | Ubuntu 22.04 / 24.04 |
| Python | 3.11以上 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GTX 1080 Ti（VRAM 11GB） |

---

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository_url>
cd chestxray
```

### 2. 仮想環境の作成・有効化

```bash
python3 -m venv chestxray_env
source chestxray_env/bin/activate
```

### 3. 依存ライブラリのインストール

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pandas scikit-learn albumentations matplotlib tqdm tensorboard grad-cam
```

### 4. データセットのダウンロード

```bash
pip install kaggle

# Kaggle APIキーを設定（~/.kaggle/kaggle.json）
kaggle datasets download -d nih-chest-xrays/data -p ./

# 解凍
unzip data.zip -d ./

# 全画像を1フォルダに統合
mkdir -p images
for i in $(seq -w 1 12); do
  mv images_0${i}/images/*.png images/
done
```

---

## ディレクトリ構成

```
chestxray/
├── images/                  # 112,120枚の画像（PNG）
├── Data_Entry_2017.csv      # ラベルデータ
├── train_val_list.txt       # 訓練・検証用ファイルリスト
├── test_list.txt            # テスト用ファイルリスト
├── BBox_List_2017.csv       # バウンディングボックス情報
├── train.py                 # 学習スクリプト
├── checkpoints/
│   └── best_model.pth       # ベストモデル
├── runs/                    # TensorBoardログ
├── 要件定義書.md
└── README.md
```

---

## 学習

```bash
source chestxray_env/bin/activate

# 学習開始
python3 train.py

# バックグラウンドで実行
nohup python3 train.py > train.log 2>&1 &

# 進捗確認
tail -f train.log
```

### 主要パラメータ（train.py）

| パラメータ | デフォルト値 | 説明 |
|----------|------------|------|
| BATCH_SIZE | 32 | バッチサイズ |
| EPOCHS | 30 | 学習エポック数 |
| LR | 1e-4 | 学習率 |
| NUM_WORKERS | 8 | データローダーのワーカー数 |

---

## 学習の可視化

```bash
source chestxray_env/bin/activate
tensorboard --logdir runs

# ブラウザで確認
# http://localhost:6006
```

---

## 対象疾患（14クラス）

| 疾患名 | 日本語 |
|--------|--------|
| Atelectasis | 無気肺 |
| Consolidation | 浸潤影 |
| Infiltration | 浸潤 |
| Pneumothorax | 気胸 |
| Edema | 肺水腫 |
| Emphysema | 肺気腫 |
| Fibrosis | 肺線維症 |
| Effusion | 胸水 |
| Pneumonia | 肺炎 |
| Pleural_Thickening | 胸膜肥厚 |
| Cardiomegaly | 心肥大 |
| Nodule | 結節 |
| Mass | 腫瘤 |
| Hernia | ヘルニア |

---

## 評価指標

AUC-ROC を疾患ごとに計算し、Mean AUCで総合評価する。

| モデル | Mean AUC |
|--------|---------|
| 論文ベースライン（DenseNet-121） | 0.841 |
| 本プロジェクト目標 | 0.800以上 |

---

## GPU使用状況の確認

```bash
# リアルタイム監視
watch -n 2 nvidia-smi

# 学習中のGPU使用率
nvidia-smi | grep GPU-Util
```

---

## 参考文献

- Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks", CVPR 2017
- Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning", 2017
- [NIH Clinical Center ChestX-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Kaggle Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
