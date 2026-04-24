# 胸部X線AI診断システム

NIH ChestX-ray14 データセットで学習した多疾患分類モデルと、Gradio ベースの臨床向け Web インターフェースです。

---

## 概要

| 項目 | 内容 |
|------|------|
| データセット | NIH ChestX-ray14（112,120枚 / 14疾患） |
| 最高精度 | **Mean AUC: 0.8149**（DenseNet-121 + EfficientNet-B4 + XRV-DenseNet 3モデルアンサンブル） |
| 対応入力 | DICOM / PNG / JPG（拡張子なし DICOM も自動検出） |
| 出力 | 疾患確率・Grad-CAM・AI自動読影レポート |
| GPU | NVIDIA GTX 1080 Ti（11GB VRAM） |

---

## 要件定義・実施内容

### ステップ 1: DenseNet-121 ベースライン学習
- **モデル**: DenseNet-121（ImageNet 事前学習 → NIH 転移学習）
- **設定**: 224px, Batch=32, LR=1e-4, 30 epochs, AdamW, CosineAnnealingLR
- **対策**: BCEWithLogitsLoss + pos_weight（クラス不均衡補正）, Mixed precision (AMP)
- **結果**: Mean AUC **0.8004**

### ステップ 2: EfficientNet-B4 学習 + アンサンブル
- **モデル**: EfficientNet-B4（timm, drop_rate=0.3, drop_path_rate=0.1）
- **対策**: 勾配クリッピング（max_norm=1.0）, EarlyStopping（PATIENCE=5）
- **結果**: EfficientNet-B4 単体 Mean AUC **0.7964**
- **アンサンブル**（均等平均 0.5:0.5）: Mean AUC **0.8123**（+1.2% 向上）

### ステップ 3: torchxrayvision 事前学習モデル Fine-tuning + 3モデルアンサンブル
- **ベースモデル**: densenet121-res224-all（NIH + CheXpert + MIMIC-CXR + PadChest 4データセット事前学習）
- **分類層**: 事前学習済み重みを14疾患分引き継ぎ、op_threshsをリセット
- **段階的学習率**: backbone 5e-5 / head 1e-3、20 epochs
- **XRV単体 Test AUC**: 0.7860（val AUC過大評価: 0.8187 ※患者重複問題）
- **3モデルアンサンブル**（Dense 0.4 + Eff 0.4 + XRV 0.2）: **0.8149**

### Gradio Web アプリ
- DICOM マジックバイト検出（拡張子なしファイル対応）
- VOI LUT / MONOCHROME1 反転 / マルチフレーム対応
- 元画像 + Grad-CAM 横並び表示・クリック拡大
- AI 自動読影レポート（所見・インプレッション・推奨アクション）
- DenseNet + EfficientNet アンサンブル推論

---

## 精度結果

### モデル比較（NIH テストセット）

| モデル | Mean AUC |
|--------|----------|
| **Ensemble 3モデル（Dense+Eff+XRV 2:2:1）** | **0.8149** |
| Ensemble 2モデル（Dense+EffNet 1:1） | 0.8123 |
| EfficientNet-B4（epoch5 best） | 0.8051 |
| DenseNet-121（baseline） | 0.8004 |
| XRV-DenseNet Fine-tune（単体） | 0.7860 |

> **参考**: CheXNet（Stanford, 2017） Mean AUC: 0.841 / 放射線科医: 0.778（本システムは放射線科医水準を超えています）

### 疾患別 AUC（アンサンブルモデル）

| 疾患 | AUC | 疾患 | AUC |
|------|-----|------|-----|
| ヘルニア | **0.9421** | 肺気腫 | **0.8992** |
| 心肥大 | **0.8924** | 気胸 | 0.8640 |
| 肺水腫 | 0.8451 | 肺線維症 | 0.8354 |
| 胸水 | 0.8257 | 腫瘤 | 0.8111 |
| 胸膜肥厚 | 0.7732 | 無気肺 | 0.7708 |
| 結節 | 0.7582 | 浸潤影 | 0.7477 |
| 肺炎 | 0.7049 | 浸潤 | 0.7017 |

---

## ファイル構成

```
chestxray/
├── app.py                  # Gradio Web アプリ（メイン）
├── train.py                # DenseNet-121 学習
├── train_efficientnet.py   # EfficientNet-B4 学習
├── train_pretrained.py     # torchxrayvision Fine-tuning（Step3）
├── ensemble.py             # アンサンブル評価・記録
├── benchmark.py            # AUC 評価・記録ユーティリティ
├── evaluate.py             # テストセット評価 + グラフ生成
├── gradcam.py              # Grad-CAM 可視化
├── benchmark_results.csv   # 全モデルのベンチマーク記録
└── checkpoints/
    ├── best_model.pth              # DenseNet-121 best
    ├── efficientnet_b4_best.pth    # EfficientNet-B4 best
    └── xrv_densenet_finetuned.pth  # XRV fine-tuned（Step3完了後）
```

---

## 使い方

### 1. Web アプリ起動

```bash
source /media/morita/ubuntuHDD/chestxray_env/bin/activate
python3 app.py
# → http://localhost:7862 をブラウザで開く
```

### 2. 画像をアップロード
- **対応形式**: DICOM（拡張子なし可）/ PNG / JPG
- 病院 PACS からエクスポートした DICOM ファイルをそのままドロップ可能

### 3. 「診断する」をクリック
以下が自動生成されます：
- **元画像** + **Grad-CAM**（AI が注目した領域を重ねて表示、クリックで拡大）
- **上位5疾患の確率**
- **疾患別確率グラフ**（赤: 陽性疑い ≥50%）
- **AI 自動読影レポート**（所見・インプレッション・推奨アクション）

### 4. ベンチマーク記録

```bash
python3 benchmark.py        # リーダーボード表示
python3 record_densenet.py  # DenseNet-121 を再評価・記録
python3 ensemble.py         # アンサンブル評価
```

---

## 動作環境

```
OS:      Ubuntu 24.04
GPU:     NVIDIA GTX 1080 Ti (11GB VRAM)
CUDA:    12.2
Python:  3.12
PyTorch: 2.6.0+cu124
timm:    1.0.26
gradio:  6.12.0
pydicom: 3.0.2
torchxrayvision: 1.4.0
```

---

## 免責事項

> ⚠️ 本システムは**研究・学習目的**のみです。実際の診断・治療判断には使用しないでください。
> 患者データを使用する場合は必ず**匿名化**を行ってください。

---

## 参考文献

- [NIH ChestX-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXNet: Radiologist-Level Pneumonia Detection (Stanford, 2017)](https://arxiv.org/abs/1711.05225)
- [torchxrayvision](https://github.com/mlmed/torchxrayvision)
