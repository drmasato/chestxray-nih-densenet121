import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

DATA_DIR  = '/media/morita/ubuntuHDD/chestxray'
IMG_DIR   = f'{DATA_DIR}/images'
MODEL_PATH = f'{DATA_DIR}/checkpoints/best_model.pth'
OUT_DIR   = f'{DATA_DIR}/results'
os.makedirs(OUT_DIR, exist_ok=True)

DISEASES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

DISEASES_JP = [
    '無気肺', '浸潤影', '浸潤', '気胸',
    '肺水腫', '肺気腫', '肺線維症', '胸水', '肺炎',
    '胸膜肥厚', '心肥大', '結節', '腫瘤', 'ヘルニア'
]

class NIHChestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['Image Index']}").convert('RGB')
        label = torch.FloatTensor(row[DISEASES].values.astype(float))
        if self.transform:
            img = self.transform(img)
        return img, label

class ChestXrayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(nn.Linear(1024, 14))

    def forward(self, x):
        return self.model(x)

def main():
    device = torch.device('cuda')

    # データ準備
    df = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')
    for d in DISEASES:
        df[d] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0)

    with open(f'{DATA_DIR}/test_list.txt') as f:
        test_files = set(f.read().splitlines())
    df_test = df[df['Image Index'].isin(test_files)].reset_index(drop=True)
    print(f"テストデータ: {len(df_test)}枚")

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_ds     = NIHChestDataset(df_test, IMG_DIR, tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=8, pin_memory=True)

    # モデル読み込み
    model = ChestXrayModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("モデル読み込み完了")

    # 推論
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="評価中"):
            imgs = imgs.to(device)
            with autocast('cuda'):
                outputs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_preds.append(outputs)
            all_labels.append(labels.numpy())

    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    # AUC計算
    print("\n===== テストセット評価結果 =====")
    aucs = []
    for i, (disease, jp) in enumerate(zip(DISEASES, DISEASES_JP)):
        if labels[:, i].sum() > 0:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            aucs.append(auc)
            print(f"  {disease:<22} ({jp:<8}): {auc:.4f}")
    mean_auc = np.mean(aucs)
    print(f"\n  {'Mean AUC':<22}           : {mean_auc:.4f}")
    print(f"  論文ベースライン（DenseNet-121）: 0.8414")

    # 1. AUCバーグラフ
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if a >= 0.85 else 'steelblue' if a >= 0.80 else 'orange' for a in aucs]
    bars = ax.barh(DISEASES_JP, aucs, color=colors)
    ax.axvline(x=mean_auc, color='red', linestyle='--', label=f'Mean AUC: {mean_auc:.4f}')
    ax.axvline(x=0.8414,   color='gray', linestyle=':', label='論文ベースライン: 0.8414')
    ax.set_xlabel('AUC-ROC')
    ax.set_title('疾患別 AUC-ROC（テストセット）')
    ax.set_xlim(0.5, 1.0)
    ax.legend()
    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.005, bar.get_y() + bar.get_height()/2,
                f'{auc:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/auc_barplot.png', dpi=150)
    print(f"\n保存: {OUT_DIR}/auc_barplot.png")

    # 2. ROC曲線
    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    axes = axes.flatten()
    for i, (disease, jp) in enumerate(zip(DISEASES, DISEASES_JP)):
        if labels[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
            auc = roc_auc_score(labels[:, i], preds[:, i])
            axes[i].plot(fpr, tpr, color='steelblue', lw=2)
            axes[i].plot([0,1],[0,1],'k--', lw=1)
            axes[i].set_title(f'{jp}\nAUC={auc:.3f}', fontsize=10)
            axes[i].set_xlabel('FPR', fontsize=8)
            axes[i].set_ylabel('TPR', fontsize=8)
    for j in range(len(DISEASES), len(axes)):
        axes[j].axis('off')
    plt.suptitle(f'ROC曲線（テストセット）  Mean AUC={mean_auc:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/roc_curves.png', dpi=150)
    print(f"保存: {OUT_DIR}/roc_curves.png")
    print("\n評価完了！")

if __name__ == '__main__':
    main()
