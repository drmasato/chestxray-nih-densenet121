"""
精度ベンチマーク記録スクリプト
各モデルのテスト結果を benchmark_results.csv に追記する
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import torchvision.transforms as transforms
from torchvision import models

DATA_DIR = '/media/morita/ubuntuHDD/chestxray'
IMG_DIR  = f'{DATA_DIR}/images'
RESULTS_CSV = f'{DATA_DIR}/benchmark_results.csv'

DISEASES = [
    'Atelectasis','Consolidation','Infiltration','Pneumothorax',
    'Edema','Emphysema','Fibrosis','Effusion','Pneumonia',
    'Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia'
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

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            with autocast('cuda'):
                out = torch.sigmoid(model(imgs)).cpu().numpy()
            all_preds.append(out)
            all_labels.append(labels.numpy())

    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    # NaN チェック（勾配爆発時のフェイルセーフ）
    if np.isnan(preds).any():
        print("⚠️ 予測値にNaNが含まれています。学習が不安定です。")
        return {'Mean': 0.0}

    aucs = {}
    for i, d in enumerate(DISEASES):
        if labels[:, i].sum() > 0:
            aucs[d] = roc_auc_score(labels[:, i], preds[:, i])
    aucs['Mean'] = np.mean(list(aucs.values()))
    return aucs

def record_result(model_name, model, loader, device, note=""):
    print(f"\n===== {model_name} 評価中 =====")
    aucs = evaluate_model(model, loader, device)

    # 表示
    for d, auc in aucs.items():
        print(f"  {d:<22}: {auc:.4f}")

    # CSV に追記
    row = {
        'datetime':   datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model':      model_name,
        'mean_auc':   aucs['Mean'],
        'note':       note,
    }
    row.update({d: aucs.get(d, None) for d in DISEASES})

    df_new = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        df_old = pd.read_csv(RESULTS_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(RESULTS_CSV, index=False)
    print(f"\n✓ 記録済み: {RESULTS_CSV}")
    print(f"  Mean AUC: {aucs['Mean']:.4f}")
    return aucs

def get_test_loader(img_size=224, batch_size=64):
    df = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')
    for d in DISEASES:
        df[d] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0)
    with open(f'{DATA_DIR}/test_list.txt') as f:
        test_files = set(f.read().splitlines())
    df_test = df[df['Image Index'].isin(test_files)].reset_index(drop=True)

    tf = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = NIHChestDataset(df_test, IMG_DIR, tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=8, pin_memory=True)

def print_leaderboard():
    if not os.path.exists(RESULTS_CSV):
        print("まだ記録がありません")
        return
    df = pd.read_csv(RESULTS_CSV)
    df_sorted = df[['datetime','model','mean_auc','note']].sort_values(
        'mean_auc', ascending=False
    )
    print("\n===== ベンチマーク結果 =====")
    print(df_sorted.to_string(index=False))

if __name__ == '__main__':
    print_leaderboard()
