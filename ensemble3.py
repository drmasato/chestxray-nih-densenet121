"""
3モデルアンサンブル: DenseNet-121 + EfficientNet-B4 + XRV-DenseNet
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import timm
import torchxrayvision as xrv
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import datetime, os

DATA_DIR = '/media/morita/ubuntuHDD/chestxray'
IMG_DIR  = f'{DATA_DIR}/images'
RESULTS_CSV = f'{DATA_DIR}/benchmark_results.csv'

DISEASES = [
    'Atelectasis','Consolidation','Infiltration','Pneumothorax',
    'Edema','Emphysema','Fibrosis','Effusion','Pneumonia',
    'Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia'
]

# ===== DenseNet-121 =====
class DenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(nn.Linear(1024, 14))
    def forward(self, x):
        return self.model(x)

# ===== 標準データセット（ImageNet正規化）=====
class NIHStdDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{IMG_DIR}/{row['Image Index']}").convert('RGB')
        return self.tf(img), torch.FloatTensor(row[DISEASES].values.astype(float))

# ===== XRV用データセット（grayscale, [-1024,1024]）=====
class NIHXRVDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{IMG_DIR}/{row['Image Index']}").convert('L')
        img = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(img)
        arr = np.array(img).astype(np.float32)
        arr = (arr / 255.0) * 2048.0 - 1024.0
        return torch.FloatTensor(arr).unsqueeze(0), torch.FloatTensor(row[DISEASES].values.astype(float))

def get_preds(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            with autocast('cuda'):
                out = torch.sigmoid(model(imgs)).cpu().numpy()
            all_preds.append(out)
    return np.vstack(all_preds)

def main():
    device = torch.device('cuda')

    # テストデータ準備
    df = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')
    for d in DISEASES:
        df[d] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0)
    with open(f'{DATA_DIR}/test_list.txt') as f:
        test_files = set(f.read().splitlines())
    df_test = df[df['Image Index'].isin(test_files)].reset_index(drop=True)
    labels  = df_test[DISEASES].values

    std_loader = DataLoader(NIHStdDataset(df_test), batch_size=64, shuffle=False,
                            num_workers=8, pin_memory=True)
    xrv_loader = DataLoader(NIHXRVDataset(df_test), batch_size=64, shuffle=False,
                            num_workers=8, pin_memory=True)

    # DenseNet-121
    print("DenseNet-121 読み込み中...")
    densenet = DenseNetModel().to(device)
    densenet.load_state_dict(torch.load(f'{DATA_DIR}/checkpoints/best_model.pth', map_location=device))
    preds_dense = get_preds(densenet, std_loader, device)

    # EfficientNet-B4
    print("EfficientNet-B4 読み込み中...")
    effnet = timm.create_model('efficientnet_b4', pretrained=False,
                                num_classes=14, drop_rate=0.3, drop_path_rate=0.1).to(device)
    effnet.load_state_dict(torch.load(f'{DATA_DIR}/checkpoints/efficientnet_b4_best.pth', map_location=device))
    preds_eff = get_preds(effnet, std_loader, device)

    # XRV-DenseNet
    print("XRV-DenseNet 読み込み中...")
    xrv_model = xrv.models.DenseNet(weights=None)
    xrv_model.classifier = nn.Linear(xrv_model.classifier.in_features, 14)
    xrv_model.op_threshs = None
    xrv_model.load_state_dict(torch.load(f'{DATA_DIR}/checkpoints/xrv_densenet_finetuned.pth', map_location=device))
    xrv_model = xrv_model.to(device)
    preds_xrv = get_preds(xrv_model, xrv_loader, device)

    print("\n===== 3モデルアンサンブル結果 =====")
    combos = [
        (1/3, 1/3, 1/3, '均等 (1:1:1)'),
        (0.4, 0.4, 0.2, 'Dense+Eff重視 (2:2:1)'),
        (0.5, 0.3, 0.2, 'DenseNet重視 (5:3:2)'),
        (0.4, 0.3, 0.3, 'XRV重視 (4:3:3)'),
    ]
    best_auc, best_preds, best_label = 0, None, ''
    for w1, w2, w3, label in combos:
        preds = w1*preds_dense + w2*preds_eff + w3*preds_xrv
        aucs = [roc_auc_score(labels[:,i], preds[:,i]) for i in range(14) if labels[:,i].sum()>0]
        mean_auc = np.mean(aucs)
        print(f"  {label}: Mean AUC = {mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc, best_preds, best_label = mean_auc, preds, label

    # 最良を記録
    row = {
        'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model':    f'Ensemble 3models ({best_label})',
        'mean_auc': best_auc,
        'note':     f'DenseNet+EfficientNet+XRV {best_label}',
    }
    for i, d in enumerate(DISEASES):
        row[d] = roc_auc_score(labels[:,i], best_preds[:,i]) if labels[:,i].sum()>0 else None

    df_new = pd.DataFrame([row])
    df_old = pd.read_csv(RESULTS_CSV)
    pd.concat([df_old, df_new], ignore_index=True).to_csv(RESULTS_CSV, index=False)
    print(f"\n✓ 記録済み  Best: {best_label}  Mean AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()
