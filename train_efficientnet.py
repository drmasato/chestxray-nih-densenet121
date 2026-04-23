"""
Step 1: EfficientNet-B4 学習スクリプト
完了後に benchmark.py で自動記録
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from benchmark import record_result, get_test_loader, NIHChestDataset

DATA_DIR    = '/media/morita/ubuntuHDD/chestxray'
IMG_DIR     = f'{DATA_DIR}/images'
SAVE_PATH   = f'{DATA_DIR}/checkpoints/efficientnet_b4_best.pth'
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
IMG_SIZE    = 224
NUM_WORKERS = 8
PATIENCE    = 5

DISEASES = [
    'Atelectasis','Consolidation','Infiltration','Pneumothorax',
    'Edema','Emphysema','Fibrosis','Effusion','Pneumonia',
    'Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia'
]

train_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def main():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"入力サイズ: {IMG_SIZE}x{IMG_SIZE}")

    # データ準備
    df = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')
    for d in DISEASES:
        df[d] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0)

    with open(f'{DATA_DIR}/train_val_list.txt') as f:
        train_val = set(f.read().splitlines())
    df_tv    = df[df['Image Index'].isin(train_val)]
    df_train = df_tv.sample(frac=0.9, random_state=42)
    df_val   = df_tv.drop(df_train.index)
    print(f"Train: {len(df_train)}  Val: {len(df_val)}")

    train_ds    = NIHChestDataset(df_train, IMG_DIR, train_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

    # EfficientNet-B4（Dropout追加で過学習抑制）
    model = timm.create_model('efficientnet_b4', pretrained=True,
                               num_classes=14, drop_rate=0.3,
                               drop_path_rate=0.1)
    model = model.to(device)
    print("EfficientNet-B4 読み込み完了")

    pos_weight = torch.tensor([
        (len(df_train) - df_train[d].sum()) / (df_train[d].sum() + 1e-6)
        for d in DISEASES
    ]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler('cuda')
    writer    = SummaryWriter(f'{DATA_DIR}/runs/efficientnet_b4')

    best_auc   = 0.0
    no_improve = 0
    val_loader = get_test_loader(img_size=IMG_SIZE, batch_size=64)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            aucs = record_result(
                f'EfficientNet-B4 epoch{epoch}',
                model, val_loader, device,
                note=f'512px, epoch={epoch}'
            )
            writer.add_scalar('AUC/val', aucs['Mean'], epoch)
            if aucs['Mean'] > best_auc:
                best_auc   = aucs['Mean']
                no_improve = 0
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"★ Best model saved (AUC: {best_auc:.4f})")
            else:
                no_improve += 1
                print(f"  改善なし {no_improve}/{PATIENCE}")
                if no_improve >= PATIENCE:
                    print("EarlyStopping 発動")
                    break

        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

    print(f"\n学習完了！Best AUC: {best_auc:.4f}")
    writer.close()

if __name__ == '__main__':
    main()
