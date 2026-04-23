"""
Step 3: torchxrayvision 事前学習モデルのFine-tuning
densenet121-res224-all: NIH + CheXpert + MIMIC-CXR + PadChest 事前学習済み
期待 AUC: 0.83+
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
import torchxrayvision as xrv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from benchmark import record_result, DISEASES
import datetime, csv

DATA_DIR   = '/media/morita/ubuntuHDD/chestxray'
IMG_DIR    = f'{DATA_DIR}/images'
SAVE_PATH  = f'{DATA_DIR}/checkpoints/xrv_densenet_finetuned.pth'
RESULTS_CSV = f'{DATA_DIR}/benchmark_results.csv'
BATCH_SIZE = 32
EPOCHS     = 20
LR_HEAD    = 1e-3
LR_BACK    = 5e-5   # backbone は低め
IMG_SIZE   = 224
PATIENCE   = 5
NUM_WORKERS = 8


# ===== torchxrayvision 用データセット =====
# XRV モデルは grayscale [-1024, 1024] 正規化が必要
class NIHChestXRVDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.aug_tf = transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['Image Index']}").convert('L')
        img = (self.aug_tf if self.augment else self.val_tf)(img)
        arr = np.array(img).astype(np.float32)
        # [0,255] → [-1024, 1024]
        arr = (arr / 255.0) * 2048.0 - 1024.0
        tensor = torch.FloatTensor(arr).unsqueeze(0)  # (1, H, W)
        label = torch.FloatTensor(row[DISEASES].values.astype(float))
        return tensor, label


def evaluate_xrv(model, loader, device):
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
    if np.isnan(preds).any():
        print("⚠️ NaN検出")
        return {'Mean': 0.0}
    aucs = {}
    for i, d in enumerate(DISEASES):
        if labels[:, i].sum() > 0:
            aucs[d] = roc_auc_score(labels[:, i], preds[:, i])
    aucs['Mean'] = np.mean(list(aucs.values()))
    return aucs


def record_xrv_result(model_name, aucs, note=""):
    print(f"\n===== {model_name} =====")
    for d, v in aucs.items():
        print(f"  {d:<22}: {v:.4f}")

    row = {
        'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model':    model_name,
        'mean_auc': aucs['Mean'],
        'note':     note,
    }
    row.update({d: aucs.get(d, None) for d in DISEASES})
    df_new = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        df_old = pd.read_csv(RESULTS_CSV)
        pd.concat([df_old, df_new], ignore_index=True).to_csv(RESULTS_CSV, index=False)
    else:
        df_new.to_csv(RESULTS_CSV, index=False)
    print(f"✓ 記録済み  Mean AUC: {aucs['Mean']:.4f}")


def main():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # データ準備
    df = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')
    for d in DISEASES:
        df[d] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0)

    with open(f'{DATA_DIR}/train_val_list.txt') as f:
        train_val = set(f.read().splitlines())
    with open(f'{DATA_DIR}/test_list.txt') as f:
        test_files = set(f.read().splitlines())

    df_tv    = df[df['Image Index'].isin(train_val)]
    df_train = df_tv.sample(frac=0.9, random_state=42)
    df_val   = df_tv.drop(df_train.index)
    df_test  = df[df['Image Index'].isin(test_files)].reset_index(drop=True)
    print(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")

    train_loader = DataLoader(
        NIHChestXRVDataset(df_train, IMG_DIR, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        NIHChestXRVDataset(df_val, IMG_DIR, augment=False),
        batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        NIHChestXRVDataset(df_test, IMG_DIR, augment=False),
        batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ===== torchxrayvision 事前学習モデル =====
    print("\ntorchxrayvision DenseNet-121 (densenet121-res224-all) 読み込み中...")
    base = xrv.models.DenseNet(weights="densenet121-res224-all")

    # 分類頭を14疾患用に付け替え（事前学習済み重みの最初14行を引き継ぐ）
    in_feats = base.classifier.in_features
    pretrained_w = base.classifier.weight.data[:14].clone()
    pretrained_b = base.classifier.bias.data[:14].clone()
    base.classifier = nn.Linear(in_feats, 14)
    base.classifier.weight.data = pretrained_w
    base.classifier.bias.data   = pretrained_b
    base.op_threshs = None   # 18クラス用しきい値をリセット
    model = base.to(device)
    print(f"  分類頭: {in_feats} → 14（事前学習重み引き継ぎ）")

    # 段階的学習率
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n],
         'lr': LR_BACK},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD},
    ], weight_decay=1e-5)

    pos_weight = torch.tensor([
        (len(df_train) - df_train[d].sum()) / (df_train[d].sum() + 1e-6)
        for d in DISEASES
    ]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler('cuda')
    writer    = SummaryWriter(f'{DATA_DIR}/runs/xrv_densenet')

    best_auc   = 0.0
    no_improve = 0

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
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)

        if epoch % 5 == 0 or epoch == 1:
            aucs = evaluate_xrv(model, val_loader, device)
            writer.add_scalar('AUC/val', aucs['Mean'], epoch)
            record_xrv_result(
                f'XRV-DenseNet epoch{epoch}', aucs,
                note=f'torchxrayvision pretrain, epoch={epoch}'
            )
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

    # テストセットで最終評価
    print("\n===== テストセット最終評価 =====")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_aucs = evaluate_xrv(model, test_loader, device)
    record_xrv_result(
        'XRV-DenseNet (test)', test_aucs,
        note=f'torchxrayvision pretrain, best epoch, test set'
    )
    print(f"\n学習完了！Best Val AUC: {best_auc:.4f}  Test AUC: {test_aucs['Mean']:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
