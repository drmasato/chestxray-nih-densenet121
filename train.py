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
from torchvision import models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ===== 設定 =====
DATA_DIR   = '/media/morita/ubuntuHDD/chestxray'
IMG_DIR    = f'{DATA_DIR}/images'
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
NUM_WORKERS = 8
SAVE_DIR   = f'{DATA_DIR}/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

DISEASES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

# ===== Dataset =====
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

# ===== Transform =====
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===== Model =====
class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.model = models.densenet121(weights='IMAGENET1K_V1')
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 14)
        )

    def forward(self, x):
        return self.model(x)

# ===== AUC評価 =====
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            with autocast('cuda'):
                outputs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_preds.append(outputs)
            all_labels.append(labels.numpy())

    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    aucs = []
    for i, disease in enumerate(DISEASES):
        if labels[:, i].sum() > 0:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            aucs.append(auc)
            print(f"  {disease:<22}: {auc:.4f}")
    mean_auc = np.mean(aucs)
    print(f"  {'Mean AUC':<22}: {mean_auc:.4f}")
    return mean_auc

# ===== Main =====
def main():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # データ読み込み
    df = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')
    for disease in DISEASES:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)

    with open(f'{DATA_DIR}/train_val_list.txt') as f:
        train_val = set(f.read().splitlines())
    with open(f'{DATA_DIR}/test_list.txt') as f:
        test_files = set(f.read().splitlines())

    df_trainval = df[df['Image Index'].isin(train_val)]
    df_test     = df[df['Image Index'].isin(test_files)]
    df_train    = df_trainval.sample(frac=0.9, random_state=42)
    df_val      = df_trainval.drop(df_train.index)

    print(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")

    train_ds = NIHChestDataset(df_train, IMG_DIR, train_tf)
    val_ds   = NIHChestDataset(df_val,   IMG_DIR, val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

    # モデル
    model = ChestXrayModel().to(device)

    # クラス不均衡対策
    pos_weight = torch.tensor([
        (len(df_train) - df_train[d].sum()) / (df_train[d].sum() + 1e-6)
        for d in DISEASES
    ]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler('cuda')
    writer    = SummaryWriter(f'{DATA_DIR}/runs')

    best_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # 学習
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        # 評価（5エポックごと）
        if epoch % 5 == 0 or epoch == 1:
            print(f"\n--- Epoch {epoch} Validation ---")
            mean_auc = evaluate(model, val_loader, device)
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('AUC/val',    mean_auc, epoch)

            if mean_auc > best_auc:
                best_auc = mean_auc
                torch.save(model.state_dict(), f'{SAVE_DIR}/best_model.pth')
                print(f"  ★ Best model saved (AUC: {best_auc:.4f})")

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

    print(f"\n学習完了！Best AUC: {best_auc:.4f}")
    writer.close()

if __name__ == '__main__':
    main()
