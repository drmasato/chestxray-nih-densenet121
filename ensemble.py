"""
Step 2: DenseNet-121 + EfficientNet-B4 アンサンブル
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import timm
from torch.amp import autocast
from benchmark import record_result, get_test_loader, NIHChestDataset, DISEASES

DATA_DIR = '/media/morita/ubuntuHDD/chestxray'

# ===== モデル定義 =====
class DenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(nn.Linear(1024, 14))
    def forward(self, x):
        return self.model(x)

def make_efficientnet():
    return timm.create_model('efficientnet_b4', pretrained=False,
                              num_classes=14, drop_rate=0.3,
                              drop_path_rate=0.1)

def get_predictions(model, loader, device):
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

    # モデル読み込み
    print("DenseNet-121 読み込み中...")
    densenet = DenseNetModel().to(device)
    densenet.load_state_dict(torch.load(
        f'{DATA_DIR}/checkpoints/best_model.pth', map_location=device))

    print("EfficientNet-B4 読み込み中...")
    efficientnet = make_efficientnet().to(device)
    efficientnet.load_state_dict(torch.load(
        f'{DATA_DIR}/checkpoints/efficientnet_b4_best.pth', map_location=device))

    loader = get_test_loader(img_size=224, batch_size=64)

    # 各モデルの予測
    print("DenseNet-121 推論中...")
    preds_dense = get_predictions(densenet, loader, device)

    print("EfficientNet-B4 推論中...")
    preds_effnet = get_predictions(efficientnet, loader, device)

    # ラベル取得
    df = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')
    for d in DISEASES:
        df[d] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0)
    with open(f'{DATA_DIR}/test_list.txt') as f:
        test_files = set(f.read().splitlines())
    df_test = df[df['Image Index'].isin(test_files)].reset_index(drop=True)
    labels  = df_test[DISEASES].values

    from sklearn.metrics import roc_auc_score
    DISEASES_JP = [
        '無気肺','浸潤影','浸潤','気胸','肺水腫','肺気腫',
        '肺線維症','胸水','肺炎','胸膜肥厚','心肥大','結節','腫瘤','ヘルニア'
    ]

    print("\n===== アンサンブル結果 =====")

    # 重み付き平均を試す（DenseNet重視）
    for w1, w2, label in [(0.5, 0.5, '均等'), (0.6, 0.4, 'DenseNet重視'), (0.4, 0.6, 'EfficientNet重視')]:
        preds_ensemble = w1 * preds_dense + w2 * preds_effnet
        aucs = []
        for i, (d, jp) in enumerate(zip(DISEASES, DISEASES_JP)):
            if labels[:, i].sum() > 0:
                auc = roc_auc_score(labels[:, i], preds_ensemble[:, i])
                aucs.append(auc)
        mean_auc = np.mean(aucs)
        print(f"  {label} ({w1:.1f}:{w2:.1f}): Mean AUC = {mean_auc:.4f}")

    # ベストの組み合わせで記録
    best_preds = 0.5 * preds_dense + 0.5 * preds_effnet

    # ダミーモデルを使ってrecord_result互換に
    import datetime, os
    import csv
    row = {
        'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model':    'Ensemble DenseNet+EfficientNet',
        'mean_auc': np.mean([roc_auc_score(labels[:, i], best_preds[:, i])
                             for i in range(14) if labels[:, i].sum() > 0]),
        'note':     '均等アンサンブル (0.5+0.5)',
    }
    for i, d in enumerate(DISEASES):
        if labels[:, i].sum() > 0:
            row[d] = roc_auc_score(labels[:, i], best_preds[:, i])
        else:
            row[d] = None

    results_path = f'{DATA_DIR}/benchmark_results.csv'
    df_new = pd.DataFrame([row])
    df_old = pd.read_csv(results_path)
    pd.concat([df_old, df_new], ignore_index=True).to_csv(results_path, index=False)
    print(f"\n✓ 記録済み  Mean AUC: {row['mean_auc']:.4f}")

if __name__ == '__main__':
    main()
