import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
import cv2

DATA_DIR   = '/media/morita/ubuntuHDD/chestxray'
IMG_DIR    = f'{DATA_DIR}/images'
MODEL_PATH = f'{DATA_DIR}/checkpoints/best_model.pth'
OUT_DIR    = f'{DATA_DIR}/results/gradcam'
os.makedirs(OUT_DIR, exist_ok=True)

DISEASES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
DISEASES_JP = [
    '無気肺', '浸潤影', '浸潤', '気胸', '肺水腫', '肺気腫',
    '肺線維症', '胸水', '肺炎', '胸膜肥厚', '心肥大', '結節', '腫瘤', 'ヘルニア'
]

class ChestXrayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(nn.Linear(1024, 14))

    def forward(self, x):
        return self.model(x)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.sigmoid(output[0]).detach().cpu().numpy()

def overlay_cam(img_np, cam, alpha=0.4):
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (img_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlay

def main():
    device = torch.device('cuda')

    # モデル読み込み
    model = ChestXrayModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Grad-CAM対象層（DenseNet-121の最終特徴層）
    target_layer = model.model.features.denseblock4.denselayer16.conv2
    gradcam = GradCAM(model, target_layer)

    # テストデータから各疾患の陽性サンプルを1枚ずつ取得
    df = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')
    for d in DISEASES:
        df[d] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0)

    with open(f'{DATA_DIR}/test_list.txt') as f:
        test_files = set(f.read().splitlines())
    df_test = df[df['Image Index'].isin(test_files)].reset_index(drop=True)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Grad-CAM生成中...")
    fig, axes = plt.subplots(3, 5, figsize=(20, 13))
    axes = axes.flatten()

    for i, (disease, jp) in enumerate(zip(DISEASES, DISEASES_JP)):
        # 各疾患の陽性サンプルを取得
        samples = df_test[df_test[disease] == 1]
        if len(samples) == 0:
            axes[i].axis('off')
            continue

        row = samples.iloc[0]
        img_path = f"{IMG_DIR}/{row['Image Index']}"
        img_orig = Image.open(img_path).convert('RGB')
        img_orig = img_orig.resize((224, 224))
        img_np   = np.array(img_orig)

        input_tensor = tf(img_orig).unsqueeze(0).to(device)
        cam, probs = gradcam.generate(input_tensor, i)
        overlay = overlay_cam(img_np, cam)

        axes[i].imshow(overlay)
        axes[i].set_title(f'{jp}\n予測: {probs[i]:.2f}', fontsize=9)
        axes[i].axis('off')
        print(f"  {disease} ({jp}): 予測={probs[i]:.3f}")

    # 残り軸を非表示
    for j in range(len(DISEASES), len(axes)):
        axes[j].axis('off')

    plt.suptitle('Grad-CAM 診断根拠の可視化（テストセット陽性サンプル）', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/gradcam_all_diseases.png', dpi=150, bbox_inches='tight')
    print(f"\n保存: {OUT_DIR}/gradcam_all_diseases.png")
    print("Grad-CAM完了！")

if __name__ == '__main__':
    main()
