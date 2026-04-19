import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import gradio as gr
import pydicom
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
import matplotlib.pyplot as plt

MODEL_PATH = '/media/morita/ubuntuHDD/chestxray/checkpoints/best_model.pth'
THRESHOLD  = 0.5

DISEASES    = [
    'Atelectasis','Consolidation','Infiltration','Pneumothorax',
    'Edema','Emphysema','Fibrosis','Effusion','Pneumonia',
    'Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia'
]
DISEASES_JP = [
    '無気肺','浸潤影','浸潤','気胸','肺水腫','肺気腫',
    '肺線維症','胸水','肺炎','胸膜肥厚','心肥大','結節','腫瘤','ヘルニア'
]

# ===== モデル =====
class ChestXrayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(nn.Linear(1024, 14))
    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = ChestXrayModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"モデル読み込み完了 ({device})")

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===== DICOM読み込み =====
def load_dicom(path):
    dcm = pydicom.dcmread(path)

    # pixel_array取得
    arr = dcm.pixel_array.astype(float)

    # マルチフレームの場合は最初のフレームを使用
    if arr.ndim == 3 and arr.shape[0] > 3:
        arr = arr[0]

    # Photometric Interpretationに応じた反転処理
    pi = getattr(dcm, 'PhotometricInterpretation', '')
    if pi == 'MONOCHROME1':
        arr = arr.max() - arr  # 白黒反転

    # ウィンドウ処理（VOI LUT があれば使用、なければ正規化）
    try:
        from pydicom.pixels import apply_voi_lut
        arr = apply_voi_lut(arr, dcm).astype(float)
    except Exception:
        pass

    # 0〜255に正規化
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    pil_img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')

    # メタ情報取得（患者名・IDは取得しない）
    date     = str(getattr(dcm, 'StudyDate',    '不明'))
    view     = str(getattr(dcm, 'ViewPosition', '不明'))
    modality = str(getattr(dcm, 'Modality',     '不明'))
    meta_str = f"モダリティ: {modality}　撮影日: {date}　体位: {view}"
    return pil_img, meta_str

# ===== DICOMかどうかをマジックバイトで判定 =====
def is_dicom(path):
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False

# ===== 推論関数 =====
def predict(file_obj):
    if file_obj is None:
        return {}, None, None, ""

    path = file_obj.name
    ext  = os.path.splitext(path)[-1].lower()
    meta_info = ""

    if ext == '.dcm' or is_dicom(path):
        pil_img, meta_info = load_dicom(path)
    else:
        pil_img = Image.open(path).convert('RGB')

    img_rgb  = pil_img.resize((224, 224))
    img_np   = np.array(img_rgb).astype(np.float32) / 255.0
    input_t  = tf(pil_img).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        probs = torch.sigmoid(model(input_t)).cpu().numpy()[0]

    # 結果ラベル（上位疾患）
    label_dict = {}
    for jp, prob in zip(DISEASES_JP, probs):
        label_dict[jp] = float(prob)

    # Grad-CAM（最も確率の高い疾患）
    top_idx = int(np.argmax(probs))
    target_layer = [model.model.features.denseblock4.denselayer16.conv2]

    try:
        cam = GradCAM(model=model, target_layers=target_layer)
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        grayscale_cam = cam(input_tensor=input_t,
                            targets=[ClassifierOutputTarget(top_idx)])[0]
        cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        cam_pil = Image.fromarray(cam_img)
    except Exception:
        cam_pil = pil_img

    # 棒グラフ
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#e74c3c' if p >= THRESHOLD else '#3498db' for p in probs]
    bars = ax.barh(DISEASES_JP, probs, color=colors)
    ax.axvline(x=THRESHOLD, color='gray', linestyle='--', linewidth=1, label=f'閾値 {THRESHOLD}')
    ax.set_xlim(0, 1)
    ax.set_xlabel('確率')
    ax.set_title(f'診断結果  （赤: 陽性疑い ≥ {THRESHOLD}）')
    for bar, p in zip(bars, probs):
        if p >= 0.1:
            ax.text(p + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{p*100:.0f}%', va='center', fontsize=8)
    ax.legend()
    plt.tight_layout()

    return label_dict, cam_pil, fig, meta_info

# ===== Gradio UI =====
with gr.Blocks(title="胸部X線AI診断") as demo:
    gr.Markdown("""
    # 胸部X線AI診断システム
    **DenseNet-121 / NIH ChestX-ray14学習済み（Mean AUC: 0.8052）**

    > ⚠️ 本ツールは研究・学習目的のみです。臨床診断には使用しないでください。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input  = gr.File(
                label="胸部X線画像をアップロード（DICOM / PNG / JPG）",
            )
            meta_output = gr.Textbox(label="DICOMメタ情報", interactive=False)
            run_btn     = gr.Button("診断する", variant="primary")

        with gr.Column(scale=1):
            cam_output  = gr.Image(label="Grad-CAM（注目領域）")

    with gr.Row():
        label_output = gr.Label(num_top_classes=5, label="上位5疾患")
        chart_output = gr.Plot(label="疾患別確率")

    run_btn.click(
        fn=predict,
        inputs=file_input,
        outputs=[label_output, cam_output, chart_output, meta_output]
    )

    gr.Markdown("""
    ### 使い方
    1. 胸部X線画像（**DICOM / PNG / JPG**）をアップロード
    2. 「診断する」をクリック
    3. 上位5疾患の確率と注目領域（Grad-CAM）を確認

    > ⚠️ 実際の患者画像を使用する場合は**必ず匿名化済みのもの**を使用してください。
    """)

if __name__ == '__main__':
    demo.launch(share=False, server_port=7861)
