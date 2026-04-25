import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import gradio as gr
import pydicom
import os
import datetime
import timm
import torchxrayvision as xrv
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, XGradCAM,
    EigenCAM, EigenGradCAM, LayerCAM, HiResCAM, ScoreCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CAM_METHODS = {
    'GradCAM':      GradCAM,
    'GradCAM++':    GradCAMPlusPlus,
    'XGradCAM':     XGradCAM,
    'EigenCAM':     EigenCAM,
    'EigenGradCAM': EigenGradCAM,
    'LayerCAM':     LayerCAM,
    'HiResCAM':     HiResCAM,
    'ScoreCAM':     ScoreCAM,
}
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
import matplotlib.pyplot as plt

MODEL_PATH   = '/media/morita/ubuntuHDD/chestxray/checkpoints/best_model.pth'
EFFNET_PATH  = '/media/morita/ubuntuHDD/chestxray/checkpoints/efficientnet_b4_best.pth'
XRV_PATH     = '/media/morita/ubuntuHDD/chestxray/checkpoints/xrv_densenet_finetuned.pth'
THRESHOLD    = 0.5

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

# DenseNet-121
model = ChestXrayModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# EfficientNet-B4
effnet = timm.create_model('efficientnet_b4', pretrained=False,
                            num_classes=14, drop_rate=0.3, drop_path_rate=0.1)
effnet.load_state_dict(torch.load(EFFNET_PATH, map_location=device))
effnet = effnet.to(device)
effnet.eval()

# XRV-DenseNet（4データセット事前学習 Fine-tune）
xrv_model = xrv.models.DenseNet(weights=None)
xrv_model.classifier = nn.Linear(xrv_model.classifier.in_features, 14)
xrv_model.op_threshs = None
xrv_model.load_state_dict(torch.load(XRV_PATH, map_location=device))
xrv_model = xrv_model.to(device)
xrv_model.eval()

print(f"モデル読み込み完了 ({device})  3モデルアンサンブル (DenseNet + EfficientNet + XRV)")

# ===== モデル選択マップ =====
MODEL_OPTIONS = {
    "Ensemble 3モデル (AUC: 0.8149) ★最新":  ("ensemble3",  None),
    "Ensemble 2モデル (AUC: 0.8123)":          ("ensemble2",  None),
    "DenseNet-121    (AUC: 0.8004) baseline":  ("densenet",   None),
    "EfficientNet-B4 (AUC: 0.8051)":           ("efficientnet", None),
    "XRV-DenseNet    (AUC: 0.7860) 4dataset":  ("xrv",        None),
}

def get_probs(model_key, input_t, xrv_t):
    if model_key == "densenet":
        with torch.no_grad():
            return torch.sigmoid(model(input_t)).cpu().numpy()[0]
    elif model_key == "efficientnet":
        with torch.no_grad():
            return torch.sigmoid(effnet(input_t)).cpu().numpy()[0]
    elif model_key == "xrv":
        with torch.no_grad():
            return torch.sigmoid(xrv_model(xrv_t)).cpu().numpy()[0]
    elif model_key == "ensemble2":
        with torch.no_grad():
            p1 = torch.sigmoid(model(input_t)).cpu().numpy()[0]
            p2 = torch.sigmoid(effnet(input_t)).cpu().numpy()[0]
        return 0.5 * p1 + 0.5 * p2
    else:  # ensemble3
        with torch.no_grad():
            p1 = torch.sigmoid(model(input_t)).cpu().numpy()[0]
            p2 = torch.sigmoid(effnet(input_t)).cpu().numpy()[0]
            p3 = torch.sigmoid(xrv_model(xrv_t)).cpu().numpy()[0]
        return 0.4 * p1 + 0.4 * p2 + 0.2 * p3

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

# ===== レポート生成 =====
# 疾患ごとの推奨文（英語ベース臨床知識）
REPORT_HINTS = {
    'Atelectasis':        ('無気肺',       '気管支閉塞や術後変化が疑われます。深呼吸訓練・胸部CTを推奨します。'),
    'Consolidation':      ('浸潤影',       '細菌性肺炎や出血が疑われます。臨床症状と血液検査（WBC/CRP）で評価してください。'),
    'Infiltration':       ('浸潤',         '感染性または炎症性病変が疑われます。経過観察または抗菌薬投与を検討してください。'),
    'Pneumothorax':       ('気胸',         '**緊急性あり。** 呼吸困難の程度を確認し、必要に応じて脱気処置を行ってください。'),
    'Edema':              ('肺水腫',       '心不全や腎不全に伴う肺水腫が疑われます。利尿薬・循環動態の評価を推奨します。'),
    'Emphysema':          ('肺気腫',       'COPD関連の肺気腫が疑われます。肺機能検査（スパイロメトリー）を推奨します。'),
    'Fibrosis':           ('肺線維症',     '間質性肺疾患が疑われます。HRCT・呼吸器内科紹介を推奨します。'),
    'Effusion':           ('胸水',         '胸水貯留が疑われます。原因精査（心不全・悪性腫瘍・感染）のため胸部エコーを推奨します。'),
    'Pneumonia':          ('肺炎',         '肺炎が疑われます。発熱・SpO2・炎症マーカーの確認と適切な抗菌薬投与を検討してください。'),
    'Pleural_Thickening': ('胸膜肥厚',     '過去の炎症・石綿曝露歴の確認を推奨します。'),
    'Cardiomegaly':       ('心肥大',       '心拡大が疑われます。心エコーによる左室機能評価を推奨します。'),
    'Nodule':             ('結節',         '肺結節が疑われます。悪性腫瘍除外のため胸部CTおよびフォローアップを推奨します。'),
    'Mass':               ('腫瘤',         '**要精査。** 肺腫瘤が疑われます。悪性腫瘍の除外のため早急に胸部CT・専門科紹介を推奨します。'),
    'Hernia':             ('ヘルニア',     '横隔膜ヘルニアが疑われます。上部消化管精査を推奨します。'),
}

def generate_report(probs):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    pos  = [(DISEASES[i], probs[i]) for i in range(14) if probs[i] >= 0.5]
    warn = [(DISEASES[i], probs[i]) for i in range(14) if 0.3 <= probs[i] < 0.5]
    neg  = [(DISEASES[i], probs[i]) for i in range(14) if probs[i] < 0.3]

    lines = [
        "=" * 48,
        "  胸部X線AI自動読影レポート",
        f"  生成日時: {now}",
        f"  使用モデル: DenseNet-121 + EfficientNet-B4 + XRV-DenseNet 3モデルアンサンブル (Mean AUC: 0.8149)",
        "=" * 48,
        "",
        "【所見】",
    ]

    if pos:
        lines.append("▶ 陽性疑い（確率≥50%）:")
        for d, p in sorted(pos, key=lambda x: -x[1]):
            jp, hint = REPORT_HINTS[d]
            lines.append(f"  ・{jp}（{p*100:.0f}%）")
            lines.append(f"    → {hint}")
    else:
        lines.append("  陽性疑い所見なし")

    lines.append("")
    if warn:
        lines.append("▷ 要注意（確率30〜49%）:")
        for d, p in sorted(warn, key=lambda x: -x[1]):
            jp, _ = REPORT_HINTS[d]
            lines.append(f"  ・{jp}（{p*100:.0f}%）— 経過観察を推奨")

    lines += [
        "",
        "【インプレッション】",
    ]
    if pos:
        for rank, (d, p) in enumerate(sorted(pos, key=lambda x: -x[1]), 1):
            jp, _ = REPORT_HINTS[d]
            conf = "高" if p >= 0.7 else "中"
            lines.append(f"  {rank}. {jp} を疑います（確率 {p*100:.0f}%、信頼度: {conf}）")
    else:
        lines.append("  明らかな異常所見は検出されませんでした。")

    lines += [
        "",
        "─" * 48,
        "⚠️ 本レポートはAIによる自動生成です。",
        "   必ず専門医による確認・読影を行ってください。",
        "   患者画像は必ず匿名化済みのものを使用してください。",
        "─" * 48,
    ]
    return "\n".join(lines)


# ===== ベンチマーク履歴グラフ =====
BENCHMARK_CSV = '/media/morita/ubuntuHDD/chestxray/benchmark_results.csv'

def build_benchmark_fig():
    df = pd.read_csv(BENCHMARK_CSV)
    # ベストのみ（同名モデルは最高AUCの行）
    best = df.loc[df.groupby('model')['mean_auc'].idxmax()].sort_values('mean_auc')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 全モデルの Mean AUC 棒グラフ
    colors = ['#e74c3c' if 'Ensemble' in m else '#3498db' for m in best['model']]
    axes[0].barh(best['model'], best['mean_auc'], color=colors)
    axes[0].axvline(x=0.778, color='gray', linestyle='--', linewidth=1, label='放射線科医 0.778')
    axes[0].axvline(x=0.841, color='orange', linestyle='--', linewidth=1, label='CheXNet 0.841')
    axes[0].set_xlim(0.65, 0.97)
    axes[0].set_xlabel('Mean AUC')
    axes[0].set_title('モデル進化の比較（赤: アンサンブル）')
    axes[0].legend(fontsize=8)
    for i, (_, row) in enumerate(best.iterrows()):
        axes[0].text(row['mean_auc'] + 0.002, i, f"{row['mean_auc']:.4f}", va='center', fontsize=8)

    # 右: 疾患別 AUC ヒートマップ（上位モデル5つ）
    top5 = best.tail(5)[DISEASES].T
    top5.columns = best.tail(5)['model'].str[:20].values
    im = axes[1].imshow(top5.values.astype(float), cmap='RdYlGn', vmin=0.65, vmax=0.97, aspect='auto')
    axes[1].set_xticks(range(len(top5.columns)))
    axes[1].set_xticklabels(top5.columns, rotation=30, ha='right', fontsize=7)
    axes[1].set_yticks(range(len(DISEASES_JP)))
    axes[1].set_yticklabels(DISEASES_JP, fontsize=8)
    axes[1].set_title('疾患別 AUC ヒートマップ（上位5モデル）')
    plt.colorbar(im, ax=axes[1])
    for i in range(len(DISEASES)):
        for j in range(len(top5.columns)):
            val = top5.values[i, j]
            if not np.isnan(float(val)):
                axes[1].text(j, i, f'{float(val):.2f}', ha='center', va='center', fontsize=6)

    plt.tight_layout()
    return fig

# ===== 推論関数 =====
def predict(file_obj, model_choice, cam_method='GradCAM++', cam_threshold=0.4):
    if file_obj is None:
        return {}, None, None, None, "", ""

    model_key = MODEL_OPTIONS.get(model_choice, ("ensemble3", None))[0]

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

    # XRV用テンソル
    gray = pil_img.resize((224, 224)).convert('L')
    gray_arr = (np.array(gray).astype(np.float32) / 255.0) * 2048.0 - 1024.0
    xrv_t = torch.FloatTensor(gray_arr).unsqueeze(0).unsqueeze(0).to(device)

    probs = get_probs(model_key, input_t, xrv_t)

    # 結果ラベル（上位疾患）
    label_dict = {}
    for jp, prob in zip(DISEASES_JP, probs):
        label_dict[jp] = float(prob)

    # CAM（最も確率の高い疾患）
    top_idx = int(np.argmax(probs))
    top_name = DISEASES_JP[top_idx]
    target_layer = [model.model.features.denseblock4.denselayer16.conv2]

    try:
        cam_cls = CAM_METHODS.get(cam_method, GradCAMPlusPlus)
        cam = cam_cls(model=model, target_layers=target_layer)
        grayscale_cam = cam(input_tensor=input_t,
                            targets=[ClassifierOutputTarget(top_idx)])[0]
        # 閾値以下をゼロにして集中させる
        focused = np.where(grayscale_cam >= cam_threshold, grayscale_cam, 0.0)
        # 正規化
        if focused.max() > 0:
            focused = focused / focused.max()
        cam_img = show_cam_on_image(img_np, focused, use_rgb=True)
        cam_pil = Image.fromarray(cam_img)
    except Exception as e:
        print(f"CAM error: {e}")
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

    # レポート生成
    report = generate_report(probs)

    orig_pil = pil_img.resize((224, 224))
    return label_dict, orig_pil, cam_pil, fig, meta_info, report

# ===== Gradio UI =====
with gr.Blocks(title="胸部X線AI診断") as demo:
    gr.Markdown("""
    # 胸部X線AI診断システム
    **NIH ChestX-ray14学習済み｜最高 Mean AUC: 0.8149（3モデルアンサンブル）**
    > ⚠️ 研究・学習目的のみ。臨床診断には使用しないでください。
    """)

    with gr.Tabs():
        # ===== TAB 1: 診断 =====
        with gr.Tab("診断"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input  = gr.File(label="画像をアップロード（DICOM / PNG / JPG）")
                    meta_output = gr.Textbox(label="DICOMメタ情報", interactive=False)
                    model_selector = gr.Dropdown(
                        choices=list(MODEL_OPTIONS.keys()),
                        value=list(MODEL_OPTIONS.keys())[0],
                        label="使用モデル",
                    )
                    with gr.Row():
                        cam_selector = gr.Dropdown(
                            choices=list(CAM_METHODS.keys()),
                            value='GradCAM++',
                            label="CAM 手法",
                        )
                        cam_thresh = gr.Slider(
                            minimum=0.0, maximum=0.9, value=0.4, step=0.05,
                            label="集中度",
                        )
                    run_btn = gr.Button("診断する", variant="primary")

            with gr.Row():
                orig_output = gr.Image(label="元画像")
                cam_output  = gr.Image(label="CAM（注目領域）")

            with gr.Row():
                label_output = gr.Label(num_top_classes=5, label="上位5疾患")
                chart_output = gr.Plot(label="疾患別確率")

            with gr.Row():
                report_output = gr.Textbox(label="AI自動読影レポート", lines=20, interactive=False)

            run_btn.click(
                fn=predict,
                inputs=[file_input, model_selector, cam_selector, cam_thresh],
                outputs=[label_output, orig_output, cam_output, chart_output, meta_output, report_output]
            )

        # ===== TAB 2: モデル進化の記録 =====
        with gr.Tab("モデル進化の記録"):
            gr.Markdown("""
            ### 学習・改善の記録
            各モデルのテストセット AUC（NIH ChestX-ray14 公式テスト 25,596枚）
            """)
            refresh_btn = gr.Button("グラフを更新", variant="secondary")
            bench_plot  = gr.Plot(label="精度の進化")
            refresh_btn.click(fn=build_benchmark_fig, inputs=[], outputs=bench_plot)
            demo.load(fn=build_benchmark_fig, inputs=[], outputs=bench_plot)

            gr.Markdown("""
            | モデル | Mean AUC | 改善点 |
            |--------|----------|--------|
            | DenseNet-121 | 0.8004 | ImageNet転移学習ベースライン |
            | EfficientNet-B4 | 0.8051 | Dropout・EarlyStopping追加 |
            | Ensemble 2モデル | 0.8123 | 予測値の平均アンサンブル |
            | XRV Fine-tune | 0.7860 | 4データセット事前学習（単体） |
            | **Ensemble 3モデル** | **0.8149** | 多様性アンサンブルで最高精度 |
            | 放射線科医（参考）| 0.778 | Wang et al. 2017 |
            | CheXNet（参考）| 0.841 | Stanford 2017 |

            > 将来追加予定: Swin Transformer / ViT-B
            """)

if __name__ == '__main__':
    demo.launch(share=False, server_port=7862)
