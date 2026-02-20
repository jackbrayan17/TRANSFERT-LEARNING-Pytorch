"""
Application Streamlit — Transfer Learning | EfficientNet-B4
Rice Leaf Disease Classification — Fine-tuning from ImageNet
Interface professionnelle complète
"""

import os
import sys
import json
import io
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Transfer Learning — EfficientNet-B4 Rice",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1826 100%); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1826 0%, #0a0e1a 100%);
        border-right: 1px solid #1a3a4a;
    }

    h1, h2, h3 { color: #e0f0ff !important; font-family: 'Segoe UI', sans-serif; }
    h1 { font-size: 2.2rem !important; font-weight: 700 !important; }

    .metric-card {
        background: linear-gradient(135deg, #0f2030 0%, #152535 100%);
        border: 1px solid #1a4060;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #00b4d8; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #90b0cc; margin: 0; font-weight: 500; }
    .metric-delta { font-size: 0.8rem; color: #00e5a0; margin: 2px 0 0 0; }

    .pred-result {
        background: linear-gradient(135deg, #0f2030, #152535);
        border: 2px solid #00b4d8;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .pred-class { font-size: 2rem; font-weight: 800; color: #00b4d8; }
    .pred-conf  { font-size: 1.2rem; color: #00e5a0; }

    .info-box {
        background: rgba(0, 180, 216, 0.1);
        border-left: 4px solid #00b4d8;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e0f0ff;
    }
    .warn-box {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid #ff6b6b;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e0f0ff;
    }
    .success-box {
        background: rgba(0, 229, 160, 0.1);
        border-left: 4px solid #00e5a0;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e0f0ff;
    }

    .stTabs [data-baseweb="tab"] {
        background: #0f2030;
        border-radius: 8px;
        color: #90b0cc;
        border: 1px solid #1a4060;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00b4d8, #0077b6) !important;
        color: white !important;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00b4d8, #00e5a0);
    }

    .stButton > button {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0077b6, #00b4d8);
        box-shadow: 0 4px 15px rgba(0,180,216,0.4);
    }

    hr { border-color: #1a4060; opacity: 0.5; }

    /* Badge Transfer Learning */
    .tl-badge {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 700; display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
BASE   = Path(__file__).parent.parent
TL_DIR = Path(__file__).parent
DATA   = BASE / 'riceleaf'
IMG_SIZE = 380

CLASS_INFO = {
    "blast"       : {"color": "#ff6b6b", "icon": "🔴", "fr": "Pyriculariose",
                     "desc": "Magnaporthe oryzae — taches losangiques grises/brunes.",
                     "severity": "Élevée", "treatment": "Fongicides triazoles"},
    "healthy"     : {"color": "#00e5a0", "icon": "🟢", "fr": "Saine",
                     "desc": "Feuille en bonne santé sans symptômes.",
                     "severity": "Aucune", "treatment": "Bonnes pratiques agronomiques"},
    "insect"      : {"color": "#ffd93d", "icon": "🟡", "fr": "Dégâts Insectes",
                     "desc": "Cicadelles, punaises — dommages mécaniques.",
                     "severity": "Modérée", "treatment": "Insecticides"},
    "leaf_folder" : {"color": "#00b4d8", "icon": "🔵", "fr": "Enrouleur",
                     "desc": "Cnaphalocrosis medinalis — feuilles enroulées.",
                     "severity": "Modérée", "treatment": "Insecticides + parasitoïdes"},
    "scald"       : {"color": "#f77f00", "icon": "🟠", "fr": "Brûlure",
                     "desc": "Helminthosporium oryzae — taches brunes.",
                     "severity": "Faible-Modérée", "treatment": "Fongicides cuivriques"},
    "stripes"     : {"color": "#9b5de5", "icon": "🟣", "fr": "Rayures Virales",
                     "desc": "Rice Stripe Virus transmis par cicadelles.",
                     "severity": "Élevée", "treatment": "Contrôle vecteur"},
    "tungro"      : {"color": "#e63946", "icon": "🔺", "fr": "Tungro",
                     "desc": "RTBV+RTSV — jaunissement, nanisme.",
                     "severity": "Très élevée", "treatment": "Variétés résistantes"},
}

CLASSES = sorted(CLASS_INFO.keys())
PALETTE = [CLASS_INFO[c]["color"] for c in CLASSES]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
def build_efficientnet_b4(num_classes):
    model = models.efficientnet_b4(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_f, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    ckpt_path  = TL_DIR / 'efficientnet_b4_checkpoint.pth'
    pth_path   = TL_DIR / 'efficientnet_b4_best.pth'
    stats_path = TL_DIR / 'dataset_stats.json'

    if not stats_path.exists():
        return None, None, None, None

    with open(stats_path) as f:
        stats = json.load(f)

    nc = stats['num_classes']
    model = build_efficientnet_b4(nc)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        mean = ckpt.get('mean', IMAGENET_MEAN)
        std  = ckpt.get('std',  IMAGENET_STD)
    elif pth_path.exists():
        model.load_state_dict(torch.load(pth_path, map_location='cpu'))
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        return None, None, None, None

    model.eval()
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return model, transform, stats['classes'], {'mean': mean, 'std': std}

@st.cache_data
def load_results():
    p = TL_DIR / 'training_results.json'
    return json.load(open(p)) if p.exists() else None

# ══════════════════════════════════════════════════════════════════════════════
# INFÉRENCE
# ══════════════════════════════════════════════════════════════════════════════
def predict(model, transform, image: Image.Image, classes: list):
    inp = transform(image.convert('RGB')).unsqueeze(0)
    t0  = time.perf_counter()
    with torch.no_grad():
        out   = model(inp)
        probs = torch.softmax(out, 1)[0].numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    pred_idx   = probs.argmax()
    return classes[pred_idx], float(probs[pred_idx]), {c: float(p) for c,p in zip(classes, probs)}, elapsed_ms

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
def make_proba_chart(probs_dict, pred_class):
    cls_sorted = sorted(probs_dict, key=lambda x: probs_dict[x], reverse=True)
    colors = [CLASS_INFO[c]["color"] for c in cls_sorted]
    vals   = [probs_dict[c]*100 for c in cls_sorted]
    labels = [f'{CLASS_INFO[c]["icon"]} {c}' for c in cls_sorted]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker=dict(color=colors, opacity=0.9),
        text=[f'{v:.1f}%' for v in vals], textposition='outside',
        textfont=dict(color='rgba(255,255,255,0.9)', size=11),
        hovertemplate='<b>%{y}</b><br>%{x:.2f}%<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,32,48,0.5)',
        height=320, margin=dict(l=10,r=80,t=10,b=10),
        xaxis=dict(range=[0,max(vals)*1.3], ticksuffix='%', color='#90b0cc',
                   gridcolor='rgba(26,64,96,0.4)'),
        yaxis=dict(color='#90b0cc', autorange='reversed'),
        font=dict(color='#e0f0ff'),
    )
    return fig

def make_gauge_tl(confidence, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence*100,
        number={'suffix':'%','font':{'size':32,'color':color}},
        gauge={
            'axis': {'range':[0,100],'tickcolor':'#90b0cc'},
            'bar': {'color':color},
            'bgcolor': '#0f2030',
            'bordercolor': '#1a4060',
            'steps': [
                {'range':[0,50],'color':'#0a0e1a'},
                {'range':[50,75],'color':'#0f1826'},
                {'range':[75,100],'color':'#152535'},
            ],
            'threshold': {'line':{'color':'#ff6b6b','width':3},'thickness':0.8,'value':90}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=200, margin=dict(l=20,r=20,t=20,b=20))
    return fig

def training_curves_tl(results):
    h_e = results['efficientnet_b4']['history']
    h_r = results['resnet50']['history']
    ep_e = list(range(1, len(h_e['train_acc'])+1))
    ep_r = list(range(1, len(h_r['train_acc'])+1))

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=['EfficientNet-B4 Loss','EfficientNet-B4 Accuracy',
                        'Comparaison Loss (Test)','Comparaison Accuracy (Test)'],
        vertical_spacing=0.12, horizontal_spacing=0.08)

    fig.add_trace(go.Scatter(x=ep_e,y=h_e['train_loss'],name='Train',line=dict(color='#00b4d8',width=2)),1,1)
    fig.add_trace(go.Scatter(x=ep_e,y=h_e['test_loss'], name='Test', line=dict(color='#ff6b6b',width=2)),1,1)
    fig.add_trace(go.Scatter(x=ep_e,y=h_e['train_acc'],name='Train',line=dict(color='#00b4d8',width=2),showlegend=False),1,2)
    fig.add_trace(go.Scatter(x=ep_e,y=h_e['test_acc'], name='Test', line=dict(color='#ff6b6b',width=2),showlegend=False),1,2)
    fig.add_trace(go.Scatter(x=ep_e,y=h_e['test_loss'],name=f"EfficientNet-B4 ({results['efficientnet_b4']['best_acc']:.1f}%)",
                              line=dict(color='#00b4d8',width=2.5)),2,1)
    fig.add_trace(go.Scatter(x=ep_r,y=h_r['test_loss'],name=f"ResNet-50 ({results['resnet50']['best_acc']:.1f}%)",
                              line=dict(color='#ffd93d',width=2.5,dash='dash')),2,1)
    fig.add_trace(go.Scatter(x=ep_e,y=h_e['test_acc'],name='EfficientNet-B4',
                              line=dict(color='#00b4d8',width=2.5),showlegend=False),2,2)
    fig.add_trace(go.Scatter(x=ep_r,y=h_r['test_acc'],name='ResNet-50',
                              line=dict(color='#ffd93d',width=2.5,dash='dash'),showlegend=False),2,2)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,32,48,0.5)',
        height=500, font=dict(color='#90b0cc'),
        legend=dict(bgcolor='rgba(15,32,48,0.8)',bordercolor='#1a4060'),
        margin=dict(l=40,r=40,t=60,b=40),
    )
    fig.update_xaxes(gridcolor='rgba(26,64,96,0.3)', color='#90b0cc')
    fig.update_yaxes(gridcolor='rgba(26,64,96,0.3)', color='#90b0cc')
    for ann in fig.layout.annotations:
        ann.font.color = '#e0f0ff'
    return fig

def dataset_dist_chart_tl():
    counts_tr = {c: len(list((DATA/'train'/c).glob('*.jpeg'))) for c in CLASSES if (DATA/'train'/c).exists()}
    counts_te = {c: len(list((DATA/'test'/c).glob('*.jpeg')))  for c in CLASSES if (DATA/'test'/c).exists()}
    fig = make_subplots(rows=1,cols=2,subplot_titles=['Train','Test'])
    for col_i, (sp, cnt) in enumerate([('Train',counts_tr),('Test',counts_te)],1):
        fig.add_trace(go.Bar(
            x=list(cnt.keys()), y=list(cnt.values()),
            marker_color=[CLASS_INFO[c]['color'] for c in cnt],
            text=list(cnt.values()), textposition='outside',
            textfont=dict(color='#e0f0ff', size=10),
            name=sp,
            hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>',
        ), 1, col_i)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,32,48,0.5)',
        height=380, showlegend=False, font=dict(color='#90b0cc'),
        margin=dict(l=20,r=20,t=50,b=60),
    )
    fig.update_xaxes(tickangle=35, color='#90b0cc')
    fig.update_yaxes(gridcolor='rgba(26,64,96,0.3)', color='#90b0cc')
    for ann in fig.layout.annotations:
        ann.font.color = '#e0f0ff'
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-size: 3rem;'>🍃</div>
        <h2 style='color: #00b4d8; margin: 0;'>EfficientNet-B4</h2>
        <p style='color: #90b0cc; font-size: 0.8rem; margin: 4px 0;'>Transfer Learning</p>
        <div class='tl-badge'>Fine-tuned on Rice Leaf</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.selectbox("Navigation", [
        "🔍 Prédiction",
        "📊 Dashboard Entraînement",
        "🗂️ Explorer le Dataset",
        "🔬 Fine-tuning Expliqué",
        "⚖️ Comparaison DL vs TL",
        "📦 Export & Déploiement",
        "ℹ️ À Propos",
    ])

    st.divider()

    model, transform, classes_model, norm_stats = load_model()
    results = load_results()

    if model is not None:
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.markdown(f"""
        <div class='info-box'>
            <b>✅ EfficientNet-B4 chargé</b><br>
            {total_p/1e6:.1f}M params total<br>
            {train_p/1e6:.1f}M trainables<br>
            Input: {IMG_SIZE}×{IMG_SIZE} · 7 classes
        </div>
        """, unsafe_allow_html=True)
        if results:
            best = results['efficientnet_b4']['best_acc']
            ep   = results['efficientnet_b4']['epochs_run']
            st.markdown(f"""
            <div class='success-box'>
                <b>🏆 Best Accuracy</b><br>
                {best:.2f}% (test set)<br>
                {ep} epochs · EarlyStopping
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warn-box'>
            ⚠️ Modèle non trouvé.<br>
            Exécutez <code>02_training_compare.ipynb</code>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption("Transfer Learning • EfficientNet-B4")
    st.caption("Rice Leaf Disease • Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "Prédiction" in page:
    st.markdown("# 🔍 Classification — EfficientNet-B4")
    st.markdown("""
    <p style='color:#90b0cc'>Modèle EfficientNet-B4 <b>fine-tuné</b> depuis ImageNet sur 7 maladies de feuilles de riz.</p>
    <span class='tl-badge'>Transfer Learning</span>
    <span class='tl-badge'>ImageNet → Rice Leaf</span>
    <span class='tl-badge'>380×380px</span>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📤 Source d'Image")
        input_mode = st.radio("Mode", ["Upload", "Exemple dataset", "Batch mode"],
                               horizontal=True)
        uploaded = None
        if input_mode == "Upload":
            uploaded = st.file_uploader("Déposer une image", type=['jpg','jpeg','png'],
                                         label_visibility='collapsed')
        elif input_mode == "Exemple dataset":
            sel_class = st.selectbox("Classe", CLASSES)
            cls_path = DATA / 'test' / sel_class
            if cls_path.exists():
                imgs = sorted(cls_path.glob('*.jpeg'))
                if imgs:
                    idx = st.slider("Image n°", 0, min(len(imgs)-1, 50), 0)
                    uploaded = imgs[idx]
        else:
            st.info("Mode batch : uploadez plusieurs images.")
            files = st.file_uploader("Images multiples", type=['jpg','jpeg','png'],
                                      accept_multiple_files=True)
            if files and model is not None:
                st.markdown("### Résultats Batch")
                batch_results = []
                for f in files[:10]:
                    img = Image.open(f).convert('RGB')
                    pred_class, conf, probs, ms = predict(model, transform, img, classes_model)
                    batch_results.append({'Fichier': f.name, 'Prédiction': pred_class,
                                          'Confiance': f'{conf*100:.1f}%', 'Temps': f'{ms:.0f}ms'})
                st.dataframe(pd.DataFrame(batch_results), use_container_width=True, hide_index=True)

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert('RGB') if not isinstance(uploaded, Path) else Image.open(uploaded).convert('RGB')
        with col_left:
            st.image(pil_img, caption=f"{pil_img.size[0]}×{pil_img.size[1]}px → redim. {IMG_SIZE}×{IMG_SIZE}",
                     use_container_width=True)

        if model is None:
            st.error("Modèle non disponible.")
        else:
            with st.spinner("Inférence EfficientNet-B4..."):
                pred_class, confidence, probs_dict, elapsed_ms = predict(
                    model, transform, pil_img, classes_model)

            info = CLASS_INFO[pred_class]

            with col_right:
                st.markdown(f"""
                <div class='pred-result'>
                    <div style='font-size:2.5rem'>{info['icon']}</div>
                    <div class='pred-class'>{pred_class.upper()}</div>
                    <div style='color:#90b0cc; font-size:0.9rem; margin:4px 0'>{info['fr']}</div>
                    <div class='pred-conf'>Confiance : {confidence*100:.1f}%</div>
                    <div style='color:#90b0cc; font-size:0.8rem; margin-top:8px'>
                        ⚡ {elapsed_ms:.1f}ms · EfficientNet-B4 · {IMG_SIZE}×{IMG_SIZE}px
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(make_gauge_tl(confidence, info['color']),
                                 use_container_width=True, config={'displayModeBar': False})

                if confidence < 0.6:
                    st.warning("⚠️ Confiance faible")
                elif confidence < 0.8:
                    st.info("ℹ️ Confiance modérée")
                else:
                    st.success("✅ Haute confiance")

            st.markdown("### 📊 Probabilités par Classe")
            st.plotly_chart(make_proba_chart(probs_dict, pred_class),
                             use_container_width=True, config={'displayModeBar': False})

            st.markdown("### 🔬 Informations Cliniques")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>Sévérité</div>
                    <div class='metric-value' style='font-size:1.2rem; color:{info["color"]}'>{info['severity']}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>Inférence</div>
                    <div class='metric-value' style='font-size:1.2rem'>{elapsed_ms:.0f} ms</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>Rang</div>
                    <div class='metric-value' style='font-size:1.2rem; color:#00e5a0'>#{sorted(probs_dict.values(),reverse=True).index(confidence)+1}/7</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class='info-box'>
                <b>{info['icon']} Description :</b> {info['desc']}<br><br>
                <b>💊 Traitement :</b> {info['treatment']}
            </div>""", unsafe_allow_html=True)

            top3 = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            st.markdown("### 🥇 Top-3")
            cols = st.columns(3)
            for i, (cn, prob) in enumerate(top3):
                ic = CLASS_INFO[cn]
                medal = ["🥇","🥈","🥉"][i]
                with cols[i]:
                    st.markdown(f"""<div class='metric-card'>
                        <div style='font-size:1.5rem'>{medal}</div>
                        <div style='color:{ic["color"]}; font-weight:700'>{ic['icon']} {cn}</div>
                        <div class='metric-value'>{prob*100:.1f}%</div>
                        <div class='metric-label'>{ic['fr']}</div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("# 📊 Dashboard Entraînement — Transfer Learning")

    if results is None:
        st.warning("Résultats non disponibles. Exécutez `02_training_compare.ipynb`.")
    else:
        r_e = results['efficientnet_b4']
        r_r = results['resnet50']

        kpis = [
            ("Best Acc. (Test)", f"{r_e['best_acc']:.2f}%",  f"+{r_e['best_acc']-r_r['best_acc']:.1f}% vs ResNet-50", "#00b4d8"),
            ("Params Total",     f"{r_e['params_total']/1e6:.1f}M", "EfficientNet-B4", "#00e5a0"),
            ("Params Trainés",   f"{r_e['params_trainable']/1e6:.1f}M", f"({r_e['params_trainable']/r_e['params_total']*100:.0f}% backbone libre)", "#ffd93d"),
            ("Epochs",           str(r_e['epochs_run']), "EarlyStopping", "#ff6b6b"),
        ]
        cols = st.columns(4)
        for col, (label, value, delta, color) in zip(cols, kpis):
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:{color}'>{value}</div>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-delta'>{delta}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Courbes d'Entraînement")
        st.plotly_chart(training_curves_tl(results), use_container_width=True)

        st.markdown("### ⚖️ Comparaison EfficientNet-B4 vs ResNet-50")
        df_cmp = pd.DataFrame({
            'Modèle': ['EfficientNet-B4', 'ResNet-50'],
            'Params Total': [f"{r_e['params_total']/1e6:.1f}M", f"{r_r['params_total']/1e6:.1f}M"],
            'Params Trainables': [f"{r_e['params_trainable']/1e6:.1f}M", f"{r_r['params_trainable']/1e6:.1f}M"],
            'Best Acc (%)': [f"{r_e['best_acc']:.2f}", f"{r_r['best_acc']:.2f}"],
            'Epochs': [r_e['epochs_run'], r_r['epochs_run']],
            'Input': ['380×380','224×224'],
        })
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        if 'per_class_acc' in results:
            st.markdown("### 🎯 Accuracy par Classe — EfficientNet-B4")
            pca = results['per_class_acc']
            fig = go.Figure(go.Bar(
                x=CLASSES, y=pca,
                marker_color=[CLASS_INFO[c]['color'] for c in CLASSES],
                text=[f'{v:.1f}%' for v in pca], textposition='outside',
                textfont=dict(color='#e0f0ff'),
            ))
            fig.add_hline(y=np.mean(pca), line_dash='dash', line_color='#ff6b6b',
                          annotation_text=f'Moyenne: {np.mean(pca):.1f}%',
                          annotation_font_color='#ff6b6b')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,32,48,0.5)',
                height=350, font=dict(color='#90b0cc'),
                yaxis=dict(range=[0,max(pca)+10], gridcolor='rgba(26,64,96,0.3)'),
            )
            st.plotly_chart(fig, use_container_width=True)

        if 'report' in results:
            st.markdown("### 📋 Rapport de Classification")
            with st.expander("Rapport détaillé"):
                st.code(results['report'], language='text')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("# 🗂️ Explorer le Dataset")

    col1,col2,col3,col4 = st.columns(4)
    for col,(label,value,color) in zip([col1,col2,col3,col4],[
        ("Classes","7","#00b4d8"),("Train","12,983","#00e5a0"),
        ("Test","2,799","#ffd93d"),("Input TL","380×380","#ff6b6b")]):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:{color}'>{value}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Distribution")
    if DATA.exists():
        st.plotly_chart(dataset_dist_chart_tl(), use_container_width=True)

    st.markdown("### 🖼️ Galerie")
    sel_cls = st.selectbox("Classe", CLASSES)
    split   = st.select_slider("Split", ['train','test'], 'train')
    n_show  = st.slider("N images", 5, 20, 10)
    cls_path = DATA/split/sel_cls
    if cls_path.exists():
        imgs = sorted(cls_path.glob('*.jpeg'))[:n_show]
        info = CLASS_INFO[sel_cls]
        st.markdown(f"""<div class='info-box'>
            {info['icon']} <b>{sel_cls}</b> · {info['fr']} · {info['severity']}
        </div>""", unsafe_allow_html=True)
        cols = st.columns(5)
        for i,p in enumerate(imgs):
            with cols[i%5]:
                st.image(Image.open(p).resize((150,150)), caption=p.name[:10], use_container_width=True)

    st.markdown("### 🔍 Normalisation ImageNet")
    st.markdown("""
    <div class='info-box'>
        <b>Pourquoi ImageNet ?</b><br>
        EfficientNet-B4 a été entraîné sur ImageNet avec ces valeurs spécifiques.
        Les premières couches sont calibrées pour cette distribution.
        Utiliser d'autres stats perturberait les activations et dégraderait les features pré-appris.
    </div>
    """, unsafe_allow_html=True)
    for label, vals in [("Mean (RGB)", IMAGENET_MEAN), ("Std (RGB)", IMAGENET_STD)]:
        st.markdown(f"**{label} :**")
        for c, v in zip(['R','G','B'], vals):
            st.progress(v, text=f"Canal {c}: {v:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : FINE-TUNING EXPLIQUÉ
# ══════════════════════════════════════════════════════════════════════════════
elif "Fine-tuning" in page:
    st.markdown("# 🔬 Fine-tuning Expliqué")

    st.markdown("""
    ## Stratégie de Fine-tuning EfficientNet-B4

    Le transfer learning repose sur un principe fondamental : **les réseaux de neurones apprennent
    des représentations hiérarchiques** qui se généralisent entre domaines visuels similaires.
    """)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("""
        ### 📚 Architecture EfficientNet-B4
        ```
        ImageNet Features (gelées 70%)
        ├── MBConv1 → features bas niveau
        │   (contours, textures, couleurs)
        ├── MBConv6 × 2 → patterns
        ├── MBConv6 × 2 (stride 2)
        ├── MBConv6 × 3
        ├── MBConv6 × 3 (stride 2)
        ├── MBConv6 × 4
        ├── MBConv6 × 1 (stride 2)
        └── Conv1x1, Pooling

        Fine-tuned Head (entraîné):
        ├── Dropout(0.4)
        ├── Linear(1792 → 512)
        ├── ReLU
        ├── Dropout(0.2)
        └── Linear(512 → 7 classes)
        ```

        ### 🎯 Compound Scaling (EfficientNet)
        EfficientNet utilise un **compound scaling factor** qui
        optimise simultanément la profondeur, la largeur et la
        résolution du réseau — d'où `B4` = scaling factor 4.
        """)

    with col2:
        st.markdown("""
        ### ⚙️ Pourquoi ces choix ?

        **1. Backbone partiellement gelé (70%)**
        > Les couches précoces apprennent des features universels
        > (Gabor-like filters, color blobs). Les geler préserve
        > ces connaissances précieuses et réduit le risque de
        > *catastrophic forgetting*.

        **2. Learning Rate Différentiel**
        ```
        Head      :  lr = 1e-3  (apprentissage rapide)
        Backbone  :  lr = 1e-5  (fine-tuning très doux)
        ```
        > Le backbone est déjà bien optimisé — un grand LR
        > le perturberait sans bénéfice.

        **3. AdamW vs Adam**
        > AdamW sépare le weight decay de la mise à jour
        > du gradient — meilleure régularisation, convergence
        > plus stable lors du fine-tuning.

        **4. Augmentation agressive**
        > Le backbone est robuste aux transformations.
        > `RandomPerspective` simule des prises de vue variées.
        > `AutoAugment` (politique ImageNet) optimise automatiquement.
        """)

    st.markdown("### 📊 Progression du Fine-tuning")
    st.markdown("""
    | Phase | Couches actives | LR Head | LR Backbone | Objectif |
    |-------|----------------|---------|-------------|---------|
    | Epoch 1-5 | HEAD seulement | 1e-3 | 1e-5 | Adapter le classificateur |
    | Epoch 6-15 | HEAD + backbone top 30% | 1e-3 | 1e-5 | Spécialiser les features |
    | Epoch 16+ | EarlyStopping | Cosine decay | Cosine decay | Convergence |
    """)

    st.markdown("### 🧠 Ce qu'apprend EfficientNet à chaque niveau")
    levels = {
        "Couches 1-3 (gelées)": "Contours, orientations, couleurs primaires — identiques ImageNet et riz",
        "Couches 4-6 (gelées)": "Textures, patterns locaux — partiellement transférables",
        "Couches 7-9 (fine-tuned)": "Patterns de maladies, décolorations — spécifiques au riz",
        "HEAD (entraîné from scratch)": "Combinaison des features → décision 7 classes",
    }
    for level, desc in levels.items():
        st.markdown(f"""<div class='info-box'>
            <b>{level}</b><br>{desc}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : COMPARAISON DL VS TL
# ══════════════════════════════════════════════════════════════════════════════
elif "Comparaison" in page:
    st.markdown("# ⚖️ Comparaison Deep Learning vs Transfer Learning")

    dl_results_path = BASE / 'DEEP LEARNING' / 'training_results.json'
    tl_results_path = TL_DIR / 'training_results.json'

    dl_avail = dl_results_path.exists()
    tl_avail = tl_results_path.exists()

    if not (dl_avail and tl_avail):
        st.warning("Exécutez les deux notebooks d'entraînement pour voir la comparaison complète.")
        available = []
        if dl_avail: available.append("✅ DL")
        if tl_avail: available.append("✅ TL")
        if not dl_avail: available.append("❌ DL (notebook 02 DEEP LEARNING)")
        if not tl_avail: available.append("❌ TL (notebook 02 TRANSFERT LEARNING)")
        st.info("Disponible : " + " · ".join(available))

    # Tableau comparatif statique
    st.markdown("### 📊 Comparaison Théorique & Pratique")
    compare_df = pd.DataFrame({
        'Critère': [
            'Architecture', 'Poids initiaux', 'Params entraînés',
            'Epochs nécessaires', 'Accuracy (estim.)', 'Convergence',
            'Robustesse déséquilibre', 'Coût GPU (CPU)',
            'Taille modèle', 'Input size',
        ],
        'Deep Learning (RiceCNN)': [
            'ResNet-style custom', 'Aléatoires (Kaiming)', '~3.2M (100%)',
            '~25-30', 'Variable', 'Lente (~30 epochs)',
            'Moyenne (class_weights)', 'Faible', '~12 MB', '224×224',
        ],
        'Transfer Learning (EfficientNet-B4)': [
            'EfficientNet-B4 ImageNet', 'ImageNet pré-entraîné', '~5.5M (29%)',
            '~15-20', 'Supérieure', 'Rapide (~15 epochs)',
            'Bonne (features robustes)', 'Modéré', '~70 MB', '380×380',
        ],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    if dl_avail and tl_avail:
        with open(dl_results_path) as f: dl_r = json.load(f)
        tl_r = results

        # Accuracy comparaison
        st.markdown("### 🏆 Accuracy Finale — Comparaison Réelle")
        models_names = ['LightCNN\n(DL baseline)', 'RiceCNN\n(DL ResNet)', f"ResNet-50\n(TL)", f"EfficientNet-B4\n(TL)"]
        accs = [dl_r['lightcnn']['best_acc'], dl_r['ricecnn']['best_acc'],
                tl_r['resnet50']['best_acc'], tl_r['efficientnet_b4']['best_acc']]
        colors_cmp = ['#ffd93d','#6c63ff','#f77f00','#00b4d8']

        fig = go.Figure(go.Bar(
            x=models_names, y=accs,
            marker_color=colors_cmp,
            text=[f'{a:.2f}%' for a in accs], textposition='outside',
            textfont=dict(color='#e0f0ff', size=13, family='bold'),
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,32,48,0.5)',
            height=400, yaxis=dict(range=[0,100], gridcolor='rgba(26,64,96,0.3)'),
            font=dict(color='#90b0cc'), title_font_color='#e0f0ff',
            title_text='Comparaison Accuracy Test — Tous Modèles',
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Quand utiliser quoi ?")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='info-box'>
            <b>🏗️ Deep Learning from Scratch — Quand ?</b><br>
            • Dataset très large (>100k images)<br>
            • Domaine très différent d'ImageNet<br>
            • Contrainte de taille de modèle (edge devices)<br>
            • Interprétabilité totale de l'architecture<br>
            • Ressources de calcul suffisantes
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='success-box'>
            <b>🍃 Transfer Learning — Quand ? (notre cas)</b><br>
            • Dataset limité (~13k images ✓)<br>
            • Domaine similaire à ImageNet (images naturelles ✓)<br>
            • Précision maximale requise ✓<br>
            • Convergence rapide souhaitée ✓<br>
            • Infrastructure GPU limitée ✓
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif "Export" in page:
    st.markdown("# 📦 Export & Déploiement EfficientNet-B4")

    formats = [
        ("state_dict",   "efficientnet_b4_best.pth",         "#00b4d8", "PyTorch state dict"),
        ("Checkpoint",   "efficientnet_b4_checkpoint.pth",    "#00e5a0", "Avec classes/mean/std"),
        ("TorchScript",  "efficientnet_b4_torchscript.pt",    "#ffd93d", "C++ / Mobile"),
        ("ONNX",         "efficientnet_b4.onnx",              "#ff6b6b", "Universal runtime"),
    ]
    cols = st.columns(4)
    for col,(fmt,fname,color,desc) in zip(cols, formats):
        fp = TL_DIR/fname; exists = fp.exists()
        size = f"{fp.stat().st_size/1024/1024:.0f} MB" if exists else "—"
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div style='font-size:1.4rem'>{"✅" if exists else "❌"}</div>
                <div style='color:{color}; font-weight:700; margin:4px'>{fmt}</div>
                <div class='metric-label'>{fname}</div>
                <div class='metric-value' style='font-size:1.1rem; color:{color}'>{size}</div>
                <div class='metric-label' style='font-size:0.7rem; margin-top:4px'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 💻 Code de Déploiement")
    st.code("""
# Chargement du modèle EfficientNet-B4 fine-tuné
import torch, torch.nn as nn
import torchvision.transforms as T, torchvision.models as models
from PIL import Image

def load_efficientnet_b4(ckpt_path='efficientnet_b4_checkpoint.pth'):
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    model = models.efficientnet_b4(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(in_f, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, ckpt['num_classes'])
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['classes'], ckpt['mean'], ckpt['std']

# Inférence
model, classes, mean, std = load_efficientnet_b4()
transform = T.Compose([
    T.Resize((380, 380)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)  # ImageNet normalization
])

image = Image.open('rice_leaf.jpg').convert('RGB')
inp   = transform(image).unsqueeze(0)  # [1, 3, 380, 380]

with torch.no_grad():
    logits = model(inp)
    probs  = torch.softmax(logits, dim=1)[0]
    pred   = classes[probs.argmax()]
    conf   = probs.max().item()

print(f"Maladie : {pred} ({conf*100:.1f}% confiance)")
    """, language='python')

    # Benchmark si disponible
    bench_path = TL_DIR / 'benchmark.png'
    if bench_path.exists():
        st.markdown("### ⚡ Benchmark")
        st.image(str(bench_path), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
elif "propos" in page.lower():
    st.markdown("# ℹ️ À Propos — Transfer Learning")

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("""
        ### 🎯 Objectif
        Classifier 7 maladies de feuilles de riz en utilisant
        **EfficientNet-B4 pré-entraîné** sur ImageNet — fine-tuné
        sur le dataset Rice Leaf Disease.

        ### 🌾 Maladies Couvertes
        """)
        for cls, info in CLASS_INFO.items():
            st.markdown(f"""<div class='metric-card' style='text-align:left; display:flex; align-items:center; gap:12px; padding:10px 16px;'>
                <span style='font-size:1.5rem'>{info['icon']}</span>
                <div>
                    <span style='color:{info['color']}; font-weight:700'>{cls}</span>
                    <span style='color:#90b0cc'> — {info['fr']}</span><br>
                    <span style='color:#666; font-size:0.75rem'>Sévérité: {info['severity']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        ### 🛠️ Stack Technique

        | Composant | Technologie |
        |-----------|-------------|
        | Backbone | EfficientNet-B4 (ImageNet) |
        | Framework | PyTorch 2.x + torchvision |
        | Interface | Streamlit |
        | Visualisation | Plotly |
        | Export | ONNX opset 17, TorchScript |
        | Optimizer | AdamW + CosineAnnealingLR |
        | LR Différentiel | Head: 1e-3, Backbone: 1e-5 |

        ### 📋 Pourquoi EfficientNet-B4 ?

        1. **Compound scaling** — optimise depth×width×resolution
        2. **SOTA efficiency** — meilleur accuracy/params ratio
        3. **ImageNet pretrained** — 1.2M images, features robustes
        4. **380×380** — résolution adaptée aux détails de maladies

        ### 📁 Structure
        """)
        st.code("""
TRANSFERT LEARNING/
├── 01_data_mining.ipynb
├── 02_training_compare.ipynb
├── 03_export.ipynb
├── app_transfert_learning.py
├── efficientnet_b4_best.pth
├── efficientnet_b4_checkpoint.pth
├── efficientnet_b4_torchscript.pt
├── efficientnet_b4.onnx
├── dataset_stats.json
├── training_results.json
└── *.png
        """, language='text')

    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#666; font-size:0.8rem; padding:20px'>
        Transfer Learning • EfficientNet-B4 • ImageNet → Rice Leaf Disease
    </div>
    """, unsafe_allow_html=True)
