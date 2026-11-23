"""
Project Title: Cat and Dog Image Classification
"""

import base64
import html
import io
from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

# -------------------------------------------------------------------------
# 1. Page Configuration & Custom Dark Theme Styling  UI
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="DL-003: Real-Time CNN Cat vs. Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Base Dark Theme */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #0E1117 !important;
            color: #FAFAFA !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        }

        [data-testid="stHeader"] {
            background-color: rgba(14, 17, 23, 0.85) !important;
            backdrop-filter: blur(8px);
        }

        .block-container {
            max-width: 1152px !important;
            padding-top: 2.2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #262730 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.10) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }

        .sidebar-title {
            font-size: 1.125rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.015em;
            margin: 0;
            line-height: 1.3;
        }

        .sidebar-subtitle {
            font-size: 0.75rem;
            color: #9CA3AF;
            margin-top: 0.3rem;
            margin-bottom: 0;
        }

        .sidebar-divider {
            border: 0;
            border-top: 1px solid rgba(255, 255, 255, 0.10);
            margin: 1.15rem 0;
        }

        .sidebar-section-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: #FFFFFF;
            margin: 0 0 0.5rem 0;
        }

        .sidebar-desc {
            font-size: 0.75rem;
            color: #D1D5DB;
            line-height: 1.55;
            margin: 0 0 0.75rem 0;
        }

        .sidebar-metric {
            margin-bottom: 0.85rem;
        }

        .sidebar-metric-label {
            font-size: 0.75rem;
            color: #9CA3AF;
            margin: 0;
        }

        .sidebar-metric-arch {
            font-size: 1.2rem;
            font-weight: 600;
            color: #FFFFFF;
            margin: 0.15rem 0 0 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .sidebar-metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.45rem;
            font-weight: 600;
            color: #FFFFFF;
            margin: 0.1rem 0 0 0;
            line-height: 1.25;
        }

        /* Inline Code Badge */
        .st-code {
            font-family: 'JetBrains Mono', monospace;
            background-color: rgba(28, 131, 225, 0.12);
            color: #3DD56D;
            padding: 0.12rem 0.38rem;
            border-radius: 0.35rem;
            font-size: 0.85em;
        }

        /* Banners & Pills */
        .st-info-pill {
            display: inline-flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            background-color: rgba(28, 131, 225, 0.15);
            border: 1px solid rgba(28, 131, 225, 0.30);
            color: #C7E0F4;
            border-radius: 9999px;
            padding: 0.4rem 0.95rem;
            font-size: 0.72rem;
            margin-top: 0.65rem;
        }

        .st-info-box {
            background-color: rgba(28, 131, 225, 0.15);
            border: 1px solid rgba(28, 131, 225, 0.30);
            color: #C7E0F4;
            border-radius: 0.5rem;
            padding: 0.8rem 1rem;
            font-size: 0.82rem;
            line-height: 1.5;
        }

        .st-success-box {
            background-color: rgba(33, 195, 84, 0.15);
            border: 1px solid rgba(33, 195, 84, 0.32);
            color: #A7F3D0;
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            font-weight: 600;
            line-height: 1.5;
        }

        .st-success-compact {
            background-color: rgba(33, 195, 84, 0.15);
            border: 1px solid rgba(33, 195, 84, 0.32);
            color: #A7F3D0;
            border-radius: 0.5rem;
            padding: 0.65rem 0.75rem;
            font-size: 0.75rem;
            font-weight: 500;
            line-height: 1.5;
            margin-top: 0.65rem;
        }

        .st-error-box {
            background-color: rgba(239, 68, 68, 0.14);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #FECACA;
            border-radius: 0.5rem;
            padding: 0.85rem 1rem;
            font-size: 0.84rem;
            line-height: 1.55;
        }

        /* Main Header */
        .hero-header {
            text-align: center;
            padding-bottom: 1.75rem;
            margin-bottom: 1.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.10);
        }

        .hero-title {
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: #FFFFFF;
            margin: 0;
            line-height: 1.15;
        }

       .hero-subtitle,
        [data-testid="stMarkdownContainer"] p.hero-subtitle {
            font-size: 0.875rem;
            color: #D1D5DB;
            line-height: 1.6;
            max-width: 42rem;
            width: 100%;
            text-align: center !important;
            margin: 0.75rem auto 0.25rem auto !important;
        }

        /* Section Headings with Icon Badges */
        .section-heading {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 1.2rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.85rem;
        }

        .icon-badge-blue {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.85rem;
            height: 1.85rem;
            border-radius: 0.5rem;
            background-color: rgba(28, 131, 225, 0.15);
            border: 1px solid rgba(28, 131, 225, 0.30);
            color: #90CAF9;
            flex-shrink: 0;
        }

        .icon-badge-emerald {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.85rem;
            height: 1.85rem;
            border-radius: 0.5rem;
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.30);
            color: #34D399;
            flex-shrink: 0;
        }

        .upload-label {
            font-size: 0.82rem;
            color: #D1D5DB;
            margin-bottom: 0.45rem;
        }

        /* Streamlit File Uploader Restyle */
        [data-testid="stFileUploaderDropzone"] {
            background-color: #262730 !important;
            border: 1px dashed rgba(250, 250, 250, 0.20) !important;
            border-radius: 0.5rem !important;
            color: #FAFAFA !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(250, 250, 250, 0.40) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background-color: #0E1117 !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            color: #FFFFFF !important;
            border-radius: 0.375rem !important;
            font-size: 0.75rem !important;
        }

        /* Image Preview Frame */
        .preview-frame {
            border-radius: 0.5rem;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.10);
            background-color: #000000;
            margin-top: 0.85rem;
            text-align: center;
        }

        .preview-frame img {
            max-height: 380px;
            width: 100%;
            object-fit: contain;
            display: block;
            margin: 0 auto;
        }

        .preview-caption {
            font-size: 0.75rem;
            color: #9CA3AF;
            text-align: center;
            margin-top: 0.45rem;
        }

        /* 3-Column Metrics Row */
        .metrics-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1.1rem 0 1.25rem 0;
        }

        .metric-label {
            font-size: 0.7rem;
            color: #9CA3AF;
            margin: 0;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.125rem;
            font-weight: 600;
            color: #FFFFFF;
            margin: 0.2rem 0 0 0;
        }

        /* Probability Bars */
        .prob-section-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: #FFFFFF;
            margin: 0 0 0.75rem 0;
        }

        .prob-item {
            margin-bottom: 0.8rem;
        }

        .prob-text {
            font-size: 0.78rem;
            color: #E5E7EB;
            margin: 0 0 0.38rem 0;
        }

        .prob-mono {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: #FFFFFF;
        }

        .prob-track {
            height: 8px;
            width: 100%;
            border-radius: 9999px;
            background-color: #262730;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            background-color: #1C83E1;
            border-radius: 9999px;
        }

        .tensor-caption {
            font-size: 0.75rem;
            color: #9CA3AF;
            line-height: 1.55;
            margin-top: 0.85rem;
        }

        /* Feature Maps Section */
        .main-divider {
            border: 0;
            border-top: 1px solid rgba(255, 255, 255, 0.10);
            margin: 2rem 0;
        }

        .fmap-heading {
            font-size: 1.25rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 0;
        }

        .fmap-sub {
            font-size: 0.84rem;
            color: #9CA3AF;
            margin: 0.35rem 0 1rem 0;
        }

        .fmap-white-card {
            background-color: #FFFFFF;
            border-radius: 0.75rem;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.35);
        }

        .fmap-white-card img {
            width: 100%;
            height: auto;
            display: block;
            border-radius: 0.35rem;
        }

        /* Expander & Architecture Table */
        [data-testid="stExpander"] {
            background-color: #0E1117 !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 0.5rem !important;
        }

        [data-testid="stExpander"] summary {
            color: #FFFFFF !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
        }

        .arch-table-wrap {
            overflow-x: auto;
            width: 100%;
        }

        .arch-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.75rem;
            text-align: left;
        }

        .arch-table th {
            padding: 0.55rem 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.15);
            color: #9CA3AF;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        }

        .arch-table td {
            padding: 0.55rem 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-family: 'JetBrains Mono', monospace;
        }

        .arch-table tr:hover td {
            background-color: rgba(255, 255, 255, 0.03);
        }

        .col-idx { color: #9CA3AF; }
        .col-name { color: #3DD56D; }
        .col-type { color: #FFFFFF; }
        .col-shape { color: #90CAF9; }
        .col-params { color: #D1D5DB; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
CANDIDATE_MODEL_PATHS = [
    APP_DIR / "cat_dog_classifier.keras",
    Path("cat_dog_classifier.keras"),
    APP_DIR / "dl003_cats_dogs_workspace" / "outputs" / "cat_dog_classifier.keras",
]

# ImageNet-1000 class index ranges for strict Domestic Cat & Domestic Dog validation
IMAGENET_CAT_INDICES = set(range(281, 286))
IMAGENET_DOG_INDICES = set(range(151, 269))
IMAGENET_WILD_CARNIVORE_INDICES = set(range(269, 281)) | set(range(286, 300))

MIN_IMAGE_DIMENSION = 32
MIN_PIXEL_STD = 5.0
MIN_PET_GROUP_PROB = 0.20
MIN_FALLBACK_PET_PROB = 0.30


# -------------------------------------------------------------------------
# 2. Model Loaders & Pre-Inference OOD Gatekeeper
# -------------------------------------------------------------------------
@st.cache_resource
def load_keras_model_from_path(model_path_str: str) -> tf.keras.Model:
    """Loads the trained .keras/.h5 model from disk (compile=False for fast inference)."""
    return tf.keras.models.load_model(model_path_str, compile=False)


@st.cache_resource
def load_ood_gatekeeper_model() -> tf.keras.Model:
    """
    Loads a cached ImageNet-1000 MobileNetV2 gatekeeper used strictly to reject
    out-of-distribution (non-Cat / non-Dog) images before binary CNN inference.
    """
    return tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)


def find_repo_model() -> Path | None:
    """Checks if cat_dog_classifier.keras is present in the repository."""
    for p in CANDIDATE_MODEL_PATHS:
        if p.exists() and p.stat().st_size > 1024:
            return p
    return None


def pil_to_base64_png(pil_img: Image.Image) -> str:
    """Encodes a PIL image into a base64 PNG data URI for crisp dark-framed rendering."""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def fig_to_base64_png(fig: plt.Figure) -> str:
    """Renders a Matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def validate_cat_or_dog_image(pil_img: Image.Image) -> dict:
    """
    Validates whether an uploaded PIL image is a genuine domestic Cat or Dog photo.
    Rejects low-resolution/blank files, wild animals, humans, vehicles, and objects.
    """
    width, height = pil_img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return {
            "is_valid": False,
            "reason": (
                f"Image resolution ({width}×{height} px) is below the minimum "
                f"required size ({MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} px)."
            ),
            "detected_label": "Low-Resolution Image",
            "detected_conf": 0.0,
            "cat_group_prob": 0.0,
            "dog_group_prob": 0.0,
            "pet_total_prob": 0.0,
        }

    img_rgb = pil_img.convert("RGB")
    raw_arr = np.array(img_rgb, dtype="float32")
    if float(np.std(raw_arr)) < MIN_PIXEL_STD:
        return {
            "is_valid": False,
            "reason": (
                "Blank or uniform solid-color image detected. "
                "Please upload a clear photo of a Cat or Dog."
            ),
            "detected_label": "Blank Image",
            "detected_conf": 0.0,
            "cat_group_prob": 0.0,
            "dog_group_prob": 0.0,
            "pet_total_prob": 0.0,
        }

    gatekeeper = load_ood_gatekeeper_model()
    img_224 = img_rgb.resize((224, 224))
    arr_224 = np.expand_dims(np.array(img_224, dtype="float32"), axis=0)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(arr_224.copy())

    preds = gatekeeper(preprocessed, training=False).numpy()[0]

    cat_group_prob = float(np.sum(preds[list(IMAGENET_CAT_INDICES)]))
    dog_group_prob = float(np.sum(preds[list(IMAGENET_DOG_INDICES)]))
    wild_group_prob = float(np.sum(preds[list(IMAGENET_WILD_CARNIVORE_INDICES)]))
    pet_total_prob = cat_group_prob + dog_group_prob

    top1_idx = int(np.argmax(preds))
    decoded_top3 = tf.keras.applications.mobilenet_v2.decode_predictions(
        np.expand_dims(preds, axis=0), top=3
    )[0]
    top1_label = decoded_top3[0][1].replace("_", " ").title()
    top1_conf = float(decoded_top3[0][2])

    is_top1_cat_or_dog = (
        top1_idx in IMAGENET_CAT_INDICES or top1_idx in IMAGENET_DOG_INDICES
    )

    if not is_top1_cat_or_dog and wild_group_prob > pet_total_prob:
        return {
            "is_valid": False,
            "reason": (
                f"Detected {top1_label} ({top1_conf * 100:.1f}% confidence). "
                "Only domestic Cat and Dog images are supported."
            ),
            "detected_label": top1_label,
            "detected_conf": top1_conf,
            "cat_group_prob": cat_group_prob,
            "dog_group_prob": dog_group_prob,
            "pet_total_prob": pet_total_prob,
        }

    is_valid_pet = (
        is_top1_cat_or_dog and pet_total_prob >= MIN_PET_GROUP_PROB
    ) or (pet_total_prob >= MIN_FALLBACK_PET_PROB)

    if not is_valid_pet:
        return {
            "is_valid": False,
            "reason": (
                f"Detected {top1_label} ({top1_conf * 100:.1f}% confidence; "
                f"Cat/Dog match: {pet_total_prob * 100:.1f}%). "
                "Please upload a valid Cat or Dog image."
            ),
            "detected_label": top1_label,
            "detected_conf": top1_conf,
            "cat_group_prob": cat_group_prob,
            "dog_group_prob": dog_group_prob,
            "pet_total_prob": pet_total_prob,
        }

    return {
        "is_valid": True,
        "reason": "Valid Cat or Dog image confirmed.",
        "detected_label": top1_label,
        "detected_conf": top1_conf,
        "cat_group_prob": cat_group_prob,
        "dog_group_prob": dog_group_prob,
        "pet_total_prob": pet_total_prob,
    }


# -------------------------------------------------------------------------
# 3. Real-Time CNN Forward Pass & Feature Map Extraction
# -------------------------------------------------------------------------
def run_real_cnn_inference(model: tf.keras.Model, pil_img: Image.Image) -> dict:
    """Runs a live forward pass through the loaded Keras model."""
    input_shape = model.input_shape
    target_h = input_shape[1] if input_shape[1] is not None else 160
    target_w = input_shape[2] if input_shape[2] is not None else 160

    img_rgb = pil_img.convert("RGB").resize((target_w, target_h))
    input_tensor = np.expand_dims(np.array(img_rgb, dtype="float32"), axis=0)

    t0 = time.perf_counter()
    raw_output = model(input_tensor, training=False).numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    sigmoid_prob = float(raw_output.flatten()[0])
    dog_prob = sigmoid_prob
    cat_prob = 1.0 - sigmoid_prob

    pred_idx = 1 if dog_prob >= 0.5 else 0
    confidence = dog_prob if pred_idx == 1 else cat_prob
    pred_display = (
        "dogs (Class 1: Dog 🐶)" if pred_idx == 1 else "cats (Class 0: Cat 🐱)"
    )

    return {
        "resized_image": img_rgb,
        "input_tensor": input_tensor,
        "tensor_shape": input_tensor.shape,
        "tensor_min": float(input_tensor.min()),
        "tensor_max": float(input_tensor.max()),
        "raw_sigmoid": sigmoid_prob,
        "cat_prob": cat_prob,
        "dog_prob": dog_prob,
        "pred_idx": pred_idx,
        "pred_display": pred_display,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }


def plot_live_conv_feature_maps(model: tf.keras.Model, input_tensor: np.ndarray):
    """Extracts and plots intermediate Conv2D feature maps on a crisp white canvas."""
    conv_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if not conv_layers:
        return None

    layer_outputs = [layer.output for layer in conv_layers[:4]]
    activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
    activations = activation_model(input_tensor, training=False)
    if not isinstance(activations, list):
        activations = [activations]

    num_blocks = len(activations)
    channels_to_show = 4
    fig, axes = plt.subplots(
        num_blocks,
        channels_to_show,
        figsize=(11.5, 2.35 * num_blocks),
        dpi=150,
        facecolor="white",
    )
    if num_blocks == 1:
        axes = np.expand_dims(axes, axis=0)

    for block_idx, fmap in enumerate(activations):
        fmap_np = fmap.numpy()[0]
        layer_name = conv_layers[block_idx].name
        h_dim, w_dim, c_dim = fmap_np.shape
        for ch_idx in range(channels_to_show):
            ax = axes[block_idx, ch_idx]
            ax.set_facecolor("white")
            if ch_idx < c_dim:
                ax.imshow(fmap_np[:, :, ch_idx], cmap="viridis")
                ax.set_title(
                    f"{layer_name} ({h_dim}×{w_dim}×{c_dim}) | Filter #{ch_idx + 1}",
                    fontsize=8.2,
                    fontweight="medium",
                    color="#1F2937",
                    pad=5,
                )
            ax.axis("off")

    plt.tight_layout(pad=1.0)
    return fig


def get_dynamic_layer_rows(model: tf.keras.Model) -> list[dict]:
    """Reads every layer's real name, type, output shape, and parameter count at runtime."""
    rows = []
    for idx, layer in enumerate(model.layers):
        try:
            out_shape = str(layer.output.shape)
        except Exception:
            out_shape = "Dynamic"
        rows.append(
            {
                "idx": idx,
                "name": layer.name,
                "type": layer.__class__.__name__,
                "shape": out_shape,
                "params": f"{layer.count_params():,}",
            }
        )
    return rows


# -------------------------------------------------------------------------
# 4. Left Sidebar (Invertis University Header, Model Config & Live Specs)
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div>
            <p class="sidebar-title">Invertis University Bareilly</p>
            <p class="sidebar-subtitle">CSED | Project ID: DL-003 | Group: BTech_DL_041</p>
        </div>
        <hr class="sidebar-divider" />
        <p class="sidebar-section-title">Model Configuration</p>
        <p class="sidebar-desc">
            Our pre-trained CNN is active by default. Want to test your own model?
            Upload a <span class="st-code">.keras</span> or <span class="st-code">.h5</span> file below:
        </p>
        """,
        unsafe_allow_html=True,
    )

    repo_model_path = find_repo_model()
    uploaded_model_file = st.file_uploader(
        "Upload custom .keras or .h5 model",
        type=["keras", "h5"],
        label_visibility="collapsed",
    )

    active_model = None
    active_model_filename = "cat_dog_classifier.keras"
    active_model_size_str = "0.00 MB"
    active_model_kind = "Trained CNN"

    if uploaded_model_file is not None:
        suffix = Path(uploaded_model_file.name).suffix or ".keras"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_model_file.getbuffer())
            tmp_path = tmp.name
        try:
            active_model = load_keras_model_from_path(tmp_path)
            active_model_filename = uploaded_model_file.name
            active_model_size_str = f"{uploaded_model_file.size / (1024 * 1024):.2f} MB"
            active_model_kind = "Custom CNN"
        except Exception as exc:
            st.markdown(
                f'<div class="st-error-box">Could not load uploaded model: {html.escape(str(exc))}</div>',
                unsafe_allow_html=True,
            )
    elif repo_model_path is not None:
        try:
            active_model = load_keras_model_from_path(str(repo_model_path))
            active_model_filename = repo_model_path.name
            active_model_size_str = (
                f"{repo_model_path.stat().st_size / (1024 * 1024):.2f} MB"
            )
            active_model_kind = "Trained CNN"
        except Exception as exc:
            st.markdown(
                f'<div class="st-error-box">Could not load repository model: {html.escape(str(exc))}</div>',
                unsafe_allow_html=True,
            )

    if active_model is not None:
        total_params = active_model.count_params()
        trainable_params = sum(
            int(np.prod(w.shape)) for w in active_model.trainable_weights
        )
        non_trainable_params = total_params - trainable_params
        total_layers = len(active_model.layers)

        safe_model_name = html.escape(active_model_filename)
        safe_model_size = html.escape(active_model_size_str)
        safe_arch_name = html.escape(active_model.name)

        st.markdown(
            f"""
            <div class="st-success-compact">
                Active: {html.escape(active_model_kind)} (<span class="st-code">{safe_model_name}</span>, {safe_model_size})
            </div>
            <hr class="sidebar-divider" />
            <p class="sidebar-section-title">Live Model Information</p>
            <div class="sidebar-metric">
                <p class="sidebar-metric-label">Architecture Name</p>
                <p class="sidebar-metric-arch" title="{safe_arch_name}">{safe_arch_name}</p>
            </div>
            <div class="sidebar-metric">
                <p class="sidebar-metric-label">Total Parameters</p>
                <p class="sidebar-metric-val">{total_params:,}</p>
            </div>
            <div class="sidebar-metric">
                <p class="sidebar-metric-label">Trainable Parameters</p>
                <p class="sidebar-metric-val">{trainable_params:,}</p>
            </div>
            <div class="sidebar-metric">
                <p class="sidebar-metric-label">Non-Trainable Parameters</p>
                <p class="sidebar-metric-val">{non_trainable_params:,}</p>
            </div>
            <div class="sidebar-metric">
                <p class="sidebar-metric-label">Total Layers</p>
                <p class="sidebar-metric-val">{total_layers}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="st-error-box">Model file <span class="st-code">cat_dog_classifier.keras</span> not found. Upload one above.</div>',
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------------
# 5. Main Centered Header & Active Engine Pill
# -------------------------------------------------------------------------
if active_model is None:
    st.markdown(
        """
        <div class="hero-header">
            <h1 class="hero-title">Cat and Dog Image Classification</h1>
            <p class="hero-subtitle">
                Classify cat and dog images in real time and inspect live convolutional feature maps powered by
                <span class="st-code">cat_dog_classifier.keras</span>.
            </p>
        </div>
        <div class="st-error-box">
            <strong>Model Not Loaded:</strong> Please place <span class="st-code">cat_dog_classifier.keras</span>
            in the repository root or upload a <span class="st-code">.keras</span> / <span class="st-code">.h5</span>
            model in the left sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

safe_model_name = html.escape(active_model_filename)
safe_model_size = html.escape(active_model_size_str)
safe_arch_name = html.escape(active_model.name)

st.markdown(
    f"""
    <div class="hero-header">
        <h1 class="hero-title">Cat and Dog Image Classification</h1>
        <p class="hero-subtitle">
            Classify cat and dog images in real time and inspect live convolutional feature maps powered by
            <span class="st-code">{safe_model_name}</span>.
        </p>
        <div class="st-info-pill">
            <span style="font-weight:600; color:#FFFFFF;">Active Model:</span>
            <span class="st-code">{safe_model_name}</span>
            <span style="color:#9CA3AF;">({safe_model_size})</span>
            <span style="color:#6B7280;">•</span>
            <span style="color:#E5E7EB;">{safe_arch_name}</span>
            <span style="color:#6B7280;">•</span>
            <span style="color:#D1D5DB;">{total_params:,} parameters</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------
# 6. Two-Column Inference Section (1 : 1)
# -------------------------------------------------------------------------
ICON_IMAGE_PLUS = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"/>'
    '<line x1="16" x2="22" y1="5" y2="5"/><line x1="19" x2="19" y1="2" y2="8"/>'
    '<circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>'
    "</svg>"
)

ICON_ACTIVITY = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>'
    "</svg>"
)

col_left, col_right = st.columns([1, 1], gap="large")
pil_img = None
validation_result = None
result = None

with col_left:
    st.markdown(
        f"""
        <div class="section-heading">
            <span class="icon-badge-blue">{ICON_IMAGE_PLUS}</span>
            <span>Upload Image here</span>
        </div>
        <p class="upload-label">Upload any Animal photo (JPG, PNG, WEBP):</p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_img_file = st.file_uploader(
        "Upload any Animal photo (JPG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_img_file is not None:
        try:
            uploaded_img_file.seek(0)
            with Image.open(uploaded_img_file) as probe_img:
                probe_img.verify()
            uploaded_img_file.seek(0)
            pil_img = Image.open(uploaded_img_file)
            pil_img.load()

            img_b64 = pil_to_base64_png(pil_img)
            w_px, h_px = pil_img.size
            st.markdown(
                f"""
                <div class="preview-frame">
                    <img src="data:image/png;base64,{img_b64}" alt="Uploaded test image" />
                </div>
                <p class="preview-caption">Original Image ({w_px}×{h_px} px)</p>
                """,
                unsafe_allow_html=True,
            )
        except (UnidentifiedImageError, OSError, ValueError) as img_err:
            st.markdown(
                f"""
                <div class="st-error-box" style="margin-top:0.75rem;">
                    <strong>Invalid Image File:</strong> Unable to read
                    <span class="st-code">{html.escape(uploaded_img_file.name)}</span>
                    ({html.escape(str(img_err))}). Please upload a valid JPG, PNG, or WEBP image.
                </div>
                """,
                unsafe_allow_html=True,
            )
            pil_img = None

with col_right:
    st.markdown(
        f"""
        <div class="section-heading">
            <span class="icon-badge-emerald">{ICON_ACTIVITY}</span>
            <span>Live Neural Network Output</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pil_img is not None:
        validation_result = validate_cat_or_dog_image(pil_img)

        if not validation_result["is_valid"]:
            st.markdown(
                f"""
                <div class="st-error-box">
                    <div style="font-weight:700; margin-bottom:0.35rem;">
                        Invalid Image — Not a Cat or Dog
                    </div>
                    <div>{html.escape(validation_result["reason"])}</div>
                </div>
                <p class="tensor-caption">
                    Prediction stopped: this classifier only accepts domestic Cat and Dog photos.
                    Images of other animals, people, vehicles, or objects are rejected.
                </p>
                """,
                unsafe_allow_html=True,
            )
        else:
            result = run_real_cnn_inference(active_model, pil_img)
            conf_pct = result["confidence"] * 100.0
            cat_pct = result["cat_prob"] * 100.0
            dog_pct = result["dog_prob"] * 100.0

            st.markdown(
                f"""
                <div class="st-success-box">
                    Predicted Class: {html.escape(result["pred_display"])}
                </div>

                <div class="metrics-grid-3">
                    <div>
                        <p class="metric-label">Confidence</p>
                        <p class="metric-value">{conf_pct:.2f}%</p>
                    </div>
                    <div>
                        <p class="metric-label">Raw Sigmoid σ(z)</p>
                        <p class="metric-value">{result["raw_sigmoid"]:.6f}</p>
                    </div>
                    <div>
                        <p class="metric-label">Inference Time</p>
                        <p class="metric-value">{result["latency_ms"]:.1f} ms</p>
                    </div>
                </div>

                <p class="prob-section-title">Real-Time Class Probabilities</p>

                <div class="prob-item">
                    <p class="prob-text">
                        🐱 <strong>Cat (Class 0) Probability</strong>
                        <span class="st-code">1 - σ(z)</span> :
                        <span class="prob-mono">{cat_pct:.2f}%</span>
                    </p>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: {cat_pct:.2f}%;"></div>
                    </div>
                </div>

                <div class="prob-item">
                    <p class="prob-text">
                        🐶 <strong>Dog (Class 1) Probability</strong>
                        <span class="st-code">σ(z)</span> :
                        <span class="prob-mono">{dog_pct:.2f}%</span>
                    </p>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: {dog_pct:.2f}%;"></div>
                    </div>
                </div>

                <p class="tensor-caption">
                    Live Tensor Specs: Shape <span class="st-code">{html.escape(str(result["tensor_shape"]))}</span>
                    | Raw Pixel Range <span class="st-code">[{result["tensor_min"]:.1f}, {result["tensor_max"]:.1f}]</span>
                    (normalized inside model via <span class="st-code">Rescaling(1./255)</span>)
                </p>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="st-info-box">
                Upload a Cat or Dog photo on the left to run real-time CNN inference, inspect
                class probabilities, and generate live convolutional feature maps.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------------
# 7. Live Convolutional Feature Maps Section (4x4 White-Card Matplotlib Grid)
# -------------------------------------------------------------------------
st.markdown(
    f"""
    <hr class="main-divider" />
    <h2 class="fmap-heading">Let's Know What the CNN Sees By Live Feature Maps</h2>
    <p class="fmap-sub">
        These heatmaps are generated dynamically at runtime by passing your uploaded image through each
        <span class="st-code">Conv2D</span> layer of <span class="st-code">{safe_model_name}</span>.
    </p>
    """,
    unsafe_allow_html=True,
)

if result is not None:
    fmap_fig = plot_live_conv_feature_maps(active_model, result["input_tensor"])
    if fmap_fig is not None:
        fmap_b64 = fig_to_base64_png(fmap_fig)
        plt.close(fmap_fig)
        st.markdown(
            f"""
            <div class="fmap-white-card">
                <img src="data:image/png;base64,{fmap_b64}" alt="Live Conv2D Feature Maps" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="st-info-box">No <span class="st-code">Conv2D</span> layers found in the active model.</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="st-info-box">
            Upload a valid Cat or Dog image above to generate live 4×4 <span class="st-code">Conv2D</span>
            feature map heatmaps from the active neural network.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------------
# 8. Dynamic Layer Architecture Expander (Read Directly from model.layers)
# -------------------------------------------------------------------------
st.markdown('<hr class="main-divider" />', unsafe_allow_html=True)

with st.expander(
    "🧱 Inspect Live Model Layer Architecture (Read Directly from model.layers)",
    expanded=False,
):
    layer_rows = get_dynamic_layer_rows(active_model)
    table_body_html = "".join(
        f"""
        <tr>
            <td class="col-idx">{row["idx"]}</td>
            <td class="col-name">{html.escape(row["name"])}</td>
            <td class="col-type">{html.escape(row["type"])}</td>
            <td class="col-shape">{html.escape(row["shape"])}</td>
            <td class="col-params">{html.escape(row["params"])}</td>
        </tr>
        """
        for row in layer_rows
    )
    st.markdown(
        f"""
        <div class="arch-table-wrap">
            <table class="arch-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Layer Name</th>
                        <th>Layer Type</th>
                        <th>Output Shape</th>
                        <th>Parameters</th>
                    </tr>
                </thead>
                <tbody>
                    {table_body_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
