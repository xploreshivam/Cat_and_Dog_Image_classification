"""
========================================================================
Project ID   : DL-003
Project Title: Cat and Dog Image Classification
Group ID     : BTech_DL_041 | Invertis University Bareilly (CSED)
========================================================================
"""

from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

# -------------------------------------------------------------------------
# 1. Page Configuration & Minimalist Crisp Styling
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="DL-003 | Cat & Dog Image Classifier",
    page_icon="🐾",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        .project-meta {
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.72;
            margin-bottom: 0.2rem;
        }
        .main-title {
            font-size: 1.95rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            line-height: 1.2;
        }
        .sub-caption {
            font-size: 0.95rem;
            opacity: 0.78;
            margin-bottom: 1.25rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["Cat (Class 0)", "Dog (Class 1)"]
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
    """Loads the trained .keras model from disk (compile=False for fast inference)."""
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
                f"Detected **{top1_label}** ({top1_conf * 100:.1f}% confidence). "
                "Only domestic **Cat** and **Dog** images are supported."
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
                f"Detected **{top1_label}** ({top1_conf * 100:.1f}% confidence; "
                f"Cat/Dog match: {pet_total_prob * 100:.1f}%). "
                "Please upload a valid **Cat** or **Dog** image."
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
        "pred_label": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "latency_ms": latency_ms,
    }


def plot_live_conv_feature_maps(model: tf.keras.Model, input_tensor: np.ndarray):
    """Extracts and plots intermediate Conv2D feature maps for the input image."""
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
        figsize=(11, 2.2 * num_blocks),
        dpi=130,
    )
    if num_blocks == 1:
        axes = np.expand_dims(axes, axis=0)

    for block_idx, fmap in enumerate(activations):
        fmap_np = fmap.numpy()[0]
        layer_name = conv_layers[block_idx].name
        h_dim, w_dim, c_dim = fmap_np.shape
        for ch_idx in range(channels_to_show):
            ax = axes[block_idx, ch_idx]
            ax.imshow(fmap_np[:, :, ch_idx], cmap="viridis")
            ax.set_title(
                f"{layer_name} ({h_dim}×{w_dim}×{c_dim}) · Filter {ch_idx + 1}",
                fontsize=8,
                fontweight="medium",
            )
            ax.axis("off")

    plt.tight_layout()
    return fig


def get_dynamic_layer_table(model: tf.keras.Model) -> list[dict]:
    """Reads every layer's real name, type, output shape, and parameter count at runtime."""
    rows = []
    for idx, layer in enumerate(model.layers, start=1):
        try:
            out_shape = str(layer.output.shape)
        except Exception:
            out_shape = "Dynamic"
        rows.append(
            {
                "Layer #": idx,
                "Layer Name": layer.name,
                "Layer Type": layer.__class__.__name__,
                "Output Shape": out_shape,
                "Parameters": f"{layer.count_params():,}",
            }
        )
    return rows


# -------------------------------------------------------------------------
# 4. Sidebar: Clean Project Metadata & Model Specs
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Invertis University Bareilly")
    st.caption("Department of CSED · Project ID: **DL-003** · Group: **BTech_DL_041**")
    st.divider()

    repo_model_path = find_repo_model()
    with st.expander("Switch / Upload `.keras` Model", expanded=False):
        uploaded_model_file = st.file_uploader(
            "Select a `.keras` or `.h5` file:",
            type=["keras", "h5"],
        )

    active_model = None
    active_model_source = None

    if uploaded_model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(uploaded_model_file.getbuffer())
            tmp_path = tmp.name
        try:
            active_model = load_keras_model_from_path(tmp_path)
            active_model_source = f"Custom ({uploaded_model_file.name})"
        except Exception as exc:
            st.error(f"Could not load uploaded model: {exc}")
    elif repo_model_path is not None:
        try:
            active_model = load_keras_model_from_path(str(repo_model_path))
            file_mb = repo_model_path.stat().st_size / (1024 * 1024)
            active_model_source = f"{repo_model_path.name} ({file_mb:.1f} MB)"
        except Exception as exc:
            st.error(f"Could not load repository model: {exc}")

    if active_model is not None:
        total_params = active_model.count_params()
        trainable_params = sum(
            int(np.prod(w.shape)) for w in active_model.trainable_weights
        )
        non_trainable_params = total_params - trainable_params

        st.markdown("#### Active Model Summary")
        st.caption(f"**Source:** `{active_model_source}`")
        st.caption(f"**Architecture:** `{active_model.name}`")

        s1, s2 = st.columns(2)
        s1.metric("Total Params", f"{total_params:,}")
        s2.metric("Layers", f"{len(active_model.layers)}")

        s3, s4 = st.columns(2)
        s3.metric("Trainable", f"{trainable_params:,}")
        s4.metric("Non-Trainable", f"{non_trainable_params:,}")
    else:
        st.error("Model file `cat_dog_classifier.keras` not found.")


# -------------------------------------------------------------------------
# 5. Main Header & Two-Column Workspace
# -------------------------------------------------------------------------
st.markdown(
    '<div class="project-meta">DL-003 · Computer Vision · Binary Image Classification</div>'
    '<div class="main-title">Cat and Dog Image Classification</div>'
    '<div class="sub-caption">'
    'Real-time convolutional neural network inference with pre-inference validation '
    'to ensure only valid domestic Cat and Dog images are classified.'
    '</div>',
    unsafe_allow_html=True,
)

if active_model is None:
    st.error(
        "**Model Not Loaded:** Please place `cat_dog_classifier.keras` in the repository "
        "root or upload it from the sidebar."
    )
    st.stop()

col_left, col_right = st.columns([1, 1], gap="large")
pil_img = None
validation_result = None
result = None

with col_left:
    with st.container(border=True):
        st.markdown("#### 1. Input Image")
        uploaded_img_file = st.file_uploader(
            "Upload a Cat or Dog photo (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_img_file is not None:
            try:
                uploaded_img_file.seek(0)
                with Image.open(uploaded_img_file) as probe_img:
                    probe_img.verify()
                uploaded_img_file.seek(0)
                pil_img = Image.open(uploaded_img_file)
                pil_img.load()
                st.image(
                    pil_img,
                    caption=f"{uploaded_img_file.name} · {pil_img.size[0]}×{pil_img.size[1]} px",
                    use_container_width=True,
                )
            except (UnidentifiedImageError, OSError, ValueError) as img_err:
                st.error(
                    f"**Invalid Image File:** Unable to read `{uploaded_img_file.name}` ({img_err}). "
                    "Please upload a valid JPG, PNG, or WEBP image."
                )
                pil_img = None
        else:
            st.caption("Supported formats: JPG, JPEG, PNG, WEBP (minimum 32×32 px).")

with col_right:
    with st.container(border=True):
        st.markdown("#### 2. Classification Result")

        if pil_img is not None:
            validation_result = validate_cat_or_dog_image(pil_img)

            if not validation_result["is_valid"]:
                st.error(
                    "**Invalid Image — Not a Cat or Dog**\n\n"
                    f"{validation_result['reason']}"
                )
                st.caption(
                    "Prediction stopped: this model only classifies domestic Cat and Dog photos. "
                    "Images of other animals, people, vehicles, or objects are rejected."
                )
            else:
                result = run_real_cnn_inference(active_model, pil_img)

                st.success(f"**Predicted Class:** {result['pred_label']}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
                m2.metric("Sigmoid σ(z)", f"{result['raw_sigmoid']:.4f}")
                m3.metric("Latency", f"{result['latency_ms']:.1f} ms")

                st.divider()
                st.markdown("**Class Probability Distribution**")

                c_label, c_val = st.columns([3, 1])
                c_label.caption("Cat (Class 0) · 1 − σ(z)")
                c_val.caption(f"**{result['cat_prob'] * 100:.2f}%**")
                st.progress(int(round(result["cat_prob"] * 100)))

                d_label, d_val = st.columns([3, 1])
                d_label.caption("Dog (Class 1) · σ(z)")
                d_val.caption(f"**{result['dog_prob'] * 100:.2f}%**")
                st.progress(int(round(result["dog_prob"] * 100)))

                st.caption(
                    f"Input Tensor: `{result['tensor_shape']}` · "
                    f"Pixel Range: `[{result['tensor_min']:.0f}, {result['tensor_max']:.0f}]`"
                )
        else:
            st.info("Upload a Cat or Dog photo on the left to view live CNN predictions.")


# -------------------------------------------------------------------------
# 6. Clean Diagnostic Tabs (Feature Maps & Architecture Table)
# -------------------------------------------------------------------------
st.write("")
tab_fmaps, tab_arch = st.tabs(
    ["Convolutional Feature Maps", "Model Layer Architecture"]
)

with tab_fmaps:
    if result is not None:
        st.caption(
            "Intermediate activation maps extracted live from the first four `Conv2D` layers."
        )
        fmap_fig = plot_live_conv_feature_maps(active_model, result["input_tensor"])
        if fmap_fig is not None:
            st.pyplot(fmap_fig)
            plt.close(fmap_fig)
    else:
        st.caption(
            "Upload a valid Cat or Dog image above to inspect live `Conv2D` feature maps."
        )

with tab_arch:
    st.caption(
        f"Architecture summary for `{active_model.name}` ({active_model.count_params():,} total parameters)."
    )
    st.dataframe(
        get_dynamic_layer_table(active_model),
        use_container_width=True,
        hide_index=True,
    )
