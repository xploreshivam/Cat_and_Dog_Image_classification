import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

IMAGENET_CAT_INDICES = set(range(281, 286))
IMAGENET_DOG_INDICES = set(range(151, 269))
IMAGENET_WILD_CARNIVORE_INDICES = set(range(269, 281)) | set(range(286, 300))

MIN_IMAGE_DIMENSION = 32
MIN_PIXEL_STD = 5.0
MIN_PET_GROUP_PROB = 0.20
MIN_FALLBACK_PET_PROB = 0.30

_GATEKEEPER_MODEL = None


def get_gatekeeper_model() -> tf.keras.Model:
    """Lazy-loads the ImageNet-1000 MobileNetV2 gatekeeper model."""
    global _GATEKEEPER_MODEL
    if _GATEKEEPER_MODEL is None:
        _GATEKEEPER_MODEL = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=True
        )
    return _GATEKEEPER_MODEL


def verify_and_open_image(file_obj) -> Image.Image:
    """Verifies file headers and returns a fully loaded PIL Image or raises ValueError."""
    try:
        file_obj.seek(0)
        with Image.open(file_obj) as probe:
            probe.verify()
        file_obj.seek(0)
        img = Image.open(file_obj)
        img.load()
        return img
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Corrupted or unsupported image file: {exc}") from exc


def validate_cat_or_dog_image(pil_img: Image.Image) -> dict:
    """Checks that a PIL image is a valid domestic Cat or Dog before CNN prediction."""
    width, height = pil_img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return {
            "is_valid": False,
            "reason": (
                f"Image resolution ({width}×{height} px) is too small. "
                f"Minimum required size is {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} px."
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
            "reason": "Blank or solid-color image detected. Please upload a real Cat or Dog photo.",
            "detected_label": "Blank / Low-Variance Image",
            "detected_conf": 0.0,
            "cat_group_prob": 0.0,
            "dog_group_prob": 0.0,
            "pet_total_prob": 0.0,
        }

    gatekeeper = get_gatekeeper_model()
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
                f"Other animal detected (**{top1_label}**, {top1_conf * 100:.1f}% confidence). "
                "This classifier strictly accepts domestic **Cat** and **Dog** images only."
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
                f"Non-Cat/Dog image detected (appears to be **{top1_label}**, "
                f"{top1_conf * 100:.1f}% confidence; Cat/Dog score: {pet_total_prob * 100:.1f}%). "
                "Please upload a valid **Cat** or **Dog** photo only."
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
