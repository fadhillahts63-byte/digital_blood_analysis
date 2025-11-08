# core/utils/validators.py
"""
Modul validasi untuk Digital Blood Analyzer.
Menyediakan fungsi untuk memvalidasi input sebelum diproses.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import json


def validate_image_path(image_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validasi apakah path gambar valid dan bisa dibaca.

    Args:
        image_path: Path ke gambar

    Returns:
        (is_valid, error_message)
    """
    path_obj = Path(image_path)

    if not path_obj.exists():
        return False, f"File tidak ditemukan: {image_path}"

    if not path_obj.is_file():
        return False, f"Path bukan file: {image_path}"

    # Cek ekstensi file
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if path_obj.suffix.lower() not in valid_extensions:
        return False, f"Format file tidak didukung: {path_obj.suffix}"

    # Coba baca gambar
    try:
        img = cv2.imread(str(path_obj))
        if img is None:
            return False, f"OpenCV tidak bisa membaca gambar: {image_path}"
        if img.size == 0:
            return False, f"Gambar kosong: {image_path}"
    except Exception as e:
        return False, f"Error saat membaca gambar: {e}"

    return True, "Gambar valid"


def validate_camera_index(camera_index: int) -> Tuple[bool, str]:
    """
    Validasi apakah indeks kamera valid.

    Args:
        camera_index: Indeks kamera (0, 1, 2, ...)

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(camera_index, int):
        return False, f"Indeks kamera harus integer, bukan {type(camera_index)}"

    if camera_index < 0:
        return False, f"Indeks kamera tidak boleh negatif: {camera_index}"

    # Coba buka kamera untuk validasi
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                cap.release()
                return True, "Kamera valid"
            else:
                cap.release()
                return False, f"Kamera {camera_index} bisa dibuka tapi tidak baca frame"
        else:
            return False, f"Kamera {camera_index} tidak bisa dibuka"
    except Exception as e:
        return False, f"Error saat menguji kamera {camera_index}: {e}"


def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi struktur konfigurasi.

    Args:
        config: Dictionary konfigurasi

    Returns:
        (is_valid, error_message)
    """
    required_keys = ["paths", "camera", "analysis", "ui"]
    for key in required_keys:
        if key not in config:
            return False, f"Konfigurasi tidak memiliki key: {key}"

    # Validasi paths
    paths = config.get("paths", {})
    required_paths = ["capture_dir", "reference_dir", "database"]
    for path_key in required_paths:
        if path_key not in paths:
            return False, f"Konfigurasi paths tidak memiliki key: {path_key}"

    # Validasi camera
    camera = config.get("camera", {})
    required_camera = ["resolution", "fps", "buffer_size"]
    for cam_key in required_camera:
        if cam_key not in camera:
            return False, f"Konfigurasi camera tidak memiliki key: {cam_key}"

    # Validasi resolution
    resolution = camera.get("resolution", [])
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        return False, f"Resolution harus array [width, height], bukan: {resolution}"

    # Validasi analysis
    analysis = config.get("analysis", {})
    if "confidence_threshold" not in analysis:
        return False, "Konfigurasi analysis tidak memiliki confidence_threshold"

    return True, "Konfigurasi valid"


def validate_features(features: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi struktur fitur yang diekstrak.

    Args:
        features: Dictionary fitur dari extract_features

    Returns:
        (is_valid, error_message)
    """
    required_keys = [
        "area", "perimeter", "circularity", "aspect_ratio", "extent",
        "solidity", "equivalent_diameter", "convex_area", "contour_count",
        "major_axis_length", "minor_axis_length", "eccentricity",
        "color_mean", "color_std", "texture_contrast", "texture_homogeneity",
        "texture_energy", "texture_correlation"
    ]

    for key in required_keys:
        if key not in features:
            return False, f"Fitur tidak memiliki key: {key}"

    # Validasi tipe data
    for key, value in features.items():
        if key in ["area", "perimeter", "circularity", "aspect_ratio", "extent",
                   "solidity", "equivalent_diameter", "convex_area", "contour_count",
                   "major_axis_length", "minor_axis_length", "eccentricity",
                   "texture_contrast", "texture_homogeneity", "texture_energy", "texture_correlation"]:
            if not isinstance(value, (int, float)):
                return False, f"Fitur {key} harus berupa angka, bukan {type(value)}"

        elif key in ["color_mean", "color_std"]:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                return False, f"Fitur {key} harus array 3 elemen [R, G, B], bukan: {value}"

    return True, "Fitur valid"


def validate_diagnosis_result(diagnosis: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validasi hasil diagnosis.

    Args:
        diagnosis: Dictionary hasil diagnosis

    Returns:
        (is_valid, error_message)
    """
    if diagnosis is None:
        return False, "Diagnosis adalah None"

    required_keys = ["id", "name", "cause", "solution", "confidence", "image_path"]
    for key in required_keys:
        if key not in diagnosis:
            return False, f"Diagnosis tidak memiliki key: {key}"

    # Validasi confidence
    confidence = diagnosis.get("confidence", 0)
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        return False, f"Confidence harus antara 0-1, bukan: {confidence}"

    # Validasi image_path
    image_path = diagnosis.get("image_path")
    if image_path and not Path(image_path).exists():
        return False, f"Image path tidak ditemukan: {image_path}"

    return True, "Diagnosis valid"


def validate_database_file(db_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validasi file database referensi.

    Args:
        db_path: Path ke file database JSON

    Returns:
        (is_valid, error_message)
    """
    path_obj = Path(db_path)

    if not path_obj.exists():
        return False, f"Database file tidak ditemukan: {db_path}"

    if not path_obj.is_file():
        return False, f"Path bukan file: {db_path}"

    if path_obj.suffix.lower() != '.json':
        return False, f"Database harus berupa file JSON, bukan: {path_obj.suffix}"

    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            return False, "Database harus berupa array JSON"
        # Validasi struktur item pertama
        if data:
            item = data[0]
            required_keys = ["id", "name", "cause", "solution", "image_path", "features"]
            for key in required_keys:
                if key not in item:
                    return False, f"Item database tidak memiliki key: {key}"
    except json.JSONDecodeError as e:
        return False, f"File JSON tidak valid: {e}"
    except Exception as e:
        return False, f"Error saat membaca database: {e}"

    return True, "Database valid"


def validate_preprocessed_image(img: np.ndarray) -> Tuple[bool, str]:
    """
    Validasi array gambar yang sudah diproses.

    Args:
        img: Array gambar hasil preprocessing

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(img, np.ndarray):
        return False, f"Input bukan numpy array, tapi {type(img)}"

    if img.size == 0:
        return False, "Gambar kosong"

    if len(img.shape) not in [2, 3]:  # Grayscale atau BGR
        return False, f"Gambar harus 2D (grayscale) atau 3D (BGR), bukan: {img.shape}"

    if img.dtype not in [np.uint8, np.float32, np.float64]:
        return False, f"Gambar harus dtype uint8, float32, atau float64, bukan: {img.dtype}"

    return True, "Gambar valid"


def sanitize_path(path: Union[str, Path]) -> Path:
    """
    Sanitasi path untuk menghindari path traversal.

    Args:
        path: Path input

    Returns:
        Path object yang aman
    """
    path_obj = Path(path).resolve()
    app_dir = Path(__file__).parent.parent.parent.resolve()  # core/utils/../.. = root

    try:
        # Pastikan path berada di dalam aplikasi
        path_obj.relative_to(app_dir)
        return path_obj
    except ValueError:
        raise ValueError(f"Path tidak valid: {path} (berada di luar direktori aplikasi)")


# Fungsi utilitas tambahan
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Pembagian aman untuk menghindari pembagian dengan nol.
    """
    return a / b if b != 0 else default


def normalize_angle(angle: float) -> float:
    """
    Normalisasi sudut ke range [0, 360).
    """
    return angle % 360.0