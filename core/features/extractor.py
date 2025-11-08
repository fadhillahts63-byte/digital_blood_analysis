# core/features/extractor.py
"""
Modul ekstraksi fitur gambar darah.
Mengekstrak fitur morfologi, warna, tekstur untuk analisis.
"""
import cv2
import numpy as np
from typing import Dict, Any, Union, Tuple
from pathlib import Path


def extract_features(image_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
    """
    Ekstrak fitur dari gambar darah untuk analisis lebih lanjut.

    Args:
        image_input: Path ke gambar atau array OpenCV (BGR/Grayscale)

    Returns:
        Dictionary berisi fitur-fitur yang diekstrak
    """
    # 1. Load gambar jika input adalah path
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Gambar tidak bisa dibaca dari: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("image_input harus berupa path (str/Path) atau numpy array")

    # 2. Konversi ke grayscale jika BGR
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 3. Deteksi kontur untuk ekstraksi fitur morfologi
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Ekstrak fitur morfologi
    features = {
        "area": 0,
        "perimeter": 0,
        "circularity": 0,
        "aspect_ratio": 0,
        "extent": 0,
        "solidity": 0,
        "equivalent_diameter": 0,
        "convex_area": 0,
        "contour_count": 0,
        "major_axis_length": 0,
        "minor_axis_length": 0,
        "eccentricity": 0,
        "color_mean": [0, 0, 0],
        "color_std": [0, 0, 0],
        "texture_contrast": 0,
        "texture_homogeneity": 0,
        "texture_energy": 0,
        "texture_correlation": 0
    }

    if len(contours) > 0:
        # Ambil kontur terbesar sebagai objek utama
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Fitur morfologi
        features["area"] = float(area)
        features["perimeter"] = float(perimeter)

        if perimeter > 0:
            features["circularity"] = float(4 * np.pi * area / (perimeter * perimeter))

        if len(largest_contour) >= 5:
            # Fitting ellipse untuk aspect ratio dan eccentricity
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                (major_axis, minor_axis) = ellipse[1]
                features["major_axis_length"] = float(major_axis)
                features["minor_axis_length"] = float(minor_axis)
                if major_axis > 0:
                    features["aspect_ratio"] = float(minor_axis / major_axis)
                    features["eccentricity"] = float(np.sqrt(1 - (minor_axis / major_axis) ** 2))
            except cv2.error:
                pass

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_area = w * h
        if rect_area > 0:
            features["extent"] = float(area / rect_area)

        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            features["solidity"] = float(area / hull_area)
            features["convex_area"] = float(hull_area)

        # Equivalent diameter
        if area > 0:
            features["equivalent_diameter"] = float(np.sqrt(4 * area / np.pi))

        features["contour_count"] = len(contours)

    # 5. Ekstrak fitur warna (jika BGR) - FIX: Konversi ke numpy array eksplisit
    if len(img.shape) == 3:
        # Pastikan img adalah numpy array dengan dtype yang sesuai
        img_array = np.asarray(img, dtype=np.float32)
        # Rata-rata dan std warna
        color_mean = np.mean(img_array, axis=(0, 1))
        color_std = np.std(img_array, axis=(0, 1))
        features["color_mean"] = [float(c) for c in color_mean]
        features["color_std"] = [float(c) for c in color_std]

    # 6. Ekstrak fitur tekstur (Gray-Level Co-occurrence Matrix - GLCM)
    try:
        from skimage.feature import graycomatrix, graycoprops
        # Pastikan gray adalah numpy array
        gray_array = np.asarray(gray, dtype=np.uint8)
        glcm = graycomatrix(gray_array, [1], [0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        features["texture_contrast"] = float(graycoprops(glcm, 'contrast').mean())
        features["texture_homogeneity"] = float(graycoprops(glcm, 'homogeneity').mean())
        features["texture_energy"] = float(graycoprops(glcm, 'energy').mean())
        features["texture_correlation"] = float(graycoprops(glcm, 'correlation').mean())
    except ImportError:
        # Jika skimage tidak tersedia, gunakan pendekatan sederhana
        gray_array = np.asarray(gray, dtype=np.float32)
        features["texture_contrast"] = float(gray_array.std())
        features["texture_homogeneity"] = 0
        features["texture_energy"] = 0
        features["texture_correlation"] = 0

    return features


def extract_advanced_features(image_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
    """
    Ekstrak fitur lanjutan untuk analisis spesifik darah.
    Misalnya untuk deteksi parasit malaria, inti sel, dll.
    """
    # Implementasi lanjutan sesuai kebutuhan spesifik
    features = extract_features(image_input)
    # Tambahkan fitur lanjutan di sini
    return features