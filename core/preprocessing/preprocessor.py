# core/processing/preprocessor.py
"""
Modul preprocessing gambar darah.
Melakukan normalisasi, kontras, dan pembersihan noise sebelum analisis.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional


def preprocess_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Preprocess gambar darah untuk analisis lebih lanjut.
    Fungsi ini sesuai dengan kebutuhan main.py versi terbaru.
    
    Args:
        image_input: Path ke gambar atau array OpenCV (BGR)
        
    Returns:
        Gambar yang sudah diproses (BGR format)
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

    # 2. Resize ke ukuran standar untuk konsistensi
    target_size = (512, 512)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    # 3. Konversi ke grayscale untuk preprocessing morfologi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5. Denoise (mengurangi noise tanpa menghilangkan detail penting)
    # Bilateral filter: preserve edges while smoothing
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 6. Normalisasi kontras (GANTI cv2.normalize dengan manual)
    # Ini adalah pendekatan yang aman dan tidak mengembalikan None
    if gray.size > 0:
        min_val = float(gray.min())
        max_val = float(gray.max())
        if max_val > min_val:
            # Normalisasi ke range [0, 255] secara manual
            gray = (255.0 * (gray - min_val) / (max_val - min_val)).astype(np.uint8)
        # else: jika max_val == min_val, maka gambar konstan (tidak perlu diubah)
    else:
        raise ValueError("Gambar grayscale kosong setelah preprocessing")

    # 7. Gabungkan kembali ke BGR agar output sesuai format OpenCV
    processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ✅ JAMIN fungsi selalu return nilai (tidak mungkin None)
    if processed_img is None:
        raise RuntimeError("Preprocessing gagal menghasilkan gambar")
    
    return processed_img


def preprocess_blood_smear_advanced(
    image_input: Union[str, Path, np.ndarray],
    enhance_nucleus: bool = False,
    enhance_rbc: bool = True,
    enhance_parasite: bool = False,
    threshold_low: int = 30,
    threshold_high: int = 200
) -> np.ndarray:
    """
    Preprocessing tingkat lanjut untuk gambar smear darah.
    Berguna untuk highlight fitur spesifik seperti parasit malaria, inti sel, dll.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Gambar tidak bisa dibaca dari: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("image_input harus berupa path (str/Path) atau numpy array")

    # Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE untuk kontras
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Tambahkan filter berdasarkan kebutuhan
    if enhance_parasite:
        # Filter untuk highlight area dengan kontras tinggi (potensi parasit)
        kernel = np.ones((3, 3), np.float32) / 9
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Threshold untuk highlight area penting
    _, thresh = cv2.threshold(enhanced, threshold_low, threshold_high, cv2.THRESH_BINARY)

    # Gabungkan hasil threshold dengan gambar asli untuk highlight
    highlighted = cv2.bitwise_and(img, img, mask=thresh)

    return highlighted