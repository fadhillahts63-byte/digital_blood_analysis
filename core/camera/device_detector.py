# core/camera/device_detector.py
"""
Modul deteksi perangkat kamera untuk Windows.
Menggunakan pygrabber untuk nama perangkat dan fallback ke indeks OpenCV.
"""
import cv2
import time
from typing import Dict, List, Optional

# Cek apakah pygrabber tersedia
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False
    FilterGraph = None  # type: ignore


def list_camera_devices_pygrabber() -> List[str]:
    """
    Dapatkan daftar nama kamera via pygrabber (Windows).
    """
    if not HAS_PYGRABBER or FilterGraph is None:
        return []
    try:
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception as e:
        print(f"[WARN] pygrabber gagal: {e}")
        return []


def list_camera_indices_open_cv(max_test: int = 5, warmup_frames: int = 3) -> Dict[str, int]:
    """
    Coba semua indeks kamera, dan lihat mana yang bisa baca frame.
    Berguna untuk mikroskop yang tidak muncul di pygrabber.
    """
    available = {}
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            valid_frame = False
            for _ in range(warmup_frames):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0 and frame.mean() > 10:
                    valid_frame = True
                    break
                time.sleep(0.1)
            if valid_frame:
                available[f"Kamera {i}"] = i
        cap.release()
        time.sleep(0.2)
    return available


def list_available_camera_devices() -> Dict[str, int]:
    """
    Kembalikan mapping nama_perangkat -> indeks.
    Prioritas: pygrabber > fallback OpenCV indeks.
    """
    # Coba via pygrabber dulu
    if HAS_PYGRABBER:
        try:
            pygrabber_devices = list_camera_devices_pygrabber()
            if pygrabber_devices:
                return {name: i for i, name in enumerate(pygrabber_devices)}
        except Exception as e:
            print(f"[WARN] pygrabber gagal: {e}")

    # Jika gagal, fallback ke indeks OpenCV
    print("[INFO] pygrabber gagal, coba deteksi via indeks...")
    return list_camera_indices_open_cv()