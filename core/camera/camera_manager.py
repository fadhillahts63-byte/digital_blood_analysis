# core/camera/camera_manager.py
"""
Modul manajemen kamera untuk capture gambar.
"""
import cv2
import time
from pathlib import Path
from typing import Union


def capture_sample(
    output_path: Union[str, Path],
    camera_index: int = 0,
    resolution: tuple = (1280, 720),
    warmup_frames: int = 10,
    timeout: float = 5.0
) -> bool:
    """
    Ambil gambar dari kamera dan simpan ke file.

    Args:
        output_path: Path untuk menyimpan gambar (bisa str atau Path).
        camera_index: Indeks kamera (0=default, 1=external, dll).
        resolution: Resolusi gambar (width, height).
        warmup_frames: Jumlah frame untuk pemanasan (stabilkan eksposur).
        timeout: Waktu maksimal menunggu frame (detik).

    Returns:
        True jika berhasil, False jika gagal.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Tidak dapat membuka kamera (index={camera_index}).")

        # Set resolusi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Biarkan kamera "memanas" (stabilkan eksposur)
        start_time = time.time()
        for _ in range(warmup_frames):
            ret, _ = cap.read()
            if not ret:
                raise RuntimeError("Gagal membaca frame selama warmup.")
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout saat pemanasan kamera.")

        # Ambil frame terakhir (paling stabil)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Gagal mengambil frame gambar.")

        # Konversi output_path ke Path object untuk kemudahan handling
        output_path_obj = Path(output_path)

        # Buat parent directory jika belum ada
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Simpan sebagai JPEG berkualitas tinggi
        success = cv2.imwrite(str(output_path_obj), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        if not success:
            raise IOError(f"Gagal menyimpan gambar ke: {output_path_obj}")

        return True

    except Exception as e:
        print(f"[ERROR] capture_sample gagal: {e}")
        return False

    finally:
        if cap and cap.isOpened():
            cap.release()
            cv2.waitKey(1)  # Bantu OpenCV lepas resource
            time.sleep(0.1)