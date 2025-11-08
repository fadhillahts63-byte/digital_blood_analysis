# core/analysis/diagnostic_engine.py
"""
Modul engine diagnosis untuk Digital Blood Analyzer.
Menghasilkan diagnosis berdasarkan hasil pencocokan fitur gambar darah.
"""
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


def get_diagnosis_by_image(
    image_path: Union[str, Path],
    db_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Dapatkan diagnosis berdasarkan gambar darah.

    Args:
        image_path: Path ke gambar darah yang akan dianalisis
        db_path: Path ke database referensi (opsional)
        confidence_threshold: Ambang keyakinan minimum (0.0 - 1.0)

    Returns:
        Dictionary berisi diagnosis lengkap, atau None jika tidak ditemukan
    """
    from core.preprocessing import preprocess_image
    from core.features import extract_features, find_best_match

    # 1. Preprocess gambar
    processed_img = preprocess_image(image_path)

    # 2. Ekstrak fitur dari gambar yang sudah diproses (ini adalah array, bukan path)
    features = extract_features(processed_img)

    # 3. Cari kecocokan terbaik
    match_result = find_best_match(
        features=features,
        db_path=db_path,
        threshold=confidence_threshold
    )

    if match_result is None:
        return None

    # 4. Kembalikan diagnosis lengkap
    diagnosis = {
        "id": match_result.get("id"),
        "name": match_result.get("name"),
        "cause": match_result.get("cause"),
        "solution": match_result.get("solution"),
        "confidence": float(match_result.get("score", 0)),
        "image_path": match_result.get("image_path"),
        "verified": match_result.get("verified", False),
        "features_matched": features,
        "matched_reference": match_result
    }

    return diagnosis


def get_diagnosis_by_features(
    features: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Dapatkan diagnosis berdasarkan fitur-fitur yang sudah diekstrak.
    Berguna jika fitur sudah diekstrak sebelumnya.
    """
    from core.features import find_best_match

    # Cari kecocokan terbaik
    match_result = find_best_match(
        features=features,
        db_path=db_path,
        threshold=confidence_threshold
    )

    if match_result is None:
        return None

    # Kembalikan diagnosis lengkap
    diagnosis = {
        "id": match_result.get("id"),
        "name": match_result.get("name"),
        "cause": match_result.get("cause"),
        "solution": match_result.get("solution"),
        "confidence": float(match_result.get("score", 0)),
        "image_path": match_result.get("image_path"),
        "verified": match_result.get("verified", False),
        "features_matched": features,
        "matched_reference": match_result
    }

    return diagnosis


def get_multiple_diagnoses(
    image_path: Union[str, Path],
    db_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = 0.3,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Dapatkan beberapa diagnosis kemungkinan berdasarkan gambar darah.
    Berguna untuk menampilkan beberapa kemungkinan terbaik.
    """
    from core.preprocessing import preprocess_image
    from core.features import extract_features, find_all_matches

    # 1. Preprocess gambar
    processed_img = preprocess_image(image_path)

    # 2. Ekstrak fitur
    features = extract_features(processed_img)

    # 3. Cari semua kecocokan yang melewati threshold
    all_matches = find_all_matches(
        features=features,
        db_path=db_path,
        threshold=confidence_threshold,
        top_k=top_k
    )

    # 4. Kembalikan semua diagnosis
    diagnoses = []
    for match_result, score in all_matches:
        diagnosis = {
            "id": match_result.get("id"),
            "name": match_result.get("name"),
            "cause": match_result.get("cause"),
            "solution": match_result.get("solution"),
            "confidence": float(score),
            "image_path": match_result.get("image_path"),
            "verified": match_result.get("verified", False),
            "features_matched": features,
            "matched_reference": match_result
        }
        diagnoses.append(diagnosis)

    return diagnoses


def get_diagnosis_summary(diagnosis: Optional[Dict[str, Any]]) -> str:
    """
    Dapatkan ringkasan diagnosis dalam format string.
    Berguna untuk tampilan UI.
    """
    if diagnosis is None:
        return "❌ Tidak ditemukan diagnosis yang sesuai."

    summary = (
        f"🩸 Penyakit: {diagnosis.get('name', 'Unknown')}\n"
        f"🔬 Penyebab: {diagnosis.get('cause', 'N/A')}\n"
        f"💊 Solusi: {diagnosis.get('solution', 'N/A')}\n"
        f"📊 Keyakinan: {diagnosis.get('confidence', 0) * 100:.1f}%\n"
        f"✅ Diverifikasi: {'Ya' if diagnosis.get('verified', False) else 'Tidak'}"
    )

    return summary


def validate_diagnosis(diagnosis: Optional[Dict[str, Any]]) -> bool:
    """
    Validasi apakah diagnosis valid (memiliki informasi penting).
    """
    if diagnosis is None:
        return False

    required_fields = ["id", "name", "cause", "solution", "confidence"]
    for field in required_fields:
        if field not in diagnosis or diagnosis[field] is None:
            return False

    return True


def save_diagnosis_report(
    diagnosis: Dict[str, Any],
    output_path: Union[str, Path],
    include_features: bool = False
) -> bool:
    """
    Simpan laporan diagnosis ke file JSON.
    """
    try:
        report = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "diagnosis": diagnosis
        }

        if not include_features:
            # Hapus fitur dari laporan jika tidak diminta
            report["diagnosis"]["features_matched"] = "Hidden for privacy"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan laporan: {e}")
        return False