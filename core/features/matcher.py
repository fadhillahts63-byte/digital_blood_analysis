# core/features/matcher.py
"""
Modul pencocokan fitur gambar darah.
Mencari kecocokan antara fitur gambar dengan database referensi.
"""
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from .extractor import extract_features


def load_reference_database(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Load database referensi dari file JSON.
    Struktur: [
        {
            "id": "malaria_falciparum",
            "name": "Malaria (Plasmodium falciparum)",
            "cause": "Infeksi parasit Plasmodium falciparum",
            "solution": "Pengobatan dengan artemisinin-based combination therapy (ACT)",
            "image_path": "references/malaria_falciparum.jpg",
            "features": {...},  # hasil dari extract_features
            "verified": true
        }
    ]
    """
    if db_path is None:
        # Default path ke database referensi
        db_path = Path(__file__).parent.parent.parent / "data" / "references" / "reference_db.json"

    # Konversi ke Path object jika string
    db_path_obj = Path(db_path) if isinstance(db_path, str) else db_path

    if not db_path_obj.exists():
        # Jika database tidak ada, kembalikan contoh dummy
        return [
            {
                "id": "malaria_ring_stage",
                "name": "Malaria (Ring Stage)",
                "cause": "Infeksi parasit Plasmodium dalam bentuk ring stage",
                "solution": "Pengobatan dengan ACT (Artemisinin-based Combination Therapy)",
                "image_path": "references/malaria_ring_stage.jpg",
                "features": {
                    "area": 100.0,
                    "circularity": 0.8,
                    "aspect_ratio": 1.0,
                    "color_mean": [100, 80, 60],
                    "texture_contrast": 20.0
                },
                "verified": True
            },
            {
                "id": "anemia_normocytic",
                "name": "Anemia Normocytic",
                "cause": "Penurunan produksi hemoglobin",
                "solution": "Suplementasi zat besi dan vitamin B12",
                "image_path": "references/anemia_normocytic.jpg",
                "features": {
                    "area": 80.0,
                    "circularity": 0.75,
                    "aspect_ratio": 1.1,
                    "color_mean": [120, 100, 90],
                    "texture_contrast": 15.0
                },
                "verified": True
            }
        ]

    with open(db_path_obj, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_feature_similarity(
    features1: Dict[str, Any],
    features2: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Hitung kesamaan antara dua set fitur.
    
    Args:
        features1, features2: Dictionary fitur dari extract_features
        weights: Bobot untuk setiap fitur (semakin besar, semakin penting)
        
    Returns:
        Nilai kesamaan antara 0-1
    """
    if weights is None:
        weights = {
            "area": 0.2,
            "circularity": 0.2,
            "aspect_ratio": 0.15,
            "color_mean": 0.2,
            "texture_contrast": 0.1,
            "texture_homogeneity": 0.05,
            "equivalent_diameter": 0.1
        }

    total_weight = 0.0
    similarity_score = 0.0

    for key, weight in weights.items():
        if key not in features1 or key not in features2:
            continue

        val1 = features1[key]
        val2 = features2[key]

        # Normalisasi berdasarkan tipe data
        if isinstance(val1, (list, tuple)):
            # Misalnya color_mean: [R, G, B]
            diff = np.linalg.norm(np.array(val1, dtype=np.float64) - np.array(val2, dtype=np.float64))
            max_possible_diff = np.linalg.norm(np.array([255, 255, 255], dtype=np.float64))  # Max RGB diff
            normalized_diff = float(diff / max_possible_diff if max_possible_diff > 0 else 0)
            similarity = 1.0 - normalized_diff
        elif isinstance(val1, (int, float, np.number)):
            # Pastikan val1 dan val2 adalah float
            val1_f = float(val1)
            val2_f = float(val2)
            # Normalisasi perbedaan numerik
            max_val = max(abs(val1_f), abs(val2_f), 1.0)  # Hindari pembagian dengan 0
            diff = abs(val1_f - val2_f) / max_val
            similarity = max(0.0, 1.0 - diff)
        else:
            # Default: jika tidak bisa dibandingkan, anggap tidak mirip
            similarity = 0.0

        similarity_score += float(similarity * weight)
        total_weight += weight

    return float(similarity_score / total_weight if total_weight > 0 else 0.0)


def find_best_match(
    features: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Cari kecocokan terbaik dari database referensi.
    
    Args:
        features: Fitur dari gambar sampel (hasil extract_features)
        db_path: Path ke database referensi (opsional)
        threshold: Ambang kesamaan minimum (0.0 - 1.0)
        
    Returns:
        Dictionary hasil pencocokan terbaik, atau None jika tidak ada yang cukup mirip
    """
    reference_db = load_reference_database(db_path)

    best_match = None
    best_score = 0.0

    for ref in reference_db:
        if "features" not in ref:
            continue

        similarity = calculate_feature_similarity(features, ref["features"])
        if similarity > best_score:
            best_score = similarity
            best_match = ref

    # Kembalikan hasil hanya jika melewati threshold
    if best_match and best_score >= threshold:
        best_match["score"] = best_score
        return best_match
    else:
        return None


def find_all_matches(
    features: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.3,
    top_k: int = 3
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Cari semua kecocokan yang melewati threshold, diurutkan dari yang terbaik.
    """
    reference_db = load_reference_database(db_path)
    matches = []

    for ref in reference_db:
        if "features" not in ref:
            continue

        similarity = calculate_feature_similarity(features, ref["features"])
        if similarity >= threshold:
            matches.append((ref, similarity))

    # Urutkan berdasarkan skor tertinggi
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]