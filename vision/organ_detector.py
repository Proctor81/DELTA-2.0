"""
DELTA - vision/organ_detector.py
Rilevamento multi-organo della pianta.
Identifica e segmenta foglie, fiori e frutti nella stessa immagine,
restituendo la lista degli organi presenti con maschera e ROI.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from core.config import ORGAN_CONFIG

logger = logging.getLogger("delta.vision.organ_detector")


# ─────────────────────────────────────────────
# TIPI DI ORGANO
# ─────────────────────────────────────────────

ORGAN_LEAF   = "foglia"
ORGAN_FLOWER = "fiore"
ORGAN_FRUIT  = "frutto"

ALL_ORGANS = [ORGAN_LEAF, ORGAN_FLOWER, ORGAN_FRUIT]


class OrganResult:
    """Risultato del rilevamento di un organo."""

    def __init__(
        self,
        organ_type: str,
        mask: Optional[np.ndarray],
        roi: Optional[np.ndarray],
        coverage_ratio: float,
        bounding_boxes: List[Tuple[int, int, int, int]],
    ):
        self.organ_type = organ_type
        self.mask = mask
        self.roi = roi
        self.coverage_ratio = coverage_ratio        # Frazione dell'immagine coperta
        self.bounding_boxes = bounding_boxes        # [(x, y, w, h), ...]
        self.detected = coverage_ratio >= ORGAN_CONFIG["detection_confidence"]

    def __repr__(self):
        return (
            f"OrganResult({self.organ_type}, "
            f"rilevato={self.detected}, "
            f"copertura={self.coverage_ratio:.2%})"
        )


class PlantOrganDetector:
    """
    Rileva e segmenta foglie, fiori e frutti in un'immagine di pianta.
    Utilizza analisi HSV multi-range per identificare ciascun organo
    in base al suo profilo cromatico tipico.
    """

    def __init__(self):
        if not CV2_AVAILABLE:
            logger.warning("OpenCV non disponibile — rilevamento organi simulato.")
        logger.info("PlantOrganDetector inizializzato.")

    # ─────────────────────────────────────────────
    # RILEVAMENTO COMPLETO
    # ─────────────────────────────────────────────

    def detect_all(self, image: np.ndarray) -> Dict[str, OrganResult]:
        """
        Rileva tutti gli organi presenti nell'immagine.

        Args:
            image: frame BGR acquisito dalla camera

        Returns:
            Dict {organ_type: OrganResult}
        """
        if image is None or image.size == 0:
            logger.warning("Immagine vuota — rilevamento skip.")
            return self._empty_results()

        results = {}
        results[ORGAN_LEAF]   = self._detect_leaf(image)
        if ORGAN_CONFIG.get("enable_flower_analysis", True):
            results[ORGAN_FLOWER] = self._detect_flower(image)
        if ORGAN_CONFIG.get("enable_fruit_analysis", True):
            results[ORGAN_FRUIT]  = self._detect_fruit(image)

        detected = [k for k, v in results.items() if v.detected]
        logger.info("Organi rilevati: %s", detected if detected else ["nessuno"])
        return results

    # ─────────────────────────────────────────────
    # RILEVAMENTO FOGLIA
    # ─────────────────────────────────────────────

    def _detect_leaf(self, image: np.ndarray) -> OrganResult:
        cfg = ORGAN_CONFIG["leaf"]
        lower = np.array(cfg["hsv_lower"], dtype=np.uint8)
        upper = np.array(cfg["hsv_upper"], dtype=np.uint8)
        mask, roi, ratio, boxes = self._single_range_segment(
            image, lower, upper, cfg["min_area"]
        )
        return OrganResult(ORGAN_LEAF, mask, roi, ratio, boxes)

    # ─────────────────────────────────────────────
    # RILEVAMENTO FIORE
    # ─────────────────────────────────────────────

    def _detect_flower(self, image: np.ndarray) -> OrganResult:
        cfg = ORGAN_CONFIG["flower"]
        mask, roi, ratio, boxes = self._multi_range_segment(
            image, cfg["ranges"], cfg["min_area"]
        )
        return OrganResult(ORGAN_FLOWER, mask, roi, ratio, boxes)

    # ─────────────────────────────────────────────
    # RILEVAMENTO FRUTTO
    # ─────────────────────────────────────────────

    def _detect_fruit(self, image: np.ndarray) -> OrganResult:
        cfg = ORGAN_CONFIG["fruit"]
        mask, roi, ratio, boxes = self._multi_range_segment(
            image, cfg["ranges"], cfg["min_area"]
        )
        return OrganResult(ORGAN_FRUIT, mask, roi, ratio, boxes)

    # ─────────────────────────────────────────────
    # HELPER: SEGMENTAZIONE HSV SINGOLO RANGE
    # ─────────────────────────────────────────────

    @staticmethod
    def _single_range_segment(
        image: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        min_area: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, List]:
        if not CV2_AVAILABLE:
            return None, image, 0.0, []
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            return PlantOrganDetector._process_mask(image, mask, min_area)
        except Exception as exc:
            logger.error("Errore segmentazione HSV: %s", exc)
            return None, None, 0.0, []

    # ─────────────────────────────────────────────
    # HELPER: SEGMENTAZIONE HSV MULTI-RANGE
    # ─────────────────────────────────────────────

    @staticmethod
    def _multi_range_segment(
        image: np.ndarray,
        ranges: List[Dict],
        min_area: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, List]:
        if not CV2_AVAILABLE:
            return None, image, 0.0, []
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for r in ranges:
                lower = np.array(r["lower"], dtype=np.uint8)
                upper = np.array(r["upper"], dtype=np.uint8)
                combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv, lower, upper))
            return PlantOrganDetector._process_mask(image, combined_mask, min_area)
        except Exception as exc:
            logger.error("Errore segmentazione multi-range HSV: %s", exc)
            return None, None, 0.0, []

    # ─────────────────────────────────────────────
    # HELPER: ELABORAZIONE MASCHERA
    # ─────────────────────────────────────────────

    @staticmethod
    def _process_mask(
        image: np.ndarray,
        mask: np.ndarray,
        min_area: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, List]:
        # Pulizia morfologica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        total_pixels = mask.shape[0] * mask.shape[1]
        mask_pixels  = int(np.sum(mask > 0))
        ratio = mask_pixels / total_pixels if total_pixels > 0 else 0.0

        # Estrazione contorni / bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        largest_roi = None
        largest_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
            if area > largest_area:
                largest_area = area
                largest_roi  = image[y:y+h, x:x+w]

        return mask, largest_roi, ratio, boxes

    # ─────────────────────────────────────────────
    # HELPER: RISULTATI VUOTI
    # ─────────────────────────────────────────────

    @staticmethod
    def _empty_results() -> Dict[str, OrganResult]:
        return {
            organ: OrganResult(organ, None, None, 0.0, [])
            for organ in ALL_ORGANS
        }

    # ─────────────────────────────────────────────
    # UTILITÀ
    # ─────────────────────────────────────────────

    @staticmethod
    def primary_organ(results: Dict[str, OrganResult]) -> str:
        """Restituisce l'organo con maggiore copertura (priorità fiore > frutto > foglia)."""
        priority = [ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_LEAF]
        for organ in priority:
            if organ in results and results[organ].detected:
                return organ
        return ORGAN_LEAF  # default

    @staticmethod
    def detected_organs(results: Dict[str, OrganResult]) -> List[str]:
        """Lista degli organi effettivamente rilevati."""
        return [o for o, r in results.items() if r.detected]

    @staticmethod
    def summary(results: Dict[str, OrganResult]) -> str:
        """Testo riassuntivo degli organi rilevati."""
        detected = PlantOrganDetector.detected_organs(results)
        if not detected:
            return "Nessun organo vegetale identificato"
        parts = []
        for organ in detected:
            r = results[organ]
            parts.append(f"{organ} ({r.coverage_ratio:.1%})")
        return "Organi rilevati: " + ", ".join(parts)
