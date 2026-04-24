"""
DELTA - vision/segmentation.py
Segmentazione degli organi vegetali dall'immagine.
Implementa segmentatori HSV e GrabCut per foglie, fiori e frutti.
"""

import logging
from typing import Optional, Tuple, List, Dict

import numpy as np

from core.config import VISION_CONFIG, ORGAN_CONFIG

logger = logging.getLogger("delta.vision.segmentation")


class LeafSegmentor:
    """
    Isola la regione di interesse (ROI) della foglia nell'immagine.
    Metodi disponibili:
    - 'hsv'    : maschera colore verde HSV (veloce, bassa complessità)
    - 'grabcut': segmentazione GrabCut OpenCV (più precisa, più lenta)
    """

    def __init__(self):
        self._method = VISION_CONFIG.get("segmentation_method", "hsv")
        self._hsv_lower = np.array(VISION_CONFIG["hsv_lower"], dtype=np.uint8)
        self._hsv_upper = np.array(VISION_CONFIG["hsv_upper"], dtype=np.uint8)
        self._min_area = VISION_CONFIG["min_leaf_area"]
        logger.debug("LeafSegmentor inizializzato con metodo='%s'.", self._method)

    # ─────────────────────────────────────────────
    # METODO PRINCIPALE
    # ─────────────────────────────────────────────

    def segment(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segmenta la foglia nell'immagine.

        Args:
            image: array numpy (H, W, 3) BGR

        Returns:
            (mask, roi):
                mask: maschera binaria (H, W) uint8, None se fallito
                roi:  crop della foglia (H', W', 3) BGR, None se area insufficiente
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            logger.error("OpenCV richiesto per la segmentazione.")
            return None, None

        if image is None or image.size == 0:
            return None, None

        try:
            if self._method == "grabcut":
                mask = self._segment_grabcut(image)
            else:
                mask = self._segment_hsv(image)

            if mask is None:
                return None, None

            roi = self._extract_roi(image, mask)
            return mask, roi

        except Exception as exc:
            logger.error("Errore segmentazione: %s", exc, exc_info=True)
            return None, None

    # ─────────────────────────────────────────────
    # SEGMENTAZIONE HSV
    # ─────────────────────────────────────────────

    def _segment_hsv(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Segmentazione basata su range colore verde HSV."""
        import cv2  # type: ignore

        # ── 1. Blur per ridurre rumore ────────────────────────
        blurred = cv2.GaussianBlur(image, (7, 7), 0)

        # ── 2. Converti in HSV ───────────────────────────────
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ── 3. Maschera colore foglia ─────────────────────────
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # ── 4. Operazioni morfologiche per pulizia ────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ── 5. Mantieni solo componente connessa più grande ───
        mask = self._keep_largest_component(mask)
        return mask

    # ─────────────────────────────────────────────
    # SEGMENTAZIONE GRABCUT
    # ─────────────────────────────────────────────

    def _segment_grabcut(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Segmentazione GrabCut: assume foglia al centro dell'immagine.
        Più accurata di HSV ma computazionalmente più costosa.
        """
        import cv2  # type: ignore

        h, w = image.shape[:2]
        margin_x = w // 6
        margin_y = h // 6
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask_gc = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(image, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            # Foreground: GC_FGD=1 o GC_PR_FGD=3
            binary = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0)
            return binary.astype(np.uint8)
        except Exception as exc:
            logger.warning("GrabCut fallito (%s), fallback su HSV.", exc)
            return self._segment_hsv(image)

    # ─────────────────────────────────────────────
    # ESTRAZIONE ROI
    # ─────────────────────────────────────────────

    def _extract_roi(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Ritaglia il bounding box della foglia segmentata.
        Restituisce None se l'area è insufficiente.
        """
        import cv2  # type: ignore

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.debug("Nessun contorno trovato nella maschera.")
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self._min_area:
            logger.debug("Area foglia troppo piccola: %.0f px² (min: %d).", area, self._min_area)
            return None

        x, y, w, h = cv2.boundingRect(largest)
        roi = image[y:y + h, x:x + w]
        logger.debug("ROI foglia estratta: (%d,%d) %dx%d, area=%.0f px².", x, y, w, h, area)
        return roi

    # ─────────────────────────────────────────────
    # UTILITÀ
    # ─────────────────────────────────────────────

    @staticmethod
    def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        """Mantiene solo la componente connessa più grande nella maschera binaria."""
        try:
            import cv2  # type: ignore
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels <= 1:
                return mask
            # Label 0 = sfondo, skip
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            return clean
        except Exception:
            return mask


# ─────────────────────────────────────────────
# SEGMENTATORE FIORE
# ─────────────────────────────────────────────

class FlowerSegmentor:
    """
    Isola la regione di interesse del fiore nell'immagine.
    Utilizza una combinazione di range HSV per rilevare fiori
    di diversi colori (giallo, bianco, rosa, rosso, viola).
    """

    def __init__(self):
        self._ranges = ORGAN_CONFIG["flower"]["ranges"]
        self._min_area = ORGAN_CONFIG["flower"]["min_area"]
        logger.debug("FlowerSegmentor inizializzato con %d range HSV.", len(self._ranges))

    def segment(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segmenta il fiore nell'immagine.

        Returns:
            (mask, roi): maschera binaria e ROI del fiore più grande
        """
        try:
            import cv2
        except ImportError:
            return None, None

        if image is None or image.size == 0:
            return None, None

        try:
            hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
            combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for r in self._ranges:
                lower = np.array(r["lower"], dtype=np.uint8)
                upper = np.array(r["upper"], dtype=np.uint8)
                combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

            roi = self._extract_largest_roi(image, combined)
            return combined, roi

        except Exception as exc:
            logger.error("Errore segmentazione fiore: %s", exc, exc_info=True)
            return None, None

    def _extract_largest_roi(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid = [c for c in contours if cv2.contourArea(c) >= self._min_area]
        if not valid:
            return None
        largest = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return image[y:y+h, x:x+w]


# ─────────────────────────────────────────────
# SEGMENTATORE FRUTTO
# ─────────────────────────────────────────────

class FruitSegmentor:
    """
    Isola la regione di interesse del frutto nell'immagine.
    Supporta frutti rossi (pomodoro), arancioni (agrumi),
    gialli (banana, limone) e verdi (uva, kiwi).
    """

    def __init__(self):
        self._ranges = ORGAN_CONFIG["fruit"]["ranges"]
        self._min_area = ORGAN_CONFIG["fruit"]["min_area"]
        logger.debug("FruitSegmentor inizializzato con %d range HSV.", len(self._ranges))

    def segment(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segmenta il frutto nell'immagine.

        Returns:
            (mask, roi): maschera binaria e ROI del frutto più grande
        """
        try:
            import cv2
        except ImportError:
            return None, None

        if image is None or image.size == 0:
            return None, None

        try:
            hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
            combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for r in self._ranges:
                lower = np.array(r["lower"], dtype=np.uint8)
                upper = np.array(r["upper"], dtype=np.uint8)
                combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

            roi = self._extract_largest_roi(image, combined)
            return combined, roi

        except Exception as exc:
            logger.error("Errore segmentazione frutto: %s", exc, exc_info=True)
            return None, None

    def _extract_largest_roi(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid = [c for c in contours if cv2.contourArea(c) >= self._min_area]
        if not valid:
            return None
        largest = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return image[y:y+h, x:x+w]



class LeafSegmentor:
    """
    Isola la regione di interesse (ROI) della foglia nell'immagine.
    Metodi disponibili:
    - 'hsv'    : maschera colore verde HSV (veloce, bassa complessità)
    - 'grabcut': segmentazione GrabCut OpenCV (più precisa, più lenta)
    """

    def __init__(self):
        self._method = VISION_CONFIG.get("segmentation_method", "hsv")
        self._hsv_lower = np.array(VISION_CONFIG["hsv_lower"], dtype=np.uint8)
        self._hsv_upper = np.array(VISION_CONFIG["hsv_upper"], dtype=np.uint8)
        self._min_area = VISION_CONFIG["min_leaf_area"]
        logger.debug("LeafSegmentor inizializzato con metodo='%s'.", self._method)

    # ─────────────────────────────────────────────
    # METODO PRINCIPALE
    # ─────────────────────────────────────────────

    def segment(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segmenta la foglia nell'immagine.

        Args:
            image: array numpy (H, W, 3) BGR

        Returns:
            (mask, roi):
                mask: maschera binaria (H, W) uint8, None se fallito
                roi:  crop della foglia (H', W', 3) BGR, None se area insufficiente
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            logger.error("OpenCV richiesto per la segmentazione.")
            return None, None

        if image is None or image.size == 0:
            return None, None

        try:
            if self._method == "grabcut":
                mask = self._segment_grabcut(image)
            else:
                mask = self._segment_hsv(image)

            if mask is None:
                return None, None

            roi = self._extract_roi(image, mask)
            return mask, roi

        except Exception as exc:
            logger.error("Errore segmentazione: %s", exc, exc_info=True)
            return None, None

    # ─────────────────────────────────────────────
    # SEGMENTAZIONE HSV
    # ─────────────────────────────────────────────

    def _segment_hsv(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Segmentazione basata su range colore verde HSV."""
        import cv2  # type: ignore

        # ── 1. Blur per ridurre rumore ────────────────────────
        blurred = cv2.GaussianBlur(image, (7, 7), 0)

        # ── 2. Converti in HSV ───────────────────────────────
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ── 3. Maschera colore foglia ─────────────────────────
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # ── 4. Operazioni morfologiche per pulizia ────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ── 5. Mantieni solo componente connessa più grande ───
        mask = self._keep_largest_component(mask)
        return mask

    # ─────────────────────────────────────────────
    # SEGMENTAZIONE GRABCUT
    # ─────────────────────────────────────────────

    def _segment_grabcut(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Segmentazione GrabCut: assume foglia al centro dell'immagine.
        Più accurata di HSV ma computazionalmente più costosa.
        """
        import cv2  # type: ignore

        h, w = image.shape[:2]
        margin_x = w // 6
        margin_y = h // 6
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask_gc = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(image, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            # Foreground: GC_FGD=1 o GC_PR_FGD=3
            binary = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0)
            return binary.astype(np.uint8)
        except Exception as exc:
            logger.warning("GrabCut fallito (%s), fallback su HSV.", exc)
            return self._segment_hsv(image)

    # ─────────────────────────────────────────────
    # ESTRAZIONE ROI
    # ─────────────────────────────────────────────

    def _extract_roi(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Ritaglia il bounding box della foglia segmentata.
        Restituisce None se l'area è insufficiente.
        """
        import cv2  # type: ignore

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.debug("Nessun contorno trovato nella maschera.")
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self._min_area:
            logger.debug("Area foglia troppo piccola: %.0f px² (min: %d).", area, self._min_area)
            return None

        x, y, w, h = cv2.boundingRect(largest)
        roi = image[y:y + h, x:x + w]
        logger.debug("ROI foglia estratta: (%d,%d) %dx%d, area=%.0f px².", x, y, w, h, area)
        return roi

    # ─────────────────────────────────────────────
    # UTILITÀ
    # ─────────────────────────────────────────────

    @staticmethod
    def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        """Mantiene solo la componente connessa più grande nella maschera binaria."""
        try:
            import cv2  # type: ignore
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels <= 1:
                return mask
            # Label 0 = sfondo, skip
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            return clean
        except Exception:
            return mask
