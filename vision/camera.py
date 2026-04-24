"""
DELTA - vision/camera.py
Acquisizione immagini/frame dalla Raspberry Pi Camera Module.
Supporta picamera2 (Pi Camera Module 3) e OpenCV come fallback.
Supporta anche il caricamento manuale da cartella di input (modalità no-camera).
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np

from core.config import VISION_CONFIG

logger = logging.getLogger("delta.vision.camera")


class CameraModule:
    """
    Gestisce l'acquisizione immagini dalla Raspberry Pi Camera o webcam USB.
    Ordine di priorità: picamera2 → OpenCV (cv2.VideoCapture).
    """

    def __init__(self):
        self._camera = None
        self._backend = None
        self._captures_dir = Path(VISION_CONFIG["captures_dir"])
        self._captures_dir.mkdir(parents=True, exist_ok=True)
        self._init_camera()

    # ─────────────────────────────────────────────
    # INIZIALIZZAZIONE
    # ─────────────────────────────────────────────

    def _init_camera(self):
        """Inizializza la camera: prova picamera2, poi OpenCV."""
        if self._try_init_picamera2():
            return
        if self._try_init_opencv():
            return
        logger.warning(
            "Nessuna camera disponibile. Le acquisizioni restituiranno None."
        )

    def _try_init_picamera2(self) -> bool:
        """Tenta l'inizializzazione con picamera2 (Raspberry Pi Camera Module)."""
        try:
            from picamera2 import Picamera2  # type: ignore

            cam = Picamera2()
            config = cam.create_still_configuration(
                main={
                    "size": (
                        VISION_CONFIG["capture_width"],
                        VISION_CONFIG["capture_height"],
                    ),
                    "format": "RGB888",
                }
            )
            cam.configure(config)
            cam.start()
            self._camera = cam
            self._backend = "picamera2"
            logger.info("Camera inizializzata con picamera2 (%dx%d).",
                        VISION_CONFIG["capture_width"], VISION_CONFIG["capture_height"])
            return True
        except (ImportError, Exception) as exc:
            logger.debug("picamera2 non disponibile: %s", exc)
            return False

    def _try_init_opencv(self) -> bool:
        """Tenta l'inizializzazione con OpenCV VideoCapture."""
        try:
            import cv2  # type: ignore

            idx = VISION_CONFIG["camera_index"]
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                logger.warning("OpenCV: camera indice %d non aperta.", idx)
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VISION_CONFIG["capture_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VISION_CONFIG["capture_height"])
            cap.set(cv2.CAP_PROP_FPS, VISION_CONFIG["fps"])

            # Imposta formato se possibile
            fourcc = cv2.VideoWriter_fourcc(*VISION_CONFIG["capture_format"])
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            self._camera = cap
            self._backend = "opencv"
            logger.info("Camera inizializzata con OpenCV (indice %d).", idx)
            return True
        except (ImportError, Exception) as exc:
            logger.debug("OpenCV VideoCapture non disponibile: %s", exc)
            return False

    # ─────────────────────────────────────────────
    # ACQUISIZIONE FRAME
    # ─────────────────────────────────────────────

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Acquisisce un singolo frame dalla camera.

        Returns:
            Array numpy (H, W, 3) BGR oppure None se non disponibile.
        """
        if self._camera is None:
            logger.warning("Camera non disponibile - generazione frame dummy.")
            return self._dummy_frame()

        try:
            if self._backend == "picamera2":
                return self._capture_picamera2()
            elif self._backend == "opencv":
                return self._capture_opencv()
        except Exception as exc:
            logger.error("Errore acquisizione frame: %s", exc, exc_info=True)

        return None

    def _capture_picamera2(self) -> Optional[np.ndarray]:
        """Acquisisce frame con picamera2 (RGB → BGR per compatibilità OpenCV)."""
        try:
            import cv2  # type: ignore
            frame_rgb = self._camera.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if VISION_CONFIG["save_captures"]:
                self._save_frame(frame_bgr)
            return frame_bgr
        except Exception as exc:
            logger.error("Errore picamera2 capture: %s", exc)
            return None

    def _capture_opencv(self) -> Optional[np.ndarray]:
        """Acquisisce frame con OpenCV VideoCapture."""
        ret, frame = self._camera.read()
        if not ret or frame is None:
            logger.error("OpenCV: lettura frame fallita.")
            return None
        if VISION_CONFIG["save_captures"]:
            self._save_frame(frame)
        return frame

    # ─────────────────────────────────────────────
    # UTILITÀ
    # ─────────────────────────────────────────────

    def _save_frame(self, frame: np.ndarray):
        """Salva il frame acquisito su disco per dataset/audit."""
        try:
            import cv2  # type: ignore
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            path = self._captures_dir / f"capture_{ts}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            logger.debug("Frame salvato: %s", path)
        except Exception as exc:
            logger.debug("Impossibile salvare frame: %s", exc)

    def _dummy_frame(self) -> np.ndarray:
        """Genera un frame verde sintetico per test senza camera."""
        h = VISION_CONFIG["capture_height"]
        w = VISION_CONFIG["capture_width"]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = 100  # Canale verde dominante
        return frame

    def release(self):
        """Rilascia le risorse della camera."""
        if self._camera is None:
            return
        try:
            if self._backend == "picamera2":
                self._camera.stop()
            elif self._backend == "opencv":
                self._camera.release()
            logger.info("Camera rilasciata.")
        except Exception as exc:
            logger.warning("Errore nel rilascio camera: %s", exc)
        finally:
            self._camera = None

    def __del__(self):
        self.release()


# ─────────────────────────────────────────────────────────────
# CARICAMENTO MANUALE DA CARTELLA (modalità no-camera)
# ─────────────────────────────────────────────────────────────

class ImageFolderLoader:
    """
    Alternativa a CameraModule: carica immagini da una cartella di input locale.
    Utile quando la videocamera non è disponibile o per analisi in batch.
    Espone la stessa interfaccia di CameraModule (capture_frame).
    """

    SUPPORTED_EXT: List[str] = VISION_CONFIG.get(
        "input_image_extensions",
        [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
    )

    def __init__(self, folder: Optional[str] = None):
        self._folder = Path(folder or VISION_CONFIG["input_images_dir"])
        self._folder.mkdir(parents=True, exist_ok=True)
        self._current_image_path: Optional[Path] = None
        logger.info("ImageFolderLoader inizializzato. Cartella: %s", self._folder)

    # ─────────────────────────────────────────────
    # QUERY CARTELLA
    # ─────────────────────────────────────────────

    def list_images(self) -> List[Path]:
        """Restituisce la lista ordinata di immagini nella cartella di input."""
        images = sorted(
            p for p in self._folder.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXT
        )
        return images

    def get_folder_path(self) -> Path:
        """Restituisce il percorso assoluto della cartella di input."""
        return self._folder

    # ─────────────────────────────────────────────
    # CARICAMENTO IMMAGINE
    # ─────────────────────────────────────────────

    def load_image(self, path: Path) -> Optional[np.ndarray]:
        """
        Carica un'immagine dal percorso specificato come array BGR numpy.

        Args:
            path: percorso del file immagine.

        Returns:
            Array numpy (H, W, 3) BGR oppure None in caso di errore.
        """
        try:
            import cv2  # type: ignore
            img = cv2.imread(str(path))
            if img is None:
                logger.error("Impossibile leggere l'immagine: %s", path)
                return None
            self._current_image_path = path
            logger.info("Immagine caricata da cartella: %s (%dx%d)",
                        path.name, img.shape[1], img.shape[0])
            return img
        except Exception as exc:
            logger.error("Errore caricamento immagine %s: %s", path, exc)
            return None

    def capture_frame(self, image_path: Optional[Path] = None) -> Optional[np.ndarray]:
        """
        Interfaccia compatibile con CameraModule.
        Se image_path è None carica la prima immagine disponibile nella cartella.

        Args:
            image_path: percorso specifico dell'immagine da caricare (opzionale).

        Returns:
            Array numpy BGR oppure None.
        """
        if image_path is not None:
            return self.load_image(image_path)

        images = self.list_images()
        if not images:
            logger.error(
                "Cartella input vuota: %s. Inserire almeno un'immagine.", self._folder
            )
            return None

        return self.load_image(images[0])

    # ─────────────────────────────────────────────
    # COMPATIBILITÀ CON CameraModule
    # ─────────────────────────────────────────────

    def release(self):
        """Nessuna risorsa da rilasciare (compatibilità con CameraModule)."""
        pass

    def __del__(self):
        pass
