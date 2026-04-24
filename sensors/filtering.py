"""
DELTA - sensors/filtering.py
Smoothing dei dati sensori tramite media mobile (rolling average).
Riduce il rumore nelle letture per diagnosi più stabili.
"""

import logging
from collections import deque
from typing import Dict, Any, Optional

logger = logging.getLogger("delta.sensors.filtering")

# Campi numerici da filtrare
NUMERIC_FIELDS = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "light_lux",
    "co2_ppm",
    "ph",
    "ec_ms_cm",
]


class SensorFilter:
    """
    Applica una media mobile su una finestra configurabile di letture.
    I valori None vengono ignorati nel calcolo della media.
    """

    def __init__(self, window: int = 5):
        """
        Args:
            window: numero di campioni per la media mobile
        """
        self._window = max(1, window)
        self._buffers: Dict[str, deque] = {
            field: deque(maxlen=self._window)
            for field in NUMERIC_FIELDS
        }
        logger.debug("SensorFilter inizializzato con finestra=%d.", self._window)

    def apply(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggiunge il campione corrente al buffer e restituisce
        un dizionario con i valori smoothed.

        Args:
            raw_data: lettura grezza dal SensorReader

        Returns:
            dizionario con valori filtrati (media mobile)
        """
        smoothed = dict(raw_data)

        for field in NUMERIC_FIELDS:
            value = raw_data.get(field)
            if value is not None:
                try:
                    self._buffers[field].append(float(value))
                except (TypeError, ValueError):
                    pass

            buf = self._buffers[field]
            if buf:
                smoothed[field] = round(sum(buf) / len(buf), 3)
            # Se il buffer è vuoto, il campo resta None (o il valore originale)

        return smoothed

    def reset(self):
        """Svuota tutti i buffer (es. dopo cambio pianta o luogo)."""
        for buf in self._buffers.values():
            buf.clear()
        logger.info("Buffer sensori azzerati.")

    def get_buffer_status(self) -> Dict[str, int]:
        """Restituisce il numero di campioni correnti per ogni campo."""
        return {field: len(buf) for field, buf in self._buffers.items()}
