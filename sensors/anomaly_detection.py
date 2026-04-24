"""
DELTA - sensors/anomaly_detection.py
Rilevamento valori anomali nei dati sensori.
Usa range fisici e deviazione statistica per identificare errori hardware o condizioni critiche.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

from core.config import SENSOR_CONFIG

logger = logging.getLogger("delta.sensors.anomaly")

# Soglie fisiche assolute: (min, max, label)
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature_c":  (SENSOR_CONFIG["temp_min"],     SENSOR_CONFIG["temp_max"]),
    "humidity_pct":   (SENSOR_CONFIG["humidity_min"], SENSOR_CONFIG["humidity_max"]),
    "pressure_hpa":   (SENSOR_CONFIG["pressure_min"], SENSOR_CONFIG["pressure_max"]),
    "light_lux":      (SENSOR_CONFIG["light_min"],    SENSOR_CONFIG["light_max"]),
    "co2_ppm":        (SENSOR_CONFIG["co2_min"],      SENSOR_CONFIG["co2_max"]),
    "ph":             (SENSOR_CONFIG["ph_min"],        SENSOR_CONFIG["ph_max"]),
    "ec_ms_cm":       (SENSOR_CONFIG["ec_min"],        SENSOR_CONFIG["ec_max"]),
}

# Variazione massima consentita tra due letture consecutive (delta fisico plausibile)
MAX_DELTA: Dict[str, float] = {
    "temperature_c": 10.0,    # °C per step
    "humidity_pct":  30.0,    # % per step
    "pressure_hpa":  20.0,    # hPa per step
    "light_lux":     50000.0, # lux per step
    "co2_ppm":       500.0,   # ppm per step
    "ph":            2.0,     # pH per step
    "ec_ms_cm":      2.0,     # mS/cm per step
}


class AnomalyDetector:
    """
    Rileva anomalie nelle letture sensori tramite:
    1. Controllo range fisici assoluti
    2. Spike detection (variazione eccessiva tra letture consecutive)
    3. Sensore bloccato (valore invariato su N letture)
    """

    def __init__(self, stuck_threshold: int = 10):
        """
        Args:
            stuck_threshold: numero di letture identiche prima di segnalare sensore bloccato
        """
        self._prev_values: Dict[str, Optional[float]] = {f: None for f in PHYSICAL_BOUNDS}
        self._stuck_counters: Dict[str, int] = {f: 0 for f in PHYSICAL_BOUNDS}
        self._stuck_threshold = stuck_threshold

    def check(self, data: Dict[str, Any]) -> List[str]:
        """
        Controlla i dati per anomalie.

        Args:
            data: dizionario dati sensori (già smoothed)

        Returns:
            Lista di stringhe descrittive per ogni anomalia trovata (vuota se tutto OK)
        """
        anomalies: List[str] = []

        for field, (min_val, max_val) in PHYSICAL_BOUNDS.items():
            value = data.get(field)
            if value is None:
                continue

            try:
                val = float(value)
            except (TypeError, ValueError):
                anomalies.append(f"[{field}] Valore non numerico: {value}")
                continue

            # ── 1. Range fisici ──────────────────────────────
            if val < min_val or val > max_val:
                anomalies.append(
                    f"[{field}] Fuori range fisico: {val:.2f} (atteso [{min_val}, {max_val}])"
                )
                logger.warning("Anomalia range %s: %.2f", field, val)

            # ── 2. Spike detection ───────────────────────────
            prev = self._prev_values[field]
            if prev is not None:
                delta = abs(val - prev)
                max_delta = MAX_DELTA.get(field, float("inf"))
                if delta > max_delta:
                    anomalies.append(
                        f"[{field}] Spike rilevato: variazione {delta:.2f} "
                        f"(max consentita: {max_delta:.2f})"
                    )
                    logger.warning("Spike sensore %s: Δ=%.2f", field, delta)

            # ── 3. Sensore bloccato ──────────────────────────
            if prev is not None and val == prev:
                self._stuck_counters[field] += 1
                if self._stuck_counters[field] >= self._stuck_threshold:
                    anomalies.append(
                        f"[{field}] Possibile sensore bloccato "
                        f"(valore invariato per {self._stuck_counters[field]} letture)"
                    )
            else:
                self._stuck_counters[field] = 0

            self._prev_values[field] = val

        return anomalies

    def reset(self):
        """Azzera la storia interna (es. dopo sostituzione sensore)."""
        for field in PHYSICAL_BOUNDS:
            self._prev_values[field] = None
            self._stuck_counters[field] = 0
        logger.info("AnomalyDetector resettato.")
