"""
DELTA - sensors/reader.py
Lettura sensori Adafruit via I2C su Raspberry Pi.
Gestisce: temperatura, umidità, pressione (BME680),
luminosità (VEML7700), CO2 (SCD41), pH ed EC (ADS1115 + elettrodi).
Fallback su inserimento manuale se hardware non disponibile.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from core.config import SENSOR_CONFIG

logger = logging.getLogger("delta.sensors.reader")


class SensorReader:
    """
    Legge tutti i sensori ambientali via I2C.
    In assenza di hardware fisico, offre modalità manuale interattiva
    o genera valori demo per lo sviluppo.
    """

    def __init__(self):
        self._hw_available = False
        self._bme680 = None
        self._veml7700 = None
        self._scd41 = None
        self._ads = None
        self._i2c = None
        self._init_hardware()

    # ─────────────────────────────────────────────
    # INIZIALIZZAZIONE HARDWARE
    # ─────────────────────────────────────────────

    def _init_hardware(self):
        """Inizializza i sensori I2C. Non lancia eccezioni se non disponibili."""
        try:
            import board  # type: ignore
            import busio  # type: ignore
            import adafruit_bme680  # type: ignore
            import adafruit_veml7700  # type: ignore

            self._i2c = busio.I2C(board.SCL, board.SDA)

            # BME680 - Temperatura / Umidità / Pressione / Gas
            self._bme680 = adafruit_bme680.Adafruit_BME680_I2C(
                self._i2c,
                address=SENSOR_CONFIG["bme680_address"],
            )
            self._bme680.sea_level_pressure = 1013.25

            # VEML7700 - Luminosità
            self._veml7700 = adafruit_veml7700.VEML7700(self._i2c)

            # SCD41 - CO2
            try:
                import adafruit_scd4x  # type: ignore
                self._scd41 = adafruit_scd4x.SCD4X(self._i2c)
                self._scd41.start_periodic_measurement()
            except (ImportError, Exception) as exc:
                logger.warning("SCD41 non disponibile: %s", exc)

            # ADS1115 - ADC per pH/EC
            try:
                import adafruit_ads1x15.ads1115 as ADS  # type: ignore
                from adafruit_ads1x15.analog_in import AnalogIn  # type: ignore
                self._ads = ADS.ADS1115(self._i2c, address=SENSOR_CONFIG["ads1115_address"])
                self._ph_channel = AnalogIn(self._ads, SENSOR_CONFIG["ph_adc_channel"])
                self._ec_channel = AnalogIn(self._ads, SENSOR_CONFIG["ec_adc_channel"])
            except (ImportError, Exception) as exc:
                logger.warning("ADS1115 non disponibile: %s", exc)

            self._hw_available = True
            logger.info("Hardware sensori inizializzato correttamente.")

        except (ImportError, Exception) as exc:
            logger.warning(
                "Hardware sensori non disponibile (%s). "
                "Uso modalità simulazione/manuale.",
                exc,
            )
            self._hw_available = False

    # ─────────────────────────────────────────────
    # LETTURA COMPLETA
    # ─────────────────────────────────────────────

    def read_all(self) -> Dict[str, Any]:
        """
        Legge tutti i sensori e restituisce un dizionario con i valori.
        Se l'hardware non è disponibile, usa valori simulati.
        """
        if self._hw_available:
            return self._read_hardware()
        else:
            return self._read_simulated()

    # ─────────────────────────────────────────────
    # LETTURA DA HARDWARE
    # ─────────────────────────────────────────────

    def _read_hardware(self) -> Dict[str, Any]:
        """Lettura reale da sensori fisici."""
        data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "hardware",
        }

        # BME680
        try:
            data["temperature_c"] = round(float(self._bme680.temperature), 2)
            data["humidity_pct"] = round(float(self._bme680.relative_humidity), 2)
            data["pressure_hpa"] = round(float(self._bme680.pressure), 2)
        except Exception as exc:
            logger.error("Errore BME680: %s", exc)
            data.update({"temperature_c": None, "humidity_pct": None, "pressure_hpa": None})

        # VEML7700
        try:
            data["light_lux"] = round(float(self._veml7700.lux), 2)
        except Exception as exc:
            logger.error("Errore VEML7700: %s", exc)
            data["light_lux"] = None

        # SCD41
        if self._scd41 is not None:
            try:
                if self._scd41.data_ready:
                    data["co2_ppm"] = float(self._scd41.CO2)
                else:
                    data["co2_ppm"] = None
            except Exception as exc:
                logger.error("Errore SCD41: %s", exc)
                data["co2_ppm"] = None
        else:
            data["co2_ppm"] = None

        # pH via ADS1115
        if self._ads is not None:
            try:
                # Calibrazione lineare: V → pH (da calibrare con soluzioni buffer)
                ph_voltage = self._ph_channel.voltage
                data["ph"] = round(self._voltage_to_ph(ph_voltage), 2)
            except Exception as exc:
                logger.error("Errore pH: %s", exc)
                data["ph"] = None

            try:
                ec_voltage = self._ec_channel.voltage
                data["ec_ms_cm"] = round(self._voltage_to_ec(ec_voltage), 3)
            except Exception as exc:
                logger.error("Errore EC: %s", exc)
                data["ec_ms_cm"] = None
        else:
            data["ph"] = None
            data["ec_ms_cm"] = None

        return data

    # ─────────────────────────────────────────────
    # SIMULAZIONE (sviluppo / demo)
    # ─────────────────────────────────────────────

    def _read_simulated(self) -> Dict[str, Any]:
        """Genera valori simulati realistici per sviluppo e test."""
        import random
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "simulated",
            "temperature_c": round(random.uniform(18.0, 32.0), 2),
            "humidity_pct": round(random.uniform(40.0, 85.0), 2),
            "pressure_hpa": round(random.uniform(1010.0, 1025.0), 2),
            "light_lux": round(random.uniform(500.0, 50000.0), 2),
            "co2_ppm": round(random.uniform(400.0, 1200.0), 2),
            "ph": round(random.uniform(5.5, 7.5), 2),
            "ec_ms_cm": round(random.uniform(0.5, 3.0), 3),
        }

    # ─────────────────────────────────────────────
    # INSERIMENTO MANUALE
    # ─────────────────────────────────────────────

    def read_manual(self) -> Dict[str, Any]:
        """Acquisisce i dati sensori tramite input manuale dell'utente."""
        print("\n─── INSERIMENTO MANUALE SENSORI ───")
        data = {"timestamp": datetime.utcnow().isoformat(), "source": "manual"}
        fields = [
            ("temperature_c", "Temperatura (°C)"),
            ("humidity_pct", "Umidità relativa (%)"),
            ("pressure_hpa", "Pressione atmosferica (hPa)"),
            ("light_lux", "Luminosità (lux)"),
            ("co2_ppm", "CO2 (ppm)"),
            ("ph", "pH suolo"),
            ("ec_ms_cm", "Conducibilità elettrica (mS/cm)"),
        ]
        for key, label in fields:
            while True:
                raw = input(f"  {label}: ").strip()
                if raw == "":
                    data[key] = None
                    break
                try:
                    data[key] = round(float(raw), 3)
                    break
                except ValueError:
                    print("  Valore non valido. Inserire un numero o premere Enter per saltare.")
        return data

    # ─────────────────────────────────────────────
    # CONVERSIONI ELETTROCHIMICHE
    # ─────────────────────────────────────────────

    @staticmethod
    def _voltage_to_ph(voltage: float) -> float:
        """
        Conversione tensione → pH.
        Calibrazione lineare con soluzioni buffer pH 4.0 e 7.0.
        Modifica i coefficienti in base alla calibrazione reale.
        """
        # Esempio: 2.5V → pH 7.0, slope -0.18V/pH
        return 7.0 + (2.5 - voltage) / 0.18

    @staticmethod
    def _voltage_to_ec(voltage: float) -> float:
        """
        Conversione tensione → EC (mS/cm).
        Dipende dal circuito di condizionamento del sensore EC.
        Calibrare con soluzione EC standard.
        """
        return voltage * 1.0  # Placeholder lineare — calibrare sul campo
