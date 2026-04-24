"""
DELTA - recommendations/agronomy_engine.py
Generatore di raccomandazioni agronomiche operative.
Produce consigli su irrigazione, nutrienti, luce, CO2 e suolo
in base alla diagnosi e ai dati ambientali correnti.
"""

import logging
from typing import Dict, Any, List

from core.config import SENSOR_CONFIG

logger = logging.getLogger("delta.recommendations")


class AgronomyEngine:
    """
    Genera raccomandazioni operative personalizzate per ogni diagnosi.
    Ogni categoria produce uno o più consigli strutturati con:
    - categoria
    - priorità
    - testo operativo
    - azione consigliata
    """

    def generate(
        self,
        diagnosis: Dict[str, Any],
        sensor_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Genera la lista completa di raccomandazioni.

        Args:
            diagnosis:   output di DiagnosisEngine.diagnose()
            sensor_data: dati sensori correnti (smoothed)

        Returns:
            Lista di dict con raccomandazioni ordinate per priorità
        """
        recs: List[Dict[str, Any]] = []

        recs.extend(self._irrigation_recs(sensor_data))
        recs.extend(self._nutrient_recs(sensor_data, diagnosis))
        recs.extend(self._light_recs(sensor_data))
        recs.extend(self._co2_recs(sensor_data))
        recs.extend(self._soil_recs(sensor_data))
        recs.extend(self._disease_recs(diagnosis))
        recs.extend(self._flower_recs(diagnosis, sensor_data))
        recs.extend(self._fruit_recs(diagnosis, sensor_data))
        recs.extend(self._quantum_risk_recs(diagnosis))

        # Ordina per priorità (1 = massima)
        recs.sort(key=lambda r: r.get("priority", 99))

        logger.debug("Raccomandazioni generate: %d.", len(recs))
        return recs

    # ─────────────────────────────────────────────
    # IRRIGAZIONE
    # ─────────────────────────────────────────────

    def _irrigation_recs(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        hum = sensor_data.get("humidity_pct")
        temp = sensor_data.get("temperature_c")

        if hum is not None:
            if hum < SENSOR_CONFIG["humidity_optimal_min"]:
                recs.append(self._rec(
                    "irrigazione", 1,
                    f"Umidità attuale {hum:.1f}% sotto il minimo ottimale "
                    f"({SENSOR_CONFIG['humidity_optimal_min']}%).",
                    "Aumentare la frequenza di irrigazione. "
                    "Verificare il substrato: se asciutto in superficie, irrigare immediatamente.",
                ))
            elif hum > SENSOR_CONFIG["humidity_fungal_risk"]:
                recs.append(self._rec(
                    "irrigazione", 2,
                    f"Umidità elevata ({hum:.1f}%) → rischio patologie fungine.",
                    "Sospendere temporaneamente l'irrigazione. "
                    "Migliorare la ventilazione. "
                    "Verificare sistema di drenaggio.",
                ))
            elif hum > SENSOR_CONFIG["humidity_optimal_max"]:
                recs.append(self._rec(
                    "irrigazione", 3,
                    f"Umidità ({hum:.1f}%) sopra l'ottimale.",
                    "Ridurre la frequenza di irrigazione. "
                    "Aumentare ventilazione per abbassare umidità.",
                ))

        if temp is not None and hum is not None:
            if temp > 30.0 and hum < 50.0:
                recs.append(self._rec(
                    "irrigazione", 1,
                    "Condizioni di stress idrico elevato (alta T + bassa umidità).",
                    "Incrementare irrigazione nelle ore fresche (mattino/sera). "
                    "Valutare nebulizzazione fogliare.",
                ))

        return recs

    # ─────────────────────────────────────────────
    # NUTRIENTI
    # ─────────────────────────────────────────────

    def _nutrient_recs(
        self, sensor_data: Dict[str, Any], diagnosis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        recs = []
        ec = sensor_data.get("ec_ms_cm")
        ph = sensor_data.get("ph")
        ai_class = diagnosis.get("ai_class", "Sano")

        if ec is not None:
            if ec < SENSOR_CONFIG["ec_optimal_min"]:
                recs.append(self._rec(
                    "nutrienti", 2,
                    f"EC bassa ({ec:.2f} mS/cm): soluzione nutritiva impoverita.",
                    "Integrare fertilizzante NPK bilanciato. "
                    "Target EC: 1.2–2.0 mS/cm per piante in crescita vegetativa.",
                ))
            elif ec > SENSOR_CONFIG["ec_toxic"]:
                recs.append(self._rec(
                    "nutrienti", 1,
                    f"EC critica ({ec:.2f} mS/cm): rischio bruciatura radicale.",
                    "Diluire immediatamente la soluzione nutritiva con acqua pulita. "
                    "Eseguire flush radicale. "
                    "Target EC: < 2.5 mS/cm.",
                ))
            elif ec > SENSOR_CONFIG["ec_optimal_max"]:
                recs.append(self._rec(
                    "nutrienti", 3,
                    f"EC elevata ({ec:.2f} mS/cm): leggero eccesso salino.",
                    "Ridurre concentrazione fertilizzante del 20-30%. "
                    "Monitorare con attenzione.",
                ))

        # Carenza azoto (da AI)
        if "Carenza_azoto" in ai_class:
            recs.append(self._rec(
                "nutrienti", 1,
                "Carenza di azoto rilevata (clorosi progressiva).",
                "Somministrare fertilizzante ad alto contenuto di N "
                "(es. urea, nitrato di calcio). "
                "Verificare pH: l'azoto si assorbe meglio tra pH 6.0–7.0.",
            ))

        # Carenza ferro (da AI)
        if "Carenza_ferro" in ai_class:
            recs.append(self._rec(
                "nutrienti", 2,
                "Carenza di ferro rilevata (clorosi internervale).",
                "Applicare chelato di ferro (Fe-EDTA o Fe-DTPA). "
                "Correggere pH verso 6.0–6.5 per migliorare disponibilità.",
            ))

        return recs

    # ─────────────────────────────────────────────
    # LUCE
    # ─────────────────────────────────────────────

    def _light_recs(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        light = sensor_data.get("light_lux")

        if light is None:
            return recs

        if light < SENSOR_CONFIG["light_photosynthesis_min"]:
            recs.append(self._rec(
                "luce", 2,
                f"Luminosità insufficiente ({light:.0f} lux): fotosintesi compromessa.",
                "Spostare la pianta in zona più illuminata. "
                "Valutare integrazione con lampade a LED full-spectrum "
                "(target: 15.000–25.000 lux per la maggior parte delle specie).",
            ))
        elif light > SENSOR_CONFIG["light_stress_high"]:
            recs.append(self._rec(
                "luce", 2,
                f"Luminosità eccessiva ({light:.0f} lux): rischio foto-inibizione.",
                "Applicare ombreggiatura (50–60%). "
                "Evitare esposizione diretta nelle ore di picco (12:00–15:00).",
            ))
        elif light < SENSOR_CONFIG["light_photosynthesis_optimal"] * 0.5:
            recs.append(self._rec(
                "luce", 3,
                f"Luminosità sotto l'ottimale ({light:.0f} lux).",
                "Migliorare l'esposizione luminosa ove possibile. "
                "Monitorare la crescita nelle settimane successive.",
            ))

        return recs

    # ─────────────────────────────────────────────
    # CO2
    # ─────────────────────────────────────────────

    def _co2_recs(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        co2 = sensor_data.get("co2_ppm")

        if co2 is None:
            return recs

        if co2 < SENSOR_CONFIG["co2_optimal_min"]:
            recs.append(self._rec(
                "co2", 3,
                f"CO₂ sotto ottimale ({co2:.0f} ppm).",
                "Migliorare ventilazione dell'ambiente per favorire ricambio d'aria. "
                "In ambienti chiusi, valutare integrazione CO₂ (target: 800–1200 ppm).",
            ))
        elif co2 >= SENSOR_CONFIG["co2_enhanced_growth"]:
            recs.append(self._rec(
                "co2", 4,
                f"CO₂ ottimale per crescita accelerata ({co2:.0f} ppm).",
                "Condizioni favorevoli. Mantenere livello attuale.",
            ))

        return recs

    # ─────────────────────────────────────────────
    # SUOLO / pH
    # ─────────────────────────────────────────────

    def _soil_recs(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        ph = sensor_data.get("ph")

        if ph is None:
            return recs

        if ph < SENSOR_CONFIG["ph_optimal_min"]:
            recs.append(self._rec(
                "suolo", 1,
                f"pH acido ({ph:.2f}): riduzione disponibilità Ca, Mg, P.",
                "Applicare calce dolomitica o bicarbonato di potassio "
                "per alzare il pH verso 6.0–7.0. "
                "Evitare fertilizzanti acidificanti.",
            ))
        elif ph > SENSOR_CONFIG["ph_optimal_max"]:
            recs.append(self._rec(
                "suolo", 1,
                f"pH alcalino ({ph:.2f}): carenza micronutrienti (Fe, Mn, Zn).",
                "Acidificare il substrato con zolfo elementare, "
                "acido fosforico diluito o fertilizzanti acidificanti. "
                "Target: pH 6.0–6.5.",
            ))
        else:
            recs.append(self._rec(
                "suolo", 5,
                f"pH nella norma ({ph:.2f}).",
                "Monitorare periodicamente. Nessun intervento necessario.",
            ))

        return recs

    # ─────────────────────────────────────────────
    # PATOLOGIE (da AI)
    # ─────────────────────────────────────────────

    def _disease_recs(self, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        ai_class = diagnosis.get("ai_class", "Sano")
        confidence = diagnosis.get("ai_confidence", 0.0)

        disease_map = {
            "Peronospora": (
                "Peronospora rilevata (oomicete).",
                "Applicare fungicida a base di rame o mancozeb. "
                "Ridurre umidità fogliare. Migliorare ventilazione. "
                "Evitare irrigazione fogliare serale.",
            ),
            "Oidio": (
                "Oidio rilevato (fungo powdery mildew).",
                "Trattare con zolfo micronizzato o bicarbonato di sodio (5%). "
                "Ridurre l'umidità relativa < 60%. "
                "Aumentare circolazione d'aria.",
            ),
            "Muffa_grigia": (
                "Muffa grigia (Botrytis cinerea) rilevata.",
                "Rimuovere e distruggere i tessuti infetti. "
                "Applicare fungicida specifico (es. iprodione, fludioxonil). "
                "Ridurre umidità sotto 70%.",
            ),
            "Alternaria": (
                "Alternariosi rilevata.",
                "Applicare fungicida a base di tebuconazolo o difenoconazolo. "
                "Rimuovere foglie infette. "
                "Evitare ristagni d'acqua.",
            ),
            "Ruggine": (
                "Ruggine fogliale rilevata.",
                "Applicare fungicida sistemico (triazoli). "
                "Rimuovere foglie infette. "
                "Evitare bagnatura fogliare prolungata.",
            ),
            "Mosaikovirus": (
                "Virus del mosaico rilevato (possibile).",
                "Non esistono cure per i virus vegetali. "
                "Isolare la pianta. Controllare e eliminare vettori (afidi, tripidi). "
                "Considerare la sostituzione della pianta.",
            ),
            "Stress_idrico": (
                "Stress idrico rilevato.",
                "Verificare substrato: reidratare gradualmente. "
                "Evitare irrigazioni eccessive improvvise dopo stress da siccità.",
            ),
        }

        for key, (problem, action) in disease_map.items():
            if key in ai_class:
                recs.append(self._rec(
                    "patologia", 1,
                    f"{problem} Confidenza AI: {confidence:.1f}%.",
                    action,
                ))
                break

        if diagnosis.get("needs_human_review", False):
            recs.append(self._rec(
                "revisione", 2,
                "Confidenza AI bassa: diagnosi visiva incerta.",
                "Si consiglia ispezione visiva manuale da parte di un agronomo. "
                "Fornire feedback al sistema per migliorare le future diagnosi.",
            ))

        return recs

    # ─────────────────────────────────────────────
    # FIORE
    # ─────────────────────────────────────────────

    def _flower_recs(
        self, diagnosis: Dict[str, Any], sensor_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        recs = []
        organ_analyses = diagnosis.get("organ_analyses", {})
        flower = organ_analyses.get("fiore")
        if not flower:
            return recs

        cls = flower.get("class", "Fiore_sano")
        conf = flower.get("confidence", 0.0) * 100

        flower_disease_map = {
            "Caduta_prematura_fiore": (
                "Caduta prematura dei fiori rilevata.",
                "Verificare irrigazione costante (evitare stress idrico). "
                "Ridurre l'apporto di azoto durante la fioritura (favorire P e K). "
                "Assicurare impollinazione adeguata (vibrazioni, api, vento).",
            ),
            "Aborto_floreale": (
                "Aborto floreale rilevato.",
                "Mantenere temperatura tra 15–30°C durante la fioritura. "
                "Evitare sbalzi termici notturni. "
                "Verificare disponibilità di boro (B): essenziale per sviluppo floreale.",
            ),
            "Mancata_allegagione": (
                "Mancata allegagione: fiori non si trasformano in frutti.",
                "Favorire impollinazione manuale con pennello morbido. "
                "Verificare presenza di impollinatori. "
                "Somministrare boro fogliare (0.2–0.5% borax) durante fioritura.",
            ),
            "Oidio_fiore": (
                f"Oidio sui fiori rilevato (confidenza: {conf:.1f}%).",
                "Applicare zolfo micronizzato evitando le ore calde. "
                "Ridurre umidità sotto 60%. "
                "Rimuovere i fiori infetti per prevenire la diffusione.",
            ),
            "Muffa_grigia_fiore": (
                f"Muffa grigia (Botrytis) sui fiori (confidenza: {conf:.1f}%).",
                "Rimuovere immediatamente i fiori infetti. "
                "Applicare fungicida specifico (iprodione o fludioxonil). "
                "Ridurre umidità e migliorare ventilazione urgentemente.",
            ),
            "Bruciatura_petali": (
                "Bruciatura dei petali da eccesso luminoso o temperature.",
                "Applicare ombreggiatura nelle ore di picco. "
                "Proteggere i fiori da vento caldo e siccità. "
                "Irrigare nelle ore mattutine.",
            ),
            "Deformazione_fiore": (
                "Deformazione floreale rilevata.",
                "Verificare presenza di acari eriofidi o tripidi. "
                "Applicare acaricida specifico. "
                "Controllare apporto di boro e calcio.",
            ),
        }

        for key, (problem, action) in flower_disease_map.items():
            if key in cls:
                recs.append(self._rec("fiore", 1, problem, action))
                break

        # Raccomandazione generale per fioritura in corso
        temp = sensor_data.get("temperature_c")
        hum  = sensor_data.get("humidity_pct")
        if temp is not None and (temp < 12 or temp > 34):
            recs.append(self._rec(
                "fiore", 1,
                f"Temperatura ({temp:.1f}°C) fuori dal range ottimale per la fioritura.",
                "Portare la temperatura tra 15–30°C durante la fioritura. "
                "Proteggere da gelate o caldo estremo con schermature appropriate.",
            ))
        if hum is not None and hum >= 80:
            recs.append(self._rec(
                "fiore", 1,
                f"Umidità elevata ({hum:.1f}%) durante la fioritura: rischio Botrytis.",
                "Migliorare immediatamente la ventilazione. "
                "Ridurre irrigazione. "
                "Applicare trattamento preventivo con prodotti a base di rame.",
            ))

        return recs

    # ─────────────────────────────────────────────
    # FRUTTO
    # ─────────────────────────────────────────────

    def _fruit_recs(
        self, diagnosis: Dict[str, Any], sensor_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        recs = []
        organ_analyses = diagnosis.get("organ_analyses", {})
        fruit = organ_analyses.get("frutto")
        if not fruit:
            return recs

        cls  = fruit.get("class", "Frutto_sano")
        conf = fruit.get("confidence", 0.0) * 100

        fruit_disease_map = {
            "Marciume_apicale": (
                "Marciume apicale del frutto rilevato (carenza calcio).",
                "Applicare calcio nitrato fogliare (0.5–1%). "
                "Mantenere irrigazione uniforme senza stress idrico. "
                "Correggere pH tra 6.0–6.5 per ottimizzare assorbimento Ca.",
            ),
            "Spaccatura_frutto": (
                f"Spaccatura del frutto rilevata (confidenza: {conf:.1f}%).",
                "Garantire irrigazione regolare e costante (evitare cicli asciutto/umido). "
                "Pacciamatura per mantenere umidità del suolo stabile. "
                "Raccogliere i frutti spaccati per prevenire diffusione di patogeni.",
            ),
            "Scottatura_solare": (
                "Scottatura solare del frutto rilevata.",
                "Applicare ombreggiatura nelle ore di massima insolazione (12–15h). "
                "Considerare caolino (argilla bianca) per protezione UV naturale. "
                "Mantenere fogliame protettivo intorno ai frutti.",
            ),
            "Muffa_grigia_frutto": (
                f"Muffa grigia (Botrytis) sul frutto (confidenza: {conf:.1f}%).",
                "Raccogliere e distruggere i frutti infetti. "
                "Applicare fungicida specifico (fludioxonil, iprodione). "
                "Ridurre umidità ambientale urgentemente.",
            ),
            "Alternaria_frutto": (
                "Alternariosi del frutto rilevata.",
                "Applicare fungicida a base di tebuconazolo. "
                "Rimuovere i frutti infetti. "
                "Migliorare drenaggio e ridurre ristagni.",
            ),
            "Rugginosità": (
                "Rugginosità della buccia rilevata.",
                "Verificare la presenza di acari rust mite (Phyllocoptruta oleivora). "
                "Applicare zolfo o acaricida specifico. "
                "Monitorare con lente d'ingrandimento.",
            ),
            "Carenza_calcio_frutto": (
                "Carenza di calcio nel frutto rilevata.",
                "Trattamento urgente con calcio cloruro (0.5%) per via fogliare. "
                "Verificare pH del substrato (6.0–6.5 ottimale per assorbimento Ca). "
                "Uniformare l'irrigazione per prevenire blocchi di assorbimento.",
            ),
        }

        for key, (problem, action) in fruit_disease_map.items():
            if key in cls:
                recs.append(self._rec("frutto", 1, problem, action))
                break

        # Raccomandazioni generali maturazione frutto
        temp = sensor_data.get("temperature_c")
        light = sensor_data.get("light_lux")
        if temp is not None and temp > 35:
            recs.append(self._rec(
                "frutto", 2,
                f"Temperatura eccessiva ({temp:.1f}°C) durante maturazione.",
                "Applicare ombreggiatura. "
                "Irrigare nelle prime ore del mattino per abbassare temperatura fogliare. "
                "Valutare coperture anticalura.",
            ))
        if light is not None and light > 90000:
            recs.append(self._rec(
                "frutto", 2,
                f"Radiazione solare intensa ({light:.0f} lux): rischio scottature.",
                "Applicare rete ombreggiante al 30–40%. "
                "Trattamento con caolino per riflettere la radiazione solare.",
            ))

        return recs

    # ─────────────────────────────────────────────
    # QUANTUM RISK
    # ─────────────────────────────────────────────

    def _quantum_risk_recs(self, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera raccomandazioni basate sul Quantum Risk Score di Grover."""
        recs = []
        qr = diagnosis.get("quantum_risk", {})
        if not qr:
            return recs

        qrs  = qr.get("quantum_risk_score", 0.0)
        qlvl = qr.get("risk_level", "nessuno")
        dom  = qr.get("dominant_description", "")
        gain = qr.get("amplification_gain", 1.0)
        adv  = qr.get("adverse_states", [])

        if qlvl == "critico":
            recs.append(self._rec(
                "quantum_risk", 1,
                f"[QUANTUM CRITICO] QRS={qrs:.3f} — Evento avverso dominante: {dom}. "
                f"Amplificazione di Grover: {gain:.1f}x. "
                f"Numero stati avversi amplificati: {len(adv)}.",
                "INTERVENTO URGENTE RICHIESTO. "
                "Tutti i fattori di rischio stanno interagendo in modo sinergico. "
                "Consultare immediatamente un agronomo specializzato. "
                "Isolare la coltura dalle piante sane. "
                "Documentare con fotografie per analisi retrospettiva.",
            ))
        elif qlvl == "alto":
            recs.append(self._rec(
                "quantum_risk", 2,
                f"[QUANTUM ALTO] QRS={qrs:.3f} — Evento critico: {dom}.",
                "Aumentare la frequenza di monitoraggio a ogni 4–6 ore. "
                "Intervenire sulle criticità principali entro 24 ore. "
                "Prepararsi a trattamenti preventivi.",
            ))
        elif qlvl == "medio":
            recs.append(self._rec(
                "quantum_risk", 3,
                f"[QUANTUM MEDIO] QRS={qrs:.3f} — Rischio contenuto: {dom}.",
                "Monitorare la situazione nelle prossime 48 ore. "
                "Correggere i parametri fuori range identificati dall'oracolo.",
            ))

        return recs

    # ─────────────────────────────────────────────
    # FACTORY
    # ─────────────────────────────────────────────

    @staticmethod
    def _rec(category: str, priority: int, problem: str, action: str) -> Dict[str, Any]:
        """Costruisce un dict raccomandazione standardizzato."""
        return {
            "category": category,
            "priority": priority,
            "problem": problem,
            "action": action,
        }
