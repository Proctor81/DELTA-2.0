"""
DELTA - ai/quantum_risk.py
Oracolo Quantistico di Grover per la quantificazione del rischio.

Implementa una simulazione classica dell'algoritmo di Grover:
- Registro quantistico di n qubit (2^n stati di rischio)
- Oracolo che marca gli stati avversi (rischi attivi)
- Operatore di diffusione di Grover (amplificazione ampiezza)
- Misura: estrae la distribuzione di probabilità degli eventi avversi

La simulazione è esatta (non approssimata): ogni amplitudine è un
numero complesso e le operazioni preservano la norma unitaria.
Il vantaggio rispetto all'approccio classico è l'amplificazione
quadratica: O(√N) iterazioni trovano lo stato avverso dominante
dove un approccio lineare richiederebbe O(N).
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from core.config import QUANTUM_CONFIG

logger = logging.getLogger("delta.ai.quantum_risk")


# ─────────────────────────────────────────────
# MAPPA RISCHI ↔ STATI QUANTISTICI
# ─────────────────────────────────────────────

# Mappa regola_id → stato quantistico (indice nel registro)
# Le regole nuove (FLOW_03/04, FRUT_03/04/05) condividono stati con le primarie
RULE_TO_STATE: Dict[str, int] = {
    "FUNG_01":   0,    # Rischio fungino        → |0000⟩
    "PHOTO_01":  1,    # Stress luminoso basso  → |0001⟩
    "PHOTO_02":  2,    # Stress luminoso alto   → |0010⟩
    "CO2_01":    3,    # Carenza CO2            → |0011⟩
    "CO2_02":    4,    # Eccesso CO2            → |0100⟩
    "PH_01":     5,    # pH acido               → |0101⟩
    "PH_02":     6,    # pH alcalino            → |0110⟩
    "EC_01":     7,    # EC tossica             → |0111⟩
    "EC_02":     8,    # EC bassa               → |1000⟩
    "TEMP_01":   9,    # Stress termico         → |1001⟩
    "AI_01":    10,    # Malattia AI-detected   → |1010⟩
    "AI_02":    11,    # Bassa confidenza AI    → |1011⟩
    "FLOW_01":  12,    # Aborto floreale        → |1100⟩
    "FLOW_02":  12,    # Caduta fiore           → |1100⟩ (condivide stato flower)
    "FLOW_03":  12,    # Malattia fiore AI      → |1100⟩
    "FLOW_04":  13,    # Botrytis fiore CRITICO → |1101⟩
    "FRUT_01":  14,    # Marciume frutto        → |1110⟩
    "FRUT_02":  15,    # Spaccatura frutto      → |1111⟩
    "FRUT_03":  14,    # Carenza Ca frutto      → |1110⟩ (condivide)
    "FRUT_04":  15,    # Scottatura frutto      → |1111⟩ (condivide)
    "FRUT_05":  14,    # Malattia frutto AI     → |1110⟩ (condivide)
}

# Mappa inversa
STATE_TO_RULE: Dict[int, str] = {v: k for k, v in RULE_TO_STATE.items()}

# Descrizioni degli stati
STATE_DESCRIPTIONS: Dict[int, str] = {
    0:  "Rischio fungino (alta umidità + temperatura)",
    1:  "Stress da carenza luminosa",
    2:  "Foto-inibizione (eccesso luce)",
    3:  "Carenza CO₂",
    4:  "Eccesso CO₂",
    5:  "pH acido (blocco nutrienti)",
    6:  "pH alcalino (carenza micronutrienti)",
    7:  "Tossicità salina (EC elevata)",
    8:  "Carenza nutrienti (EC bassa)",
    9:  "Stress termico",
    10: "Malattia rilevata da visione AI",
    11: "Diagnosi AI incerta (bassa confidenza)",
    12: "Aborto floreale",
    13: "Caduta prematura dei fiori",
    14: "Marciume del frutto",
    15: "Spaccatura/scottatura frutto",
}


# ─────────────────────────────────────────────
# ORACOLO DI GROVER
# ─────────────────────────────────────────────

class GroverRiskOracle:
    """
    Oracolo Quantistico di Grover per la quantificazione del rischio agronomico.

    Algoritmo:
    1. Preparazione: stato uniforme |ψ⟩ = Σ|i⟩/√N
    2. Oracolo U_f: inverte il segno delle ampiezze degli stati avversi
       U_f|i⟩ = -|i⟩  se i è uno stato avverso
       U_f|i⟩ =  |i⟩  altrimenti
    3. Diffusore di Grover: U_s = 2|ψ⟩⟨ψ| - I
       Amplifica le ampiezze degli stati marcati
    4. Ripetizione: O(π/4·√(N/M)) iterazioni (M = n° stati avversi)
    5. Misura: distribuzioni di probabilità → quantum risk score
    """

    def __init__(self):
        self.n_qubits = QUANTUM_CONFIG["n_qubits"]
        self.n_states = 2 ** self.n_qubits
        self.grover_iterations = QUANTUM_CONFIG["grover_iterations"]
        self.risk_weights = QUANTUM_CONFIG["risk_weights"]
        logger.info(
            "GroverRiskOracle inizializzato: %d qubit, %d stati, %d iterazioni.",
            self.n_qubits, self.n_states, self.grover_iterations,
        )

    # ─────────────────────────────────────────────
    # API PRINCIPALE
    # ─────────────────────────────────────────────

    def quantify_risk(
        self,
        activated_rule_ids: List[str],
        organ_analyses: Optional[Dict[str, Any]] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Esegue l'algoritmo di Grover per quantificare il rischio composito.

        Args:
            activated_rule_ids: lista delle rule_id attivate dalla diagnosi
            organ_analyses:     risultati analisi fiore/frutto (facoltativo)
            sensor_data:        dati sensori correnti (facoltativo)

        Returns:
            dict con:
              - quantum_risk_score:   [0,1] rischio quantistico globale
              - risk_level:           etichetta (nessuno/basso/medio/alto/critico)
              - dominant_state:       stato avverso dominante dopo amplificazione
              - dominant_description: descrizione dello stato dominante
              - probability_vector:   vettore probabilità completo (16 stati)
              - grover_iterations:    iterazioni effettivamente eseguite
              - adverse_states:       stati avversi marcati dall'oracolo
              - amplification_gain:   guadagno amplificazione rispetto al classico
        """
        logger.debug("Grover: regole attivate = %s", activated_rule_ids)

        # ── 1. Identifica stati avversi ───────────────────────
        adverse_states = self._map_rules_to_states(activated_rule_ids, organ_analyses)

        if not adverse_states:
            return self._zero_risk_result()

        # ── 2. Numero iterazioni ottimale ─────────────────────
        m = len(adverse_states)
        n = self.n_states
        optimal_iters = max(1, round((math.pi / 4) * math.sqrt(n / m)))
        iters = min(optimal_iters, self.grover_iterations)

        # ── 3. Stato iniziale: superposizione uniforme ─────────
        state = self._initialize_superposition()

        # ── 4. Amplificazione di Grover ───────────────────────
        for _ in range(iters):
            state = self._apply_oracle(state, adverse_states)
            state = self._apply_diffusion(state)

        # ── 5. Distribuzione di probabilità ───────────────────
        prob_vector = np.abs(state) ** 2
        prob_vector = prob_vector / prob_vector.sum()  # normalizzazione numerica

        # ── 6. Stato dominante ────────────────────────────────
        dominant_state = int(np.argmax(prob_vector))
        dominant_prob  = float(prob_vector[dominant_state])

        # ── 7. Quantum risk score ─────────────────────────────
        # Somma pesata delle probabilità degli stati avversi
        qrs = self._compute_quantum_risk_score(prob_vector, adverse_states, sensor_data)

        # ── 8. Guadagno di amplificazione ─────────────────────
        classical_prob = m / n
        amplification_gain = dominant_prob / classical_prob if classical_prob > 0 else 1.0

        result = {
            "quantum_risk_score": round(qrs, 4),
            "risk_level": self._score_to_level(qrs),
            "dominant_state": dominant_state,
            "dominant_description": STATE_DESCRIPTIONS.get(dominant_state, "Stato sconosciuto"),
            "probability_vector": [round(float(p), 4) for p in prob_vector],
            "grover_iterations": iters,
            "adverse_states": adverse_states,
            "amplification_gain": round(amplification_gain, 2),
        }

        logger.info(
            "Grover: QRS=%.3f, livello=%s, stato_dom=%d (%s), gain=%.2fx",
            qrs, result["risk_level"], dominant_state,
            result["dominant_description"], amplification_gain,
        )
        return result

    # ─────────────────────────────────────────────
    # STEP 1: MAPPA REGOLE → STATI
    # ─────────────────────────────────────────────

    def _map_rules_to_states(
        self,
        rule_ids: List[str],
        organ_analyses: Optional[Dict[str, Any]],
    ) -> List[int]:
        """Converte rule_id in indici di stato del registro quantistico."""
        states = set()

        for rid in rule_ids:
            if rid in RULE_TO_STATE:
                states.add(RULE_TO_STATE[rid])

        # Aggiungi stati per analisi organo-specifiche
        if organ_analyses:
            flower_result = organ_analyses.get("fiore")
            fruit_result  = organ_analyses.get("frutto")

            if flower_result and isinstance(flower_result, dict):
                cls = flower_result.get("class", "")
                if cls and cls != "Fiore_sano":
                    if "Caduta" in cls or "caduta" in cls:
                        states.add(RULE_TO_STATE["FLOW_02"])
                    elif "Aborto" in cls or "Mancata" in cls:
                        states.add(RULE_TO_STATE["FLOW_01"])
                    else:
                        states.add(RULE_TO_STATE["FLOW_01"])

            if fruit_result and isinstance(fruit_result, dict):
                cls = fruit_result.get("class", "")
                if cls and cls != "Frutto_sano":
                    if "Marciume" in cls or "Muffa" in cls:
                        states.add(RULE_TO_STATE["FRUT_01"])
                    elif "Spaccatura" in cls or "Scottatura" in cls:
                        states.add(RULE_TO_STATE["FRUT_02"])
                    else:
                        states.add(RULE_TO_STATE["FRUT_01"])

        return sorted(states)

    # ─────────────────────────────────────────────
    # STEP 2: SUPERPOSIZIONE INIZIALE
    # ─────────────────────────────────────────────

    def _initialize_superposition(self) -> np.ndarray:
        """Crea lo stato di superposizione uniforme |ψ⟩ = Σ|i⟩/√N."""
        amplitude = 1.0 / math.sqrt(self.n_states)
        return np.full(self.n_states, amplitude, dtype=complex)

    # ─────────────────────────────────────────────
    # STEP 3: ORACOLO U_f
    # ─────────────────────────────────────────────

    @staticmethod
    def _apply_oracle(state: np.ndarray, adverse_states: List[int]) -> np.ndarray:
        """
        Applica l'oracolo di Grover: inverte il segno delle ampiezze
        degli stati avversi (phase flip).
        """
        new_state = state.copy()
        for s in adverse_states:
            new_state[s] *= -1
        return new_state

    # ─────────────────────────────────────────────
    # STEP 4: DIFFUSORE DI GROVER U_s
    # ─────────────────────────────────────────────

    @staticmethod
    def _apply_diffusion(state: np.ndarray) -> np.ndarray:
        """
        Applica il diffusore di Grover: U_s = 2|ψ⟩⟨ψ| - I
        Equivale a: new_state = 2·mean(state) - state
        """
        mean_amplitude = np.mean(state)
        return 2 * mean_amplitude - state

    # ─────────────────────────────────────────────
    # STEP 5: CALCOLO QUANTUM RISK SCORE
    # ─────────────────────────────────────────────

    def _compute_quantum_risk_score(
        self,
        prob_vector: np.ndarray,
        adverse_states: List[int],
        sensor_data: Optional[Dict[str, Any]],
    ) -> float:
        """
        Calcola il Quantum Risk Score (QRS) come somma pesata delle
        probabilità degli stati avversi dopo amplificazione di Grover.

        Formula:
            QRS = Σ_i [ P_Grover(i) · w(i) ]  per i ∈ adverse_states
        """
        weights = self.risk_weights

        # Peso per ogni stato avverso in base alla categoria
        state_weights: Dict[int, float] = {
            RULE_TO_STATE["FUNG_01"]:  weights.get("fungal", 0.8),
            RULE_TO_STATE["PHOTO_01"]: weights.get("light_stress", 0.4),
            RULE_TO_STATE["PHOTO_02"]: weights.get("light_stress", 0.4),
            RULE_TO_STATE["CO2_01"]:   weights.get("environmental", 0.4),
            RULE_TO_STATE["CO2_02"]:   weights.get("environmental", 0.4),
            RULE_TO_STATE["PH_01"]:    weights.get("ph_imbalance", 0.6),
            RULE_TO_STATE["PH_02"]:    weights.get("ph_imbalance", 0.6),
            RULE_TO_STATE["EC_01"]:    weights.get("salinity", 0.9),
            RULE_TO_STATE["EC_02"]:    weights.get("nutrient", 0.5),
            RULE_TO_STATE["TEMP_01"]:  weights.get("temp_stress", 0.55),
            RULE_TO_STATE["AI_01"]:    weights.get("ai_detection", 0.85),
            RULE_TO_STATE["AI_02"]:    weights.get("ai_detection", 0.5),
            12: weights.get("flower_abort", 0.75),    # stati flower (12 condiviso)
            13: weights.get("flower_abort", 0.9),     # stato Botrytis fiore (CRITICO)
            14: weights.get("fruit_rot", 0.85),       # stati frutto marciume (14 condiviso)
            15: weights.get("fruit_crack", 0.65),     # stati frutto spaccatura (15 condiviso)
        }

        qrs = 0.0
        for state_idx in adverse_states:
            p = float(prob_vector[state_idx])
            w = state_weights.get(state_idx, 0.5)
            qrs += p * w

        # Bonus composto: se ci sono ≥3 stati avversi, il rischio è sinergico
        if len(adverse_states) >= 3:
            compound_w = weights.get("compound", 0.95)
            qrs = qrs * (1 + 0.1 * compound_w * (len(adverse_states) - 2))

        return min(1.0, qrs)

    # ─────────────────────────────────────────────
    # UTILITÀ
    # ─────────────────────────────────────────────

    @staticmethod
    def _score_to_level(score: float) -> str:
        cfg = QUANTUM_CONFIG
        if score >= cfg["risk_threshold_critical"]:
            return "critico"
        elif score >= cfg["risk_threshold_high"]:
            return "alto"
        elif score >= cfg["risk_threshold_medium"]:
            return "medio"
        elif score >= cfg["risk_threshold_low"]:
            return "basso"
        else:
            return "nessuno"

    @staticmethod
    def _zero_risk_result() -> Dict[str, Any]:
        return {
            "quantum_risk_score": 0.0,
            "risk_level": "nessuno",
            "dominant_state": -1,
            "dominant_description": "Nessun rischio avverso identificato",
            "probability_vector": [],
            "grover_iterations": 0,
            "adverse_states": [],
            "amplification_gain": 1.0,
        }
