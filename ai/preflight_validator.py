"""
Validazione pre-avvio artefatti modello: modello, labels, immagine test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from ai.tflite_inference_runner import (
    resolve_path,
    load_labels,
    make_interpreter,
    preprocess_image,
    run_inference,
    decode_prediction,
)

LOGGER = logging.getLogger("delta.ai.preflight")


class PreflightGateError(RuntimeError):
    """Sollevata quando la confidenza del modello non supera la soglia minima di deploy."""


def validate_model_artifacts(
    model_path: str,
    labels_path: str,
    image_path: str,
    threads: int = 4,
    top_k: int = 3,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Esegue una validazione end-to-end degli artefatti AI.

    Solleva PreflightGateError se la confidenza e inferiore a min_confidence.
    Solleva eccezioni in caso di errore critico.
    Restituisce un report sintetico in caso di successo.
    """
    model = resolve_path(model_path)
    labels = resolve_path(labels_path)
    image = resolve_path(image_path)

    LOGGER.info("Preflight model:  %s", model)
    LOGGER.info("Preflight labels: %s", labels)
    LOGGER.info("Preflight image:  %s", image)

    class_names = load_labels(labels)
    interpreter, input_details, output_details = make_interpreter(model, num_threads=threads)

    input_shape = tuple(int(v) for v in input_details[0]["shape"][1:4])
    input_dtype = input_details[0]["dtype"]

    image_tensor = preprocess_image(Path(image), input_shape, input_dtype)
    probs = run_inference(interpreter, input_details, output_details, image_tensor)
    result = decode_prediction(probs, class_names, top_k=top_k)

    report = {
        "ready": True,
        "model_path": str(model),
        "labels_path": str(labels),
        "image_path": str(image),
        "input_shape": tuple(int(v) for v in input_details[0]["shape"]),
        "output_shape": tuple(int(v) for v in output_details[0]["shape"]),
        "predicted_class": result["class"],
        "confidence": result["confidence"],
        "top_k": result["top_k"],
    }

    if min_confidence > 0.0 and result["confidence"] < min_confidence:
        raise PreflightGateError(
            f"Deploy bloccato: confidenza {result['confidence']:.2%} < soglia minima "
            f"{min_confidence:.2%} (classe predetta: '{result['class']}'). "
            "Verificare la qualita del dataset e ripetere training → conversione → preflight."
        )

    return report

