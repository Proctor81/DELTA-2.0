"""
DELTA - ai/download_pretrained_model.py

Scarica il dataset PlantVillage tramite tensorflow_datasets,
addestra MobileNetV2 (ImageNet weights) sulle classi DELTA,
e converte il modello in TFLite INT8.

Non richiede account Kaggle.

Utilizzo:
    python ai/download_pretrained_model.py
    python ai/download_pretrained_model.py --epochs 10 --output models
    python ai/download_pretrained_model.py --no-quantize   # salva solo .keras
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Workaround: protobuf >=5 rimuove FieldDescriptor.label nell'implementazione C;
# forzare la versione Python garantisce compatibilità con tensorflow-datasets 4.x.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

LOGGER = logging.getLogger("delta.ai.download_pretrained")

# ─────────────────────────────────────────────────────────────
# CLASSI DELTA (ordine fisso → indice usato dal modello)
# ─────────────────────────────────────────────────────────────
DELTA_LABELS: list[str] = [
    "Sano",
    "Peronospora",
    "Oidio",
    "Muffa_grigia",
    "Alternaria",
    "Ruggine",
    "Mosaikovirus",
]

# ─────────────────────────────────────────────────────────────
# MAPPATURA PlantVillage (38 classi) → DELTA
# La chiave è una sottostringa del nome classe PlantVillage
# (case-insensitive). Valore None = campione scartato.
# ─────────────────────────────────────────────────────────────
_PV_TO_DELTA: dict[str, str | None] = {
    "healthy":            "Sano",
    "late_blight":        "Peronospora",
    "leaf_blight":        "Peronospora",
    "esca":               "Peronospora",
    "powdery_mildew":     "Oidio",
    "leaf_mold":          "Muffa_grigia",
    "early_blight":       "Alternaria",
    "cercospora":         "Alternaria",
    "northern_leaf":      "Alternaria",
    "apple_scab":         "Alternaria",
    "target_spot":        "Alternaria",
    "septoria":           "Alternaria",
    "leaf_scorch":        "Alternaria",
    "bacterial_spot":     "Alternaria",
    "black_rot":          "Alternaria",
    "rust":               "Ruggine",
    "mosaic":             "Mosaikovirus",
    "yellow_leaf_curl":   "Mosaikovirus",
    # classi senza equivalente DELTA → scartate
    "spider_mites":       None,
    "haunglongbing":      None,
}


def _map_pv_label(pv_name: str) -> str | None:
    """Restituisce la classe DELTA corrispondente a una classe PlantVillage."""
    name = pv_name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for key, delta in _PV_TO_DELTA.items():
        if key in name:
            return delta
    return None


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scarica PlantVillage, addestra MobileNetV2, converte TFLite INT8"
    )
    p.add_argument("--output", default="models", help="Directory output (default: models)")
    p.add_argument("--epochs", type=int, default=8,
                   help="Epoche totali di training (default: 8; prime 3 solo head, resto fine-tuning)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--img-size", type=int, default=224, help="Dimensione immagini (default: 224)")
    p.add_argument("--fine-tune-layers", type=int, default=30,
                   help="Numero layer del backbone da sbloccare nel fine-tuning (default: 30)")
    p.add_argument("--data-dir", default=None,
                   help="Directory cache tensorflow_datasets (default: ~/tensorflow_datasets)")
    p.add_argument("--no-quantize", action="store_true",
                   help="Salta conversione TFLite INT8, salva solo .keras")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


# ─────────────────────────────────────────────────────────────
# CONTROLLO DIPENDENZE
# ─────────────────────────────────────────────────────────────

def check_dependencies() -> None:
    missing: list[str] = []
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        missing.append("tensorflow>=2.13.0")
    try:
        import tensorflow_datasets  # noqa: F401
    except ImportError:
        missing.append("tensorflow-datasets")
    if missing:
        pkgs = " ".join(missing)
        print(f"\n[ERRORE] Pacchetti mancanti: {', '.join(missing)}")
        print(f"Installare con:\n  pip install {pkgs}\n")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# CARICAMENTO DATASET
# ─────────────────────────────────────────────────────────────

def load_plantvillage(data_dir: str | None, img_size: int, batch_size: int):
    """
    Scarica (o usa cache) PlantVillage via tensorflow_datasets.
    Restituisce (train_ds, val_ds, num_classes).
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    LOGGER.info("Scaricamento dataset PlantVillage (~820 MB — solo prima volta)...")
    (train_raw, val_raw), info = tfds.load(
        "plant_village",
        split=["train[:80%]", "train[80%:]"],
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
    )

    pv_label_names: list[str] = info.features["label"].names
    LOGGER.info("Classi PlantVillage totali: %d", len(pv_label_names))

    # Tabella di mapping: indice_pv → indice_delta  (-1 = scartare)
    delta_idx = {lbl: i for i, lbl in enumerate(DELTA_LABELS)}
    pv_to_delta: list[int] = []
    for name in pv_label_names:
        mapped = _map_pv_label(name)
        pv_to_delta.append(delta_idx[mapped] if mapped and mapped in delta_idx else -1)

    covered = len({i for i in pv_to_delta if i >= 0})
    LOGGER.info("Classi DELTA coperte da PlantVillage: %d / %d", covered, len(DELTA_LABELS))

    mapping_tensor = tf.constant(pv_to_delta, dtype=tf.int32)
    num_classes = len(DELTA_LABELS)
    AUTOTUNE = tf.data.AUTOTUNE

    @tf.function
    def preprocess(image, label):
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        delta_label = mapping_tensor[label]
        return image, delta_label

    @tf.function
    def is_mapped(image, label):  # pylint: disable=unused-argument
        return label >= 0

    # Stima steps per epoch: campioni mappati / batch_size
    # PlantVillage ha ~54303 totali; le classi scartate sono ~5% → ~51588 mappati
    n_train = int(info.splits["train"].num_examples * 0.80 * 0.95)
    n_val   = int(info.splits["train"].num_examples * 0.20 * 0.95)
    steps_per_epoch  = max(1, n_train // batch_size)
    validation_steps = max(1, n_val   // batch_size)
    LOGGER.info("steps_per_epoch=%d  validation_steps=%d", steps_per_epoch, validation_steps)

    train_ds = (
        train_raw
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .filter(is_mapped)
        .shuffle(2048, seed=42)
        .repeat()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_raw
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .filter(is_mapped)
        .repeat()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds, num_classes, steps_per_epoch, validation_steps


# ─────────────────────────────────────────────────────────────
# COSTRUZIONE MODELLO
# ─────────────────────────────────────────────────────────────

def build_model(num_classes: int, img_size: int):
    """MobileNetV2 con head di classificazione per le classi DELTA."""
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # congelato per fase 1

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    # MobileNetV2 si aspetta [-1, 1]; il nostro input è [0, 1] * 255 -> preprocess_input
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_model(
    model,
    base,
    train_ds,
    val_ds,
    epochs: int,
    fine_tune_layers: int,
    keras_path: Path,
    steps_per_epoch: int,
    validation_steps: int,
) -> None:
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(keras_path),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ── Fase 1: training solo head ──────────────────────────
    head_epochs = min(3, epochs)
    LOGGER.info("Fase 1 — training head (%d epoche)...", head_epochs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    # ── Fase 2: fine-tuning ultimi N layer backbone ─────────
    remaining = epochs - head_epochs
    if fine_tune_layers > 0 and remaining > 0:
        LOGGER.info(
            "Fase 2 — fine-tuning ultimi %d layer del backbone (%d epoche)...",
            fine_tune_layers,
            remaining,
        )
        base.trainable = True
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
        )

    LOGGER.info("Modello Keras salvato: %s", keras_path)


# ─────────────────────────────────────────────────────────────
# CONVERSIONE TFLITE INT8
# ─────────────────────────────────────────────────────────────

def convert_to_tflite(keras_path: Path, tflite_path: Path, train_ds, img_size: int) -> None:
    import numpy as np
    import tensorflow as tf

    LOGGER.info("Conversione TFLite INT8 in corso...")
    model = tf.keras.models.load_model(str(keras_path))

    # Raccoglie campioni rappresentativi dal training set
    samples: list = []
    for images, _ in train_ds.take(15):
        for img in images.numpy():
            samples.append(img)
            if len(samples) >= 200:
                break
        if len(samples) >= 200:
            break

    LOGGER.info("Campioni representative dataset: %d", len(samples))

    def representative_dataset():
        for img in samples:
            yield [np.expand_dims(img.astype(np.float32), axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # Permette fallback su ops float per massima compatibilità
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    LOGGER.info("TFLite INT8 salvato: %s (%.2f MB)", tflite_path, size_mb)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    check_dependencies()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = output_dir / "plant_disease_model.keras"
    tflite_path = output_dir / "plant_disease_model.tflite"
    labels_path = output_dir / "labels.txt"

    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Classi DELTA target: %s", ", ".join(DELTA_LABELS))

    # 1. Dataset
    train_ds, val_ds, num_classes, steps_per_epoch, validation_steps = load_plantvillage(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # 2. Modello
    model, base = build_model(num_classes=num_classes, img_size=args.img_size)

    # 3. Training
    train_model(
        model, base, train_ds, val_ds,
        epochs=args.epochs,
        fine_tune_layers=args.fine_tune_layers,
        keras_path=keras_path,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    # 4. Labels
    labels_path.write_text("\n".join(DELTA_LABELS) + "\n", encoding="utf-8")
    LOGGER.info("labels.txt salvato: %s", labels_path)

    # 5. Conversione TFLite INT8
    if not args.no_quantize:
        convert_to_tflite(keras_path, tflite_path, train_ds, args.img_size)

    print("\n" + "=" * 50)
    print("  COMPLETATO")
    print("=" * 50)
    print(f"  Modello Keras  : {keras_path}")
    if not args.no_quantize:
        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        print(f"  Modello TFLite : {tflite_path}  ({size_mb:.1f} MB)")
    print(f"  Labels         : {labels_path}")
    print(f"  Classi ({num_classes})    : {', '.join(DELTA_LABELS)}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
