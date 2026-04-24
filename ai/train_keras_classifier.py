"""
Training classificatore immagini piante con TensorFlow/Keras.

Struttura dataset attesa:
    datasets/training/
        ClasseA/
            img1.jpg
            img2.jpg
        ClasseB/
            ...

Output:
    - models/plant_disease_model.keras
    - models/labels.txt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


LOGGER = logging.getLogger("delta.ai.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training Keras per classificazione malattie piante")
    parser.add_argument("--dataset", default="datasets/training", help="Directory dataset (classi per cartella)")
    parser.add_argument("--output", default="models", help="Directory output modello")
    parser.add_argument("--model-name", default="plant_disease_model.keras", help="Nome file modello Keras")
    parser.add_argument("--img-size", type=int, default=224, help="Dimensione immagini quadrate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Numero epoche")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Frazione validation set")
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate ottimizzatore")
    parser.add_argument("--fine-tune-layers", type=int, default=30, help="Numero layer finali da sbloccare")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def ensure_dataset(dataset_dir: Path):
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise RuntimeError(f"Dataset non trovato: {dataset_dir}")

    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if len(class_dirs) < 2:
        raise RuntimeError(
            "Dataset insufficiente: servono almeno 2 classi (cartelle) con immagini."
        )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / args.model_name
    labels_path = output_dir / "labels.txt"
    metadata_path = output_dir / "training_metadata.json"

    ensure_dataset(dataset_dir)

    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow non installato. Installare con: pip install tensorflow"
        ) from exc

    LOGGER.info("Dataset: %s", dataset_dir)
    LOGGER.info("Output modello: %s", model_path)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
    )

    class_names = list(train_ds.class_names)
    n_classes = len(class_names)
    LOGGER.info("Classi rilevate (%d): %s", n_classes, class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(args.img_size, args.img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3), name="image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="disease_probs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="delta_plant_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    LOGGER.info("Fase 1/2: addestramento testa classificatore")
    model.fit(train_ds, validation_data=val_ds, epochs=max(1, args.epochs // 2), callbacks=callbacks)

    if args.fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-args.fine_tune_layers]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate * 0.1),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        LOGGER.info("Fase 2/2: fine-tuning ultimi %d layer", args.fine_tune_layers)
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, initial_epoch=max(1, args.epochs // 2), callbacks=callbacks)

    loss, acc = model.evaluate(val_ds, verbose=0)
    LOGGER.info("Validation: loss=%.4f accuracy=%.4f", float(loss), float(acc))

    model.save(model_path)
    labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    metadata = {
        "dataset": str(dataset_dir),
        "model_path": str(model_path),
        "labels_path": str(labels_path),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "validation_split": args.validation_split,
        "learning_rate": args.learning_rate,
        "fine_tune_layers": args.fine_tune_layers,
        "n_classes": n_classes,
        "class_names": class_names,
        "val_loss": float(loss),
        "val_accuracy": float(acc),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== TRAINING COMPLETATO ===")
    print(f"Modello Keras: {model_path}")
    print(f"Labels:        {labels_path}")
    print(f"Val accuracy:  {acc * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
