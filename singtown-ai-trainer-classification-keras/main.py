from singtown_ai import SingTownAIClient, MOCK_TRAIN_CLASSIFICATION, stdout_watcher, file_watcher
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from zipfile import ZipFile

RUN_PATH = Path("run")
DATASET_PATH = Path("dataset")
METRICS_PATH = RUN_PATH / "metrics.csv"
BEST_KERAS_PATH = RUN_PATH / "best.keras"
RUN_PATH.mkdir(parents=True, exist_ok=True)

# mock_data=MOCK_TRAIN_CLASSIFICATION
# mock_data['task'] = {
#     "project": {
#         "labels": ["cat", "dog"],
#         "type": "CLASSIFICATION",
#     },
#     "device": "openmv-cam-h7-plus",
#     "model_name": "mobilenet_v2_0.35_128",
#     "freeze_backbone": True,
#     "batch_size": 16,
#     "epochs": 1,
#     "learning_rate": 0.001,
#     "early_stopping": 3,
#     "export_width": 128,
#     "export_height": 128,
# }

client = SingTownAIClient(
    # mock_data=mock_data,
)


@stdout_watcher(interval=1)
def on_stdout_write(content: str):
    client.log(content, end="")

@file_watcher(METRICS_PATH, interval=3)
def file_on_change(content: str):
    import csv
    from io import StringIO

    metrics = list(csv.DictReader(StringIO(content)))
    if not metrics:
        return
    client.update_metrics(metrics)

data_augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def build_dataset():
    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_PATH/"TRAIN",
        shuffle=True,
        image_size=(IMG_SZ, IMG_SZ),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=LABELS,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_PATH/"VALID",
        image_size=(IMG_SZ, IMG_SZ),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=LABELS,
    )
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img) / 127.5 - 1, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda img, label: (img / 127.5 - 1, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return train_ds, val_ds

def build_model():
    input_shape = (IMG_SZ, IMG_SZ, 3)
    inputs = keras.layers.Input(shape=input_shape)

    if MODEL_CLASS == "mobilenet" and MODEL_VERSION == "v1":
        model = keras.applications.MobileNet(
            alpha=ALPHA,
            include_top=False,
            input_shape=input_shape,
            input_tensor=inputs,
            weights="imagenet",
        )
    elif MODEL_CLASS == "mobilenet" and MODEL_VERSION == "v2":
        model = keras.applications.MobileNetV2(
            alpha=ALPHA,
            include_top=False,
            input_shape=input_shape,
            input_tensor=inputs,
            weights="imagenet",
        )
    else:
        raise RuntimeError("model not available:" + MODEL_CLASS)

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.1
    x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = keras.layers.Dense(len(LABELS), activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="MobileNet")
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def callbacks():
    earlyStopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    saveBest = keras.callbacks.ModelCheckpoint(
        BEST_KERAS_PATH,
        monitor="val_loss",
        save_best_only=True,
    )
    logger = keras.callbacks.CSVLogger(METRICS_PATH)
    return [earlyStopping, saveBest, logger]


LABELS = client.task.project.labels
MODEL_NAME = client.task.model_name
EPOCHS = client.task.epochs
BATCH_SIZE = client.task.batch_size
LEARNING_RATE = client.task.learning_rate
EXPORT_WIDTH = client.task.export_width
EXPORT_HEIGHT = client.task.export_height
MODEL_CLASS, MODEL_VERSION, ALPHA, IMG_SZ = MODEL_NAME.split("_")
ALPHA = float(ALPHA)
IMG_SZ = int(IMG_SZ)

print(f"CUDA available: {tf.test.is_gpu_available(cuda_only=True)}")

print("Download dataset")
client.export_class_folder(DATASET_PATH)

print("Build dataset")
train_ds, val_ds = build_dataset()

model = build_model()

print("Training started")
hist = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks(),
    verbose=2,
)

print("Export tflite")
model = keras.models.load_model(BEST_KERAS_PATH)

def representative_dataset():
    for file_name in (DATASET_PATH/"TEST").rglob('*'):
        if not file_name.is_file():
            continue
        if not file_name.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            continue
        img = keras.utils.load_img(file_name, target_size=(EXPORT_HEIGHT, EXPORT_WIDTH))
        array = keras.utils.img_to_array(img)
        yield [np.array([array / 127.5 - 1])]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open(RUN_PATH/"trained.tflite", "wb") as f:
    f.write(tflite_model)

with open(RUN_PATH/"labels.txt", "wb") as f:
    f.write("\n".join(LABELS).encode("utf-8"))

with ZipFile(RUN_PATH/"result.zip", 'w') as zip:
    zip.write("openmv.py", arcname="main.py")
    zip.write(RUN_PATH/"trained.tflite", arcname="trained.tflite")
    zip.write(RUN_PATH/"labels.txt", arcname="labels.txt")

client.upload_results_zip(RUN_PATH/"result.zip")
print("Finished")
