import os
import random

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50

TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10

DATASET_DIR = "datasets/UTKFace"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "age_model.keras")
FACE_CASCADE_PATH = os.path.join(
    MODEL_DIR,
    "haarcascade_frontalface_default.xml"
)

# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 3. GPU CONFIGURATION
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(" ", gpu)

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU memory-growth configuration error:", e)
else:
    print("\nNo GPU detected. Using CPU.")

print("\nTensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# ============================================================
# 4. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 5. LOAD UTKFACE DATASET
# ============================================================

def get_dataset():
    """
    Reads UTKFace filenames and extracts age from the filename.

    Typical UTKFace filename:
        100_0_0_20170112213500993.jpg

    The first value is the person's age.
    """

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"\nDataset folder not found:\n{os.path.abspath(DATASET_DIR)}\n"
            "Please make sure the UTKFace dataset is inside the project folder."
        )

    image_paths = []
    ages = []

    valid_extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(DATASET_DIR):

        if not filename.lower().endswith(valid_extensions):
            continue

        parts = filename.split("_")

        if len(parts) < 1:
            continue

        try:
            age = int(parts[0])
        except ValueError:
            continue

        # UTKFace age range
        if 0 <= age <= 120:
            image_paths.append(
                os.path.join(DATASET_DIR, filename)
            )
            ages.append(age)

    if len(image_paths) == 0:
        raise RuntimeError(
            "No valid UTKFace images were found in the dataset folder."
        )

    image_paths = np.array(image_paths)
    ages = np.array(ages, dtype=np.float32)

    print("\nDataset loaded.")
    print("Total images:", len(image_paths))
    print("Minimum age:", int(ages.min()))
    print("Maximum age:", int(ages.max()))
    print("Average age:", round(float(ages.mean()), 2))

    return image_paths, ages


# ============================================================
# 6. AGE BINS FOR STRATIFIED SPLITTING
# ============================================================

def create_age_bins(ages):
    """
    Creates age groups so train/validation/test sets
    have a similar age distribution.
    """

    bins = np.array(
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 121]
    )

    age_bins = np.digitize(ages, bins, right=False)

    return age_bins


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset(image_paths, ages):
    """
    Final split:
        80% training
        10% validation
        10% testing
    """

    age_bins = create_age_bins(ages)

    # First:
    # 90% -> train + validation
    # 10% -> test
    (
        train_val_paths,
        test_paths,
        train_val_ages,
        test_ages,
        train_val_bins,
        _,
    ) = train_test_split(
        image_paths,
        ages,
        age_bins,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=age_bins,
    )

    # Validation is 10% of the entire dataset.
    # Since train_val is 90%, validation ratio inside train_val is:
    validation_ratio = VALIDATION_SIZE / (1.0 - TEST_SIZE)

    (
        train_paths,
        val_paths,
        train_ages,
        val_ages,
        _,
        _,
    ) = train_test_split(
        train_val_paths,
        train_val_ages,
        train_val_bins,
        test_size=validation_ratio,
        random_state=SEED,
        stratify=train_val_bins,
    )

    print("\nDataset split:")
    print("Training:", len(train_paths))
    print("Validation:", len(val_paths))
    print("Testing:", len(test_paths))

    return (
        train_paths,
        train_ages,
        val_paths,
        val_ages,
        test_paths,
        test_ages,
    )


# ============================================================
# 8. IMAGE LOADING AND PREPROCESSING
# ============================================================

def load_and_preprocess_image(image_path, age):
    """
    Loads an image, supports JPG/JPEG/PNG,
    resizes it to 128x128 and normalizes pixels to [0, 1].
    """

    image = tf.io.read_file(image_path)

    # decode_image supports JPG, JPEG and PNG.
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method=tf.image.ResizeMethod.BILINEAR,
    )

    image = tf.cast(image, tf.float32) / 255.0

    age = tf.cast(age, tf.float32)

    return image, age


# ============================================================
# 9. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)


# ============================================================
# 10. TF.DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def create_dataset(image_paths, ages, training=False):
    """
    Creates an efficient tf.data pipeline.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, ages)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        dataset = dataset.map(
            lambda image, age: (
                data_augmentation(image, training=True),
                age,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ============================================================
# 11. CNN MODEL
# ============================================================

def build_model():
    """
    CNN regression model for age estimation.

    Input:
        128x128 RGB image

    Output:
        One continuous age value
    """

    inputs = layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="image",
    )

    x = inputs

    # -------------------------
    # Block 1
    # -------------------------
    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # -------------------------
    # Block 2
    # -------------------------
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # -------------------------
    # Block 3
    # -------------------------
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # -------------------------
    # Block 4
    # -------------------------
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # -------------------------
    # Regression head
    # -------------------------
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu",
    )(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        1,
        activation="linear",
        name="age",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="AgeEstimatorCNN",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss=tf.keras.losses.Huber(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


# ============================================================
# 12. TRAIN MODEL
# ============================================================

def train_model():
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    image_paths, ages = get_dataset()

    (
        train_paths,
        train_ages,
        val_paths,
        val_ages,
        test_paths,
        test_ages,
    ) = split_dataset(
        image_paths,
        ages,
    )

    train_dataset = create_dataset(
        train_paths,
        train_ages,
        training=True,
    )

    val_dataset = create_dataset(
        val_paths,
        val_ages,
        training=False,
    )

    test_dataset = create_dataset(
        test_paths,
        test_ages,
        training=False,
    )

    model = build_model()

    print("\nModel summary:")
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_mae",
            patience=5,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),

        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint.
    if os.path.exists(MODEL_PATH):
        print("\nLoading best model...")
        model = tf.keras.models.load_model(MODEL_PATH)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    test_results = model.evaluate(
        test_dataset,
        return_dict=True,
        verbose=1,
    )

    print("\nTest results:")

    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")

    return model, history


# ============================================================
# 13. LOAD EXISTING MODEL
# ============================================================

def load_existing_model():
    """
    Loads an already-trained .keras model.
    """

    if not os.path.exists(MODEL_PATH):
        return None

    print("\nLoading existing model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    return model


# ============================================================
# 14. WEBCAM AGE ESTIMATION
# ============================================================

def run_webcam(model):
    """
    Opens webcam and estimates the age of detected faces.

    Press ESC to exit.
    """

    if not os.path.exists(FACE_CASCADE_PATH):
        raise FileNotFoundError(
            f"\nHaar Cascade file not found:\n"
            f"{os.path.abspath(FACE_CASCADE_PATH)}\n"
            "Put haarcascade_frontalface_default.xml inside the models folder."
        )

    face_cascade = cv2.CascadeClassifier(
        FACE_CASCADE_PATH
    )

    if face_cascade.empty():
        raise RuntimeError(
            "Failed to load Haar Cascade classifier."
        )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    print("\nWebcam started.")
    print("Press ESC to exit.")

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("Failed to read frame from webcam.")
                break

            # Mirror the webcam image.
            frame = cv2.flip(frame, 1)

            # Haar Cascade works with grayscale.
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            for (x, y, w, h) in faces:

                # Keep coordinates inside the frame.
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                # Resize to model input size.
                face = cv2.resize(
                    face,
                    IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA,
                )

                # OpenCV uses BGR.
                # Model expects RGB.
                face = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB,
                )

                # Normalize to [0, 1].
                face = face.astype(
                    np.float32
                ) / 255.0

                # Add batch dimension:
                # (128, 128, 3)
                # ->
                # (1, 128, 128, 3)
                face = np.expand_dims(
                    face,
                    axis=0,
                )

                # Predict age.
                prediction = model.predict(
                    face,
                    verbose=0,
                )

                age = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

                # Keep result in valid UTKFace age range.
                age = np.clip(
                    age,
                    0,
                    120,
                )

                age_text = f"Age: {int(round(age))}"

                # Draw face rectangle.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # Draw age text.
                text_y = max(
                    30,
                    y1 - 10,
                )

                cv2.putText(
                    frame,
                    age_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                "Age Estimation",
                frame,
            )

            # ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AGE ESTIMATION SYSTEM")
    print("=" * 60)

    if os.path.exists(MODEL_PATH):

        print("\nExisting model found.")

        model = load_existing_model()

        if model is None:
            print("Could not load existing model.")
            return

    else:

        print("\nNo trained model found.")
        print("Training a new model...")

        model, _ = train_model()

    run_webcam(model)


# ============================================================
# 16. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
