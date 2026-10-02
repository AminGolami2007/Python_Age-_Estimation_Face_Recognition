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
    ReduceLROnPlateau
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

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "age_model.keras"
)

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
    print("GPU detected:")

    for gpu in gpus:
        print(" ", gpu)

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )
    except RuntimeError as e:
        print(e)

else:
    print("No GPU detected. Using CPU.")


# ============================================================
# 4. CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 5. DATASET PARSER
# ============================================================

def get_dataset():

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_DIR}"
        )

    image_paths = []
    ages = []

    print("\nReading dataset...")

    for filename in os.listdir(DATASET_DIR):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        parts = filename.split("_")

        if len(parts) < 2:
            continue

        try:
            age = int(parts[0])
        except ValueError:
            continue

        # UTKFace valid age range
        if age < 0 or age > 120:
            continue

        image_path = os.path.join(
            DATASET_DIR,
            filename
        )

        image_paths.append(image_path)
        ages.append(age)

    if len(image_paths) == 0:
        raise RuntimeError(
            "No valid images were found in the dataset."
        )

    image_paths = np.array(image_paths)
    ages = np.array(ages, dtype=np.float32)

    print(f"Total valid images: {len(image_paths)}")

    print(
        f"Minimum age: {ages.min():.0f}"
    )

    print(
        f"Maximum age: {ages.max():.0f}"
    )

    print(
        f"Average age: {ages.mean():.2f}"
    )

    return image_paths, ages


# ============================================================
# 6. CREATE AGE BINS
# ============================================================

def create_age_bins(ages):
    """
    Create age groups so train/validation/test
    maintain a similar age distribution.
    """

    bins = np.array(
        [
            0,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            121
        ]
    )

    age_bins = np.digitize(
        ages,
        bins
    )

    return age_bins


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset(image_paths, ages):

    age_bins = create_age_bins(ages)

    # First:
    # 90% temporary
    # 10% test

    (
        train_val_paths,
        test_paths,
        train_val_ages,
        test_ages,
        train_val_bins,
        _
    ) = train_test_split(
        image_paths,
        ages,
        age_bins,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=age_bins
    )

    # Validation fraction relative to remaining 90%
    validation_ratio = (
        VALIDATION_SIZE /
        (1.0 - TEST_SIZE)
    )

    (
        train_paths,
        val_paths,
        train_ages,
        val_ages,
        _,
        _
    ) = train_test_split(
        train_val_paths,
        train_val_ages,
        train_val_bins,
        test_size=validation_ratio,
        random_state=SEED,
        stratify=train_val_bins
    )

    print("\nDataset split:")

    print(
        f"Training:   {len(train_paths)}"
    )

    print(
        f"Validation: {len(val_paths)}"
    )

    print(
        f"Testing:    {len(test_paths)}"
    )

    return (
        train_paths,
        train_ages,
        val_paths,
        val_ages,
        test_paths,
        test_ages
    )


# ============================================================
# 8. IMAGE LOADING FUNCTION
# ============================================================

def load_and_preprocess_image(
    image_path,
    age
):

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # Normalize to [0, 1]
    image = image / 255.0

    age = tf.cast(
        age,
        tf.float32
    )

    return image, age


# ============================================================
# 9. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),

    ],
    name="data_augmentation"
)


# ============================================================
# 10. CREATE TF.DATA DATASET
# ============================================================

def create_tf_dataset(
    image_paths,
    ages,
    training=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            ages
        )
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            lambda image, age: (
                data_augmentation(
                    image,
                    training=True
                ),
                age
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 11. BUILD CNN MODEL
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    x = inputs

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # --------------------------------------------------------
    # Block 4
    # --------------------------------------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # --------------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    # --------------------------------------------------------
    # Fully Connected
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    # --------------------------------------------------------
    # Age Output
    # --------------------------------------------------------

    outputs = layers.Dense(
        1,
        activation="linear",
        name="age"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=tf.keras.losses.Huber(),

        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            ),

            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ]
    )

    return model


# ============================================================
# 12. TRAIN MODEL
# ============================================================

def train_model():

    (
        image_paths,
        ages
    ) = get_dataset()

    (
        train_paths,
        train_ages,
        val_paths,
        val_ages,
        test_paths,
        test_ages
    ) = split_dataset(
        image_paths,
        ages
    )

    train_dataset = create_tf_dataset(
        train_paths,
        train_ages,
        training=True
    )

    validation_dataset = create_tf_dataset(
        val_paths,
        val_ages,
        training=False
    )

    test_dataset = create_tf_dataset(
        test_paths,
        test_ages,
        training=False
    )

    model = build_model()

    print("\nModel architecture:\n")

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        EarlyStopping(
            monitor="val_mae",
            patience=7,
            mode="min",
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="min",
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...\n")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # Load Best Model
    # --------------------------------------------------------

    print("\nLoading best model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Final Test
    # --------------------------------------------------------

    print("\nEvaluating on TEST dataset...\n")

    test_results = model.evaluate(
        test_dataset,
        return_dict=True
    )

    print("\nFinal Test Results:")

    for metric, value in test_results.items():

        print(
            f"{metric}: {value:.4f}"
        )

    return model


# ============================================================
# 13. LOAD EXISTING MODEL
# ============================================================

def load_existing_model():

    if not os.path.exists(MODEL_PATH):

        print(
            "No trained model found."
        )

        return None

    print(
        f"Loading model:\n{MODEL_PATH}"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# 14. WEBCAM AGE ESTIMATION
# ============================================================

def run_webcam(model):

    if not os.path.exists(
        FACE_CASCADE_PATH
    ):

        raise FileNotFoundError(
            "Haar Cascade file not found:\n"
            f"{FACE_CASCADE_PATH}"
        )

    detector = cv2.CascadeClassifier(
        FACE_CASCADE_PATH
    )

    if detector.empty():

        raise RuntimeError(
            "Could not load Haar Cascade."
        )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    print("\nWebcam started.")

    print(
        "Press ESC to exit."
    )

    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "Could not read camera frame."
            )

            break

        # Mirror effect
        frame = cv2.flip(
            frame,
            1
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (
            x,
            y,
            w,
            h
        ) in faces:

            # ------------------------------------------------
            # Crop face from RGB image
            # ------------------------------------------------

            face = frame[
                y:y+h,
                x:x+w
            ]

            if face.size == 0:
                continue

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            face = cv2.resize(
                face,
                IMAGE_SIZE
            )

            # ------------------------------------------------
            # OpenCV uses BGR.
            # Convert to RGB because training images
            # are decoded as RGB.
            # ------------------------------------------------

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # Normalize exactly like training
            # ------------------------------------------------

            face = face.astype(
                np.float32
            ) / 255.0

            # ------------------------------------------------
            # Add batch dimension
            # ------------------------------------------------

            face = np.expand_dims(
                face,
                axis=0
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(
                face,
                verbose=0
            )

            predicted_age = float(
                prediction[0][0]
            )

            # Keep age in realistic range
            predicted_age = np.clip(
                predicted_age,
                0,
                120
            )

            predicted_age = int(
                round(predicted_age)
            )

            # ------------------------------------------------
            # Draw face rectangle
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # ------------------------------------------------
            # Display age
            # ------------------------------------------------

            text = (
                f"Age: {predicted_age}"
            )

            text_y = max(
                y - 10,
                30
            )

            cv2.putText(
                frame,
                text,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Display webcam
        # ----------------------------------------------------

        cv2.imshow(
            "Age Estimation",
            frame
        )

        # ESC
        if (
            cv2.waitKey(1)
            & 0xFF
            == 27
        ):
            break

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "        AGE ESTIMATION SYSTEM"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # If model doesn't exist -> train
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "\nNo trained model found."
        )

        print(
            "Training a new model..."
        )

        model = train_model()

    else:

        print(
            "\nExisting model found."
        )

        model = load_existing_model()

    # --------------------------------------------------------
    # Run webcam
    # --------------------------------------------------------

    run_webcam(
        model
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()