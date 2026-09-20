# Age Detection with CNN

A deep learning project for predicting a person’s age from facial images using a convolutional neural network (CNN). The project includes dataset preparation, model training, single-image inference, and real-time webcam age estimation with face detection.

## Overview

This project uses the UTKFace dataset and a CNN trained on grayscale face crops resized to 128x128. The model learns to estimate age as a regression target and predicts a numerical age from a detected face.

The workflow in this repository is implemented in [index.ipynb](index.ipynb) and includes:

- dataset loading and preprocessing
- face image extraction and normalization
- CNN model training with TensorFlow/Keras
- model saving
- age prediction on static images
- live age estimation from webcam frames using Haar cascade face detection

## Features

- CNN-based age estimation model
- Grayscale face preprocessing for fast training
- UTKFace dataset support
- Single-image prediction
- Real-time webcam inference
- Model saved as Keras artifact for reuse

## Tech Stack

- Python 3.9+
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- PIL (Pillow)
- Matplotlib
- scikit-learn

## Project Structure

- [index.ipynb](index.ipynb) — main notebook for training and inference
- [models/model.keras](models/model.keras) — trained Keras model
- [models/haarcascade_frontalface_default.xml](models/haarcascade_frontalface_default.xml) — OpenCV Haar cascade for face detection
- [datasets/](datasets/) — dataset folders, including UTKFace and other image data
- [test.jpg](test.jpg) — sample image for testing

## Dataset

This project uses the UTKFace dataset, which contains face images labeled with age, gender, and other metadata.

Typical filename format:

- 26_0_0_20170116194529155.jpg
- age = 26
- gender = 0

You can download the dataset from the UTKFace project or from a mirror/drive link if provided in the notebook. Make sure the folder is placed under the dataset path referenced by the code.

Expected dataset structure:

```text
datasets/
  UTKFace/
    26_0_0_20170116194529155.jpg
    30_1_0_20170116234512345.jpg
    ...
```

## Setup

1. Clone the repository

```bash
git clone <your-repo-url>
cd Age_Detection
```

2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies

```bash
pip install tensorflow opencv-python matplotlib pillow scikit-learn numpy pandas
```

If you use Jupyter Notebook:

```bash
pip install notebook
```

## Training the Model

Open [index.ipynb](index.ipynb) in Jupyter and run the cells in sequence.

The notebook performs the following:

1. Reads the UTKFace image filenames
2. Extracts age labels
3. Loads images as grayscale 128x128 arrays
4. normalizes pixel values to 0–1
5. builds a CNN regression model
6. trains on the data and validates performance
7. saves the trained model to [models/model.keras](models/model.keras)

## Running Inference

### Prediction on a single image

The notebook includes functions that preprocess a single image and pass it through the model to get an age estimate.

```python
img = load_img('test.jpg', grayscale=True)
img = img.resize((128, 128), Image.ANTIALIAS)
img = np.array(img)
img = img.reshape(1, 128, 128, 1) / 255.0
pred = model.predict(img)
print(int(pred[0][0]))
```

### Real-time webcam demo

The webcam script in the notebook uses OpenCV face detection and predicts the age for each detected face.

```python
model = load_model("models/model.keras")
detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
```

Press Esc to exit the webcam window.

## Model Architecture

The model is a convolutional neural network built with Keras:

- Conv2D layers with ReLU activation
- MaxPooling2D layers
- GlobalMaxPooling2D
- Dense hidden layer
- Final Dense layer for age regression

Loss function used:

- MSE (Mean Squared Error)

This is a regression task, so the final output is a continuous age estimate rather than a class label.

## Notes

- The model is trained on grayscale images, so the preprocessing pipeline expects 128x128 single-channel arrays.
- Face detection and age estimation depend on image quality, lighting, and dataset diversity.
- Running on the webcam may require adjusting the camera index if the default webcam is not accessible.

## Troubleshooting

### Camera not opening

Try changing:

```python
cam = cv2.VideoCapture(0)
```

to:

```python
cam = cv2.VideoCapture(1)
```

### Model not found

Make sure the model exists in the correct location:

- [models/model.keras](models/model.keras)

### Dataset not found

Ensure the dataset directory matches the path used by the notebook, usually:

```python
Base_dir = 'datasets/UTKFace/'
```

## Future Improvements

- Add gender prediction alongside age estimation
- Train on larger and more balanced datasets
- Use data augmentation to improve robustness
- Experiment with transfer learning using pretrained CNN backbones
- Deploy as a Flask/FastAPI API or desktop app

## License

This project is provided for educational and research purposes. Please check the dataset license before using UTKFace for commercial or public deployment.

## Acknowledgements

- UTKFace dataset
- OpenCV Haar cascade models
- TensorFlow and Keras community

## Contributing

Pull requests and improvements are welcome. If you want to improve the model accuracy or add new features, open an issue or submit a PR.
