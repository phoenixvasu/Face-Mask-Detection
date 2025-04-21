# Face Mask Detection System

A real-time face mask detection system using deep learning and computer vision. This project can detect whether a person is wearing a face mask or not in real-time using a webcam feed.

## Features

- Real-time face mask detection using webcam
- High accuracy detection using MobileNetV2 deep learning model
- Visual feedback with bounding boxes and confidence scores
- Training capability for custom datasets
- Support for both image and video processing

## Project Structure

```
FACE-MASK-DETECTION/
├── dataset/                  # Training dataset directory
│   ├── with_mask/           # Images of people wearing masks
│   └── without_mask/        # Images of people without masks
├── face_detector/           # Pre-trained face detection models
│   ├── deploy.prototxt      # Caffe model architecture
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Caffe model weights
├── detect_mask_video.py     # Real-time mask detection script
├── train_mask_detector.py   # Model training script
├── mask_detector.model      # Trained mask detection model
├── plot.png                 # Training accuracy/loss plot
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.x
- TensorFlow 2.13.0
- Keras 2.13.1
- OpenCV 4.8.0
- NumPy 1.24.3
- imutils 0.5.4
- Matplotlib 3.8.0
- SciPy 1.11.2

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd FACE-MASK-DETECTION
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Prepare your dataset:
   - Place images of people wearing masks in `dataset/with_mask/`
   - Place images of people without masks in `dataset/without_mask/`

2. Train the model:
```bash
python train_mask_detector.py
```

The training script will:
- Load and preprocess the dataset
- Train the MobileNetV2 model
- Save the trained model as `mask_detector.model`
- Generate a training plot as `plot.png`

### Real-time Detection

Run the detection script:
```bash
python detect_mask_video.py
```

The script will:
- Start your webcam
- Detect faces in real-time
- Classify each face as "Mask" or "No Mask"
- Display bounding boxes and confidence scores
- Press 'q' to quit

## How It Works

1. **Face Detection**: Uses OpenCV's DNN module with a pre-trained Caffe model to detect faces in the video stream.

2. **Mask Classification**: 
   - Extracts each detected face
   - Preprocesses the face image
   - Uses a fine-tuned MobileNetV2 model to classify whether the person is wearing a mask

3. **Visualization**:
   - Green bounding box and label for "Mask"
   - Red bounding box and label for "No Mask"
   - Confidence score displayed as percentage

## Model Architecture

The system uses a two-stage approach:
1. Face detection using OpenCV's DNN module
2. Mask classification using a fine-tuned MobileNetV2 model

The MobileNetV2 model is trained with:
- Binary classification (with_mask/without_mask)
- Data augmentation for better generalization
- Transfer learning from ImageNet weights
- Custom head with dropout for regularization

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 