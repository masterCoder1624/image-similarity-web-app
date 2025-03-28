# Image Similarity Detector

A web application built with FastAPI to detect and measure similarity between two images using multiple computer vision algorithms.

## Features

- Upload and compare any two images
- Multiple similarity metrics:
- Histogram comparison (color distribution)
- Structural Similarity Index (SSIM)
- Feature matching using SIFT algorithm
- Clean and intuitive user interface
- Real-time image previews
- Detailed explanation of results

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd image-similarity-detector
   ```

2. Set up a Python virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```
   
   Alternatively, you can use uvicorn directly:
   ```
   uvicorn app:app --reload
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8080
   ```

3. Upload two images using the form and click "Compare Images" to see the similarity results.

## Similarity Metrics Explained

- **Histogram Similarity**: Compares color distribution between images. Higher scores indicate similar color profiles.
  
- **Structural Similarity (SSIM)**: Measures perceived similarity in structure, luminance, and contrast. A score of 100% means identical images.
  
- **Feature Matching (SIFT)**: Detects and matches distinctive features between images. Higher scores indicate more matching features.

## Requirements

- Python 3.8+
- FastAPI
- OpenCV
- scikit-image
- NumPy

## License

MIT 