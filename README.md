
# Video Understanding and Analysis with Streamlit

## Overview

This Streamlit application provides tools for video understanding and analysis, including audio extraction, speech-to-text conversion, object detection, OCR, sentiment analysis, and question answering. It uses Whisper for audio transcription, Llama2 for natural language processing, and other computer vision techniques for video frame analysis.

## Features

- **Audio Processing:** Extracts audio from videos and converts it to text using the Whisper model.
- **Video Analysis:** Detects objects within video frames, such as license plates or other objects.
- **OCR:** Extracts text from images within the video frames.
- **Sentiment Analysis:** Analyzes the sentiment of the transcribed text.
- **Question Answering:** Uses a retrieval-augmented generation (RAG) process to answer questions based on the video content.

## Requirements

- Python 3.x
- Conda or virtual environment
- CUDA (optional, for GPU acceleration)
- Required libraries:
  - TensorFlow
  - PyTorch
  - Whisper
  - Llama2 (or other LLM)
  - OpenCV
  - PyTesseract (for OCR)
  - scikit-learn
  - pandas
  - Streamlit

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/video-understanding-project.git
   cd video-understanding-project
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure CUDA is installed** (if using GPU acceleration):
   Follow the installation steps for your system from [NVIDIA's website](https://developer.nvidia.com/cuda-toolkit).

## Usage

1. **Start the Streamlit app:**

   To run the Streamlit app, use the following command:

   ```bash
   streamlit run app.py
   ```

2. **Functionality:**
   The app allows you to upload a video file and interact with the following features:

   - **Audio Extraction:** Extract audio from the video.
   - **Audio-to-Text Conversion:** Convert the extracted audio to text.
   - **Object Detection:** Detect objects in video frames.
   - **OCR Extraction:** Extract text from images within the video frames.
   - **Sentiment Analysis:** Analyze the sentiment of extracted text.
   - **Question Answering:** Ask questions about the video content, and get responses based on the extracted text.

3. **Interacting with the App:**
   The app provides a simple user interface where you can upload videos and trigger the processing tasks. Results are displayed in real-time.

## Example

To run the Streamlit app, simply execute:

```bash
streamlit run app.py
```

This will launch a local server in your browser where you can upload a video and use the app's features to analyze it.

## Acknowledgements

- [Whisper](https://github.com/openai/whisper) for audio-to-text conversion
- [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf) for natural language processing
- [OpenCV](https://opencv.org/) for video and image processing
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for optical character recognition
