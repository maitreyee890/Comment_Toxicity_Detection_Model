# Toxicity Detection Model

This repository contains a Jupyter notebook designed to build and evaluate a machine learning model for detecting toxicity in text. The notebook leverages TensorFlow and other libraries for preprocessing, training, and testing.

## Table of Contents

1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [Notebook Workflow](#notebook-workflow)
4. [Model Details](#model-details)
5. [Usage](#usage)
6. [Results and Evaluation](#results-and-evaluation)
7. [License](#license)

## Overview

The notebook accomplishes the following:

- Preprocessing text data using `TextVectorization`.
- Creating TensorFlow data pipelines for efficient processing.
- Building a Sequential model with embedding, LSTM, and dense layers.
- Evaluating the model's performance using precision, recall, and categorical accuracy metrics.
- Visualizing training performance.
- Providing an interactive interface via Gradio for model testing.

## Setup and Installation

### Requirements

- Python 3.7+
- Libraries:
  - numpy
  - pandas
  - tensorflow
  - matplotlib
  - gradio

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Files

- `train.csv`: Dataset containing text and corresponding labels.
- `Toxicity_model.ipynb`: The main notebook file.
- `toxicity.h5`: Saved model file (generated after training).

## Notebook Workflow

### 1. Data Loading and Exploration
- The dataset is loaded using pandas.
- Initial exploration includes displaying the first few rows of data.

### 2. Preprocessing
- Text data is vectorized using TensorFlow's `TextVectorization` layer.
- Data is tokenized and transformed into integer sequences.

### 3. Data Pipelining
- TensorFlow data pipelines are created with caching, shuffling, batching, and prefetching to optimize training.

### 4. Model Creation
- A Sequential model is built with the following layers:
  - Embedding layer for word representations.
  - Bidirectional LSTM for sequence analysis.
  - Dense layers for feature extraction.
  - Sigmoid activation for multi-label classification.

### 5. Training
- The model is trained on 70% of the data with 20% used for validation.
- Binary cross-entropy loss and Adam optimizer are used.

### 6. Evaluation
- Metrics like precision, recall, and categorical accuracy are calculated on the test set.

### 7. Deployment
- The trained model is saved as `toxicity.h5`.
- Gradio is used to create a simple interface for testing predictions.

## Model Details

- **Embedding**: Converts words to dense vectors.
- **Bidirectional LSTM**: Processes input sequences in both forward and backward directions for better context understanding.
- **Dense Layers**: Fully connected layers for feature extraction and classification.
- **Output Layer**: Six units with sigmoid activation for multi-label classification.

## Usage

1. Run the notebook:

   ```bash
   jupyter notebook Toxicity_model.ipynb
   ```

2. Follow the notebook cells sequentially to preprocess data, train the model, and evaluate results.

3. Test the model using the Gradio interface:

   ```python
   import gradio as gr
   gr.Interface(fn=predict_fn, inputs=["textbox"], outputs=["label"]).launch()
   ```

## Results and Evaluation

- Training and validation loss/accuracy are plotted for analysis.
- Precision, recall, and accuracy metrics are printed for the test set.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
