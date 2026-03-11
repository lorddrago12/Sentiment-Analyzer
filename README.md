# Sentiment Analyzer

A simple project demonstrating how to convert text into numerical vectors using `CountVectorizer` and train a basic sentiment classifier with a `DecisionTreeClassifier` in scikit-learn.

## Overview

This project:

- **Builds a tiny sentiment dataset** of positive and negative phrases
- **Vectorizes text** using `sklearn.feature_extraction.text.CountVectorizer`
- **Trains a classifier** using `sklearn.tree.DecisionTreeClassifier` on the vectorized data

The core logic lives in `main.py`.

## Prerequisites

- Python 3.8+ (any modern 3.x version should work)
- `pip` for installing dependencies

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/lorddrago12/Sentiment-Analyzer.git
cd Sentiment-Analyzer
```

2. **Create and activate a virtual environment** (optional but recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install scikit-learn
```

## Usage

Currently, `main.py`:

- Defines positive and negative example texts
- Fits a `CountVectorizer` on the training text
- Trains a `DecisionTreeClassifier` on the vectorized text

To run the script:

```bash
python main.py
```

> Note: As written, the script only trains the model and does not yet print predictions. You can extend it to accept user input or test sentences and print their predicted sentiment.

## Future Improvements

- Add a CLI or simple UI for entering custom text
- Save and load the trained model (`joblib` or `pickle`)
- Add evaluation metrics (accuracy, confusion matrix) on a test set
- Expand and balance the dataset for better performance
