# NLP Mini-Projects Repository

This repository hosts a collection of mini-projects for the **Natural Language Processing (NLP)** course. Each project explores various aspects of NLP, providing hands-on implementation and insights. Below is the first completed project.

---

## Project 1: Sentiment Classification on UIT-VSFC Dataset

### Overview

The first project focuses on sentiment classification, utilizing the **UIT-VSFC dataset**. The goal is to classify text data into distinct sentiment categories, leveraging machine learning and transformer-based models.

---

### Implementation Details

#### Key Features:
- Dataset: UIT-VSFC (Vietnamese Students Feedback Corpus)
- Models: 
  - **BERT** (Bidirectional Encoder Representations from Transformers)
  - **XLNet** (Generalized Autoregressive Pretraining)

#### Project Structure:
- **`src/`**: Contains all source code files:
  - `data.py`: Handles dataset loading and preprocessing.
  - `model.py`: Defines the model architectures.
  - `trainer.py`: Contains training loops and evaluation logic.
  - `evaluator.py`: Computes metrics for performance evaluation.
  - `main.py`: Entry point for running the experiments.

- **`requirements.txt`**: Lists the Python dependencies needed to run the project.
- **`.gitignore`**: Specifies files and directories to be ignored in the repository.

---

### Results

The models were trained and evaluated on the UIT-VSFC dataset. Below are the performance metrics:

| Metric      | BERT | XLNet |
|-------------|------|-------|
| Accuracy    | 0.89 | 0.89  |
| F1-Score    | 0.74 | 0.72  |
| Precision   | 0.80 | 0.77  |
| Recall      | 0.71 | 0.70  |


---

### How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/KinhNguyenVan/NLP-PROJECT
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python src/main.py
   ```
