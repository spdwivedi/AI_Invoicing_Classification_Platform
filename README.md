# NextBill AI Invoicing Classification Platform

An enterprise-grade, multi-model microservice platform designed to automate the classification of unstructured invoice line-item text into structured corporate expense categories. The system hosts traditional statistical machine learning pipelines alongside an advanced context-aware deep learning Transformer model inside an optimized, containerized web cluster.

---

## Table of Contents

1. [Introduction & Quick Start Guide](#1-introduction--quick-start-guide)
2. [Core Project Objectives & Compliance Criteria](#2-core-project-objectives--compliance-criteria)
3. [Production Repository Directory Topography](#3-production-repository-directory-topography)
4. [Dataset Lifecycle 1: Foundational Training Seed Core (`data.csv`)](#4-dataset-lifecycle-1-foundational-training-seed-core-datacsv)
5. [Dataset Lifecycle 2: Synthetic Validation Benchmarking (`faker_benchmark_v1.csv`)](#5-dataset-lifecycle-2-synthetic-validation-benchmarking-faker_benchmark_v1csv)
6. [Dataset Lifecycle 3: Real-World Dataset Shift Challenge (`kaggle_test_data.csv`)](#6-dataset-lifecycle-3-real-world-dataset-shift-challenge-kaggle_test_datacsv)
7. [Multi-Model Preprocessing & Tokenization Workflows](#7-multi-model-preprocessing--tokenization-workflows)
8. [Baseline Pipeline 1: TF-IDF Vectorization + Logistic Regression](#8-baseline-pipeline-1-tf-idf-vectorization--logistic-regression)
9. [Baseline Pipeline 2: CountVectorizer + Multinomial Naive Bayes](#9-baseline-pipeline-2-countvectorizer--multinomial-naive-bayes)
10. [Production Core Engine: Multi-Head Self-Attention DistilBERT Transformer](#10-production-core-engine-multi-head-self-attention-distilbert-transformer)
11. [Cross-Dataset Performance Matrices & Precision-Recall Metrics Evaluation](#11-cross-dataset-performance-matrices--precision-recall-metrics-evaluation)
12. [Root-Cause Post-Mortem: Algorithmic Overfitting & Word-Count Failure Mechanics](#12-root-cause-post-mortem-algorithmic-overfitting--word-count-failure-mechanics)
13. [Global Middleware Interceptor & Grouped Hierarchical JSONL Logging System](#13-global-middleware-interceptor--grouped-hierarchical-jsonl-logging-system)
14. [Dynamic CSV Data Batch Tester Engine & Programmatic Justification Analytics](#14-dynamic-csv-data-batch-tester-engine--programmatic-justification-analytics)
15. [Asynchronous Non-Blocking On-Demand Fine-Tuning Daemon Threads](#15-asynchronous-non-blocking-on-demand-fine-tuning-daemon-threads)
16. [Interactive API Documentation & Endpoint Usage](#16-interactive-api-documentation--endpoint-usage)

---

<div align="center">

[![Live Project](https://img.shields.io/badge/🟢_Live-Project_Link-success?style=for-the-badge)](https://invoicing_classification.spdwivedi.me/)
[![GitHub](https://img.shields.io/badge/💻_GitHub-Repository-black?style=for-the-badge)](https://github.com/spdwivedi/AI_Invoicing_Classification_Platform)
[![Google Drive](https://img.shields.io/badge/📁_Google_Drive-Project_Assets-blue?style=for-the-badge)](https://drive.google.com/drive/folders/1UvtXSiwfpxa3I_Qvwv9v4DxkOXn2ulNS?usp=sharing)
[![Portfolio](https://img.shields.io/badge/👨‍💻_Portfolio-spdwivedi.me-teal?style=for-the-badge)](https://spdwivedi.me)
[![LinkedIn](https://img.shields.io/badge/🔗_LinkedIn-Connect-0A66C2?style=for-the-badge)](https://www.linkedin.com/in/spdwivedi2001/)

</div>

## 1. Introduction & Quick Start Guide

The NextBill AI platform serves as an end-to-end blueprint for deploying, benchmarking, and tracking machine learning models in a live production environment. The platform features an interactive single-page dashboard UI and an API layer that exposes automated parallel inference paths, dynamic batch CSV file processing, and asynchronous background model optimizations.

### Production Deployment via Docker (Recommended)

The entire environment is containerized using optimized, CPU-isolated Python images to guarantee performance stability and avoid dependency drift across local development machines and remote cloud infrastructure.

    # 1. Clone the repository workspace and navigate to the root directory
    cd NextBill/Code

    # 2. Compile the production container image utilizing the Python 3.11 blueprint
    docker build -t nextbill-api .

    # 3. Spin up the detached background engine mapping host networking port 8000
    docker run -d -p 8000:8000 --name nextbill-service nextbill-api

    # 4. Access the interactive web dashboard portal natively in your web browser
    # Open URL: http://127.0.0.1:8000/

### Alternative Local Native Installation

If you prefer to run the codebase directly within a local virtual environment or Conda workspace, you can execute the server using standard terminal tools:

    # 1. Initialize a clean virtual tracking layer
    python -m venv venv
    source venv/bin/activate  # On Windows PowerShell use: .\venv\Scripts\Activate.ps1

    # 2. Install optimized CPU wheels and structural library dependencies
    pip install -r requirements.txt

    # 3. Navigate into the deployment engine folder mapping context
    cd production_deployment

    # 4. Boot up the microservice cluster using the Uvicorn ASGI server
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

---

## 2. Core Project Objectives & Compliance Criteria

This system is built to satisfy the core criteria of the machine learning engineer evaluation task:

* **Standardized API Delivery (`production_deployment/main.py`):** Exposes a clean `POST /predict` endpoint that accepts structured text payloads and returns a single category label and confidence score matching the strict schema constraints.
* **Comprehensive Multi-Class Domain Support:** Models are explicitly trained to classify unstructured sequences into six mandatory corporate expense classes: `Logistics`, `Office Supplies`, `Cloud/Software`, `Utilities`, `Travel`, and `Inventory`.
* **Production Extensions (Bonus Marks Delivered):** Implements float-precision confidence scores, a multi-layer Docker architecture (`Dockerfile`), standalone automated training scripts (`train.py` across model folders), custom validation verification logs (`storage/history/web_predictions.jsonl`), and parallel inference endpoints.

---

## 3. Production Repository Directory Topography

    Code/
    ├── Dockerfile                       # Environment orchestrator utilizing Python 3.11 slim
    ├── requirements.txt                 # Exact package versions (FastAPI, Scikit-Learn, PyTorch, Transformers)
    ├── README.md                        # Project documentation
    ├── data.csv                         # The foundational seed matrix used strictly for training
    ├── faker_benchmark_v1.csv           # Evaluation Set 1: Synthetic baseline validations
    ├── kaggle_test_data.csv             # Evaluation Set 2: Unstructured organic retail data tests
    ├── save_default_reports.py          # Utility script that computes and freezes initial metrics
    │
    ├── production_deployment/           # The live API and UI hosting environment
    │   ├── main.py                      # Centralized FastAPI gateway. Manages dynamic routing and logs
    │   └── index.html                   # Interactive single-page UI dashboard
    │
    ├── v1_L/                            # Model Pipeline 1: TF-IDF + Logistic Regression
    │   ├── model.joblib                 # Serialized machine learning weights
    │   ├── main.py                      # Sandbox script for local testing
    │   └── train.py                     # Script used to fit the vectorizer and linear classifier
    │
    ├── v1_NB/                           # Model Pipeline 2: CountVectorizer + Naive Bayes
    │   ├── model.joblib                 # Serialized probability distributions
    │   ├── main.py                      # Sandbox script for isolated probability testing
    │   └── train.py                     # Script used to fit conditional probabilities
    │
    ├── v1_T/                            # Model Pipeline 3: DistilBERT Deep Learning Transformer
    │   ├── label_encoder.joblib         # Categorical integer-to-string mapping dictionary
    │   ├── saved_transformer_model/     # PyTorch architecture outputs (safetensors, vocab, config.json)
    │   ├── main.py                      # Isolated transformer testing script
    │   └── train.py                     # Hugging Face trainer script mapping contextual embeddings
    │
    └── Research & Utility Scripts/      # (Archived) Scripts used to harvest and generate raw assets
        ├── download_public_data.py      # Automated web scraper fetching real-world UCI retail descriptions
        ├── generate_data.py             # Script to build and duplicate the initial training seed matrix
        └── generate_faker_data.py       # Script synthesizing randomized corporate transaction benchmarks

---

## 4. Dataset Lifecycle 1: Foundational Training Seed Core (`data.csv`)

To prevent training data leakage, **`data.csv`** is utilized strictly as the isolated training set. It contains the base vocabulary patterns required to teach the algorithms the distinction between various corporate expense categories.

* **Generation Mechanism (`generate_data.py`):** This script initializes a dictionary of highly distinct contextual templates and duplicates them sequentially.
* **Class Distribution Strategy:** The script ensures perfect symmetry, producing an equal number of duplicate rows per category to prevent majority-class bias during model fitting.
* **Functional Purpose:** It serves as the baseline ground truth. The Traditional ML models map exact keywords from this file, while the Transformer learns the sequence structure of these templates.

---

## 5. Dataset Lifecycle 2: Synthetic Validation Benchmarking (`faker_benchmark_v1.csv`)

Because a model cannot be tested on its own training data, **`faker_benchmark_v1.csv`** serves as the first Out-of-Sample evaluation layer. It simulates a "clean" production environment.

* **Generation Mechanism (`generate_faker_data.py`):** This script uses the Python `Faker` library. It defines string templates like "Courier charges from {company} for {item} transit" and dynamically injects randomized corporate names, cities, and ID numbers.
* **Functional Detail:** By shuffling company names and cities into the text, we test if the machine learning models have overfitted to specific words, or if they correctly ignore the random noise (the fake company names) and focus on the core expense action.
* **What's Happening Here:** This benchmark proves that all three models (Linear, Bayes, and Transformer) can easily maintain ~90% accuracy when the language remains syntactically clean and the token boundaries are highly predictable.

---

## 6. Dataset Lifecycle 3: Real-World Dataset Shift Challenge (`kaggle_test_data.csv`)

To truly test the robustness of the system, **`kaggle_test_data.csv`** introduces chaotic, unstructured "Dataset Shift." It abandons synthetic cleanliness for organic, messy human text.

* **Generation Mechanism (`download_public_data.py`):** This ingestion engine programmatically downloads the public UCI Online Retail dataset. It cleans complex latin-1 encoding errors, strips null descriptions, and maps real-world wholesale items to our target categories.
* **Functional Detail:** The text strings in this file (e.g., "WHITE HANGING HEART T-LIGHT HOLDER", "KNITTED UNION FLAG HOT WATER BOTTLE") lack standard grammatical structure, missing verbs, and utilizing shorthand.
* **What's Happening Here:** This file is used dynamically to demonstrate *Model Collapse*. It proves that when keyword phrasing shifts unexpectedly, traditional word-count algorithms fail spectacularly, while the Transformer maintains its contextual logic.

---

## 7. Multi-Model Preprocessing & Tokenization Workflows

Before any raw invoice string can be evaluated by the classification algorithms, it must pass through specialized tokenization engines. The system implements two distinct structural pathways in the respective `main.py` inference scripts:

### A. The Bag-of-Words Classical Stream (`v1_L` & `v1_NB`)
Inside the `v1_L/main.py` and `v1_NB/main.py` pipelines, text is processed strictly mathematically based on word existence:
1. **Case Normalization & Cleansing:** The vectorizers standardize all text to lowercase to prevent vector duplication (ensuring "AWS", "Aws", and "aws" map to the identical matrix column).
2. **Frequency Matrix Generation:** * The Naive Bayes pipeline uses `CountVectorizer` to generate absolute integer counts of token occurrences.
    * The Logistic Regression pipeline uses `TfidfVectorizer` to generate float weights, mathematically penalizing words that appear too frequently across all categories (like "the" or "bill") and boosting highly unique category terms.

### B. The Transformer Context-Aware Stream (`v1_T`)
Inside `v1_T/main.py` (and integrated directly into the `production_deployment/main.py` gateway), text is processed for contextual geometry, not just word counting:
1. **WordPiece Sub-word Encoding:** The `AutoTokenizer` breaks complex or misspelled words into root sub-components (e.g., "Dotcomgiftshop" becomes `['Dot', '##com', '##gift', '##shop']`). This ensures that the deep learning model never encounters an absolute "Out-Of-Vocabulary" failure.
2. **Attention Mask Generation:** The tokenizer pads all incoming invoice strings to a fixed `max_length=64` tensor. It generates a binary sequence array (`1` for real words, `0` for padding) to instruct the neural network's multi-head attention blocks exactly where to focus its compute cycles and what empty space to ignore.

---

## 8. Baseline Pipeline 1: TF-IDF Vectorization + Logistic Regression

An established machine learning approach for context-constrained text categorization.

* **Architecture:** `[Input Text] ──> [TF-IDF Vectorizer] ──> [Sparse Term Matrix] ──> [Logistic Regression] ──> [Softmax Probabilities]`
* **Vectorizer Mechanics:** A Term Frequency-Inverse Document Frequency (TF-IDF) layer filters individual words, down-weighting generic tokens (like "a", "the", "bill") that appear frequently across all entries, while emphasizing category-specific terms (like "freight", "courier", "steel").
* **Classifier Optimization:** A multi-class Logistic Regression estimator fits a linear boundary to separate the TF-IDF feature distributions across the target classes. This model excels in computational efficiency and interpretability.

---

## 9. Baseline Pipeline 2: CountVectorizer + Multinomial Naive Bayes

A highly efficient statistical probability engine optimized for strict, non-overlapping vocabularies.

* **Architecture:** `[Input Text] ──> [CountVectorizer] ──> [Term Frequencies] ──> [MultinomialNB] ──> [Class Probabilities]`
* **Algorithmic Mechanics:** Operates on the strict mathematical assumption of token independence. It calculates the conditional probability of a category given the exact combination of absolute word occurrences.
* **Performance Profile:** Executes at near-zero compute latency. It performs exceptionally well when vocabulary boundaries are clear, but is highly susceptible to probability saturation if words cross-contaminate multiple classes, leading to over-confidence in the dominant class.

---

## 10. Production Core Engine: Multi-Head Self-Attention DistilBERT Transformer

The primary production engine utilizes a deep learning Neural Network to map semantic context rather than just counting words.

* **Architecture:** Hugging Face `DistilBertForSequenceClassification` loaded with pre-trained PyTorch weights, topped with a linear classification head mapped to our 6 target categorical labels.

* **Self-Attention Mechanics:** Instead of treating words as isolated features, the network calculates attention vectors determining how every single word in an invoice description relates to every other word. This allows the model to differentiate between similar words in different contexts (e.g., "hosting" in "Cloud/Software" vs. "hosting" in "Travel").
* **Deployment Optimization:** The PyTorch backend is explicitly compiled to a CPU-only wheel and restricted to `torch.set_num_threads(1)` to ensure stable, crash-free execution on resource-constrained Oracle Cloud free-tier compute instances.

---

## 11. Cross-Dataset Performance Matrices & Precision-Recall Metrics Evaluation

The platform tracks and evaluates models across two radically different environmental scenarios.

### Scenario 1: Clean Synthetic Boundaries (`faker_benchmark_v1.csv`)
*Evaluated on structured templates with clean token distributions.*

| Model Pipeline | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| **v1_T (DistilBERT)** | **91.67%** | 91.90% | 91.67% | 91.62% |
| **v1_NB (Naive Bayes)** | 89.44% | 89.62% | 89.44% | 89.31% |
| **v1_L (Logistic Reg)** | 88.89% | 88.70% | 88.89% | 88.64% |

### Scenario 2: Unstructured Real-World Dataset Shift (`kaggle_test_data.csv`)
*Evaluated on noisy, organic wholesale line items sourced from the public UCI Online Retail Corpus.*

| Model Pipeline | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| **v1_T (DistilBERT)** | **57.80%** | 61.20% | 57.80% | 58.41% |
| **v1_NB (Naive Bayes)** | 32.11% | 21.45% | 32.11% | 24.10% |
| **v1_L (Logistic Reg)** | 27.52% | 11.20% | 27.52% | 15.42% |

---

## 12. Root-Cause Post-Mortem: Algorithmic Overfitting & Word-Count Failure Mechanics

The catastrophic accuracy drop of the statistical baselines during Scenario 2 highlights the precise vulnerabilities of traditional ML systems in production:

* **v1_L (Logistic Regression) Collapse - The "Travel" Bias:** The model predicted almost every unstructured row as *Travel*. Because our training seed had sparse real-world context, the optimizer mapped minor string occurrences to heavy linear coefficients for the Travel class. Out in the wild, unfamiliar retail words triggered these coefficients.
* **v1_NB (Naive Bayes) Collapse - Probability Explosion:** Naive Bayes over-indexed heavily on *Cloud/Software*. Real-world items containing terms like "DIGITAL" or "ONLINE" triggered independent probability multiplications. Since these words appeared across diverse real items (e.g., "Dotcomgiftshop Voucher"), the network over-counted them and flooded the classification boundary.
* **The Transformer Advantage:** By evaluating the *entire sequence context*, DistilBERT correctly identifies semantic relationships that overcome isolated keyword noise, confirming that multi-head attention is essential for production-grade generalization.

---

## 13. Global Middleware Interceptor & Grouped Hierarchical JSONL Logging System

To maintain enterprise-grade auditability without the compute overhead of a heavy relational database, the application utilizes a lightweight, high-performance tracking system.

* **Global Interception:** A FastAPI `@app.middleware("http")` hook monitors all traffic inbound to port 8000. It logs discovery routes, schema queries, and automated pings silently.
* **Hierarchical JSONL Format:** Data is streamed to `storage/history/web_predictions.jsonl`. Batch CSV uploads are nested under a single parent JSON record object, allowing the frontend UI to parse and expand 1,000-row batch runs instantly without locking the browser thread.

---

## 14. Dynamic CSV Data Batch Tester Engine & Programmatic Justification Analytics

The platform exposes a `POST /upload-test-csv` endpoint capable of processing bulk invoice arrays dynamically.

* **On-the-Fly Ranking:** When a CSV containing ground-truth `category` labels is uploaded, the engine executes inference across all three isolated models concurrently, recalculates the accuracy matrices in real-time, and sorts the output to automatically display the winning architecture at the top of the dashboard.
* **Algorithmic Justification Engine:** The backend analyzes the vocabulary density of the uploaded file. If the density exceeds a programmed threshold, the UI generates a contextual explanation detailing why the DistilBERT model outperformed the baselines based on sequence variability.

---

## 15. Asynchronous Non-Blocking On-Demand Fine-Tuning Daemon Threads

To allow continuous model optimization without taking the API endpoints offline, the platform implements a background daemon core via FastAPI's `BackgroundTasks`.

* **Non-Blocking Architecture:** Users can upload a training CSV schema and define custom hyperparameter epochs. The server immediately returns a `200 OK` status, delegating the heavy tensor computations to an isolated thread.
* **State Polling:** The frontend dashboard opens an asynchronous web polling loop hitting `GET /train-status`, streaming live simulated cross-entropy loss metrics and epoch completion milestones directly to the console UI until the new architecture weights are safely persisted to the local volume layer.

---

---

## 16. Interactive API Documentation & Endpoint Usage

The NextBill API utilizes FastAPI's native OpenAPI integration to automatically generate a live, interactive documentation environment. This allows developers to inspect request schemas, execute inference, and monitor server responses directly from the browser without requiring external tools like Postman.

**🌐 Access the Live Swagger UI:** [https://invoicing-classification.spdwivedi.me/docs/](https://invoicing-classification.spdwivedi.me/docs/)

### Core Exposed Endpoints:

* **`GET /`**
    * **Purpose:** Base routing that serves the interactive HTML dashboard (`index.html`).
* **`POST /predict`**
    * **Purpose:** The primary inference gateway. Accepts a JSON payload containing raw invoice text (`description`) and an optional model selection parameter.
    * **Output:** Returns the predicted category mapping, the execution latency, and the float-precision confidence score.
* **`POST /upload-test-csv`**
    * **Purpose:** Bulk validation engine. Accepts a multipart form data CSV upload and routes it through all three models concurrently for real-time comparative matrix generation.
* **`POST /train` & `GET /train-status`**
    * **Purpose:** Triggers the asynchronous background `BackgroundTasks` daemon for non-blocking model fine-tuning, and polls the real-time cross-entropy loss metrics respectively.
* **`GET /logs`**
    * **Purpose:** Reads and parses the hierarchical `storage/history/web_predictions.jsonl` audit trail, returning paginated historical inference data to the dashboard.