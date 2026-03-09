# Classification Knowledge Library
---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Definition](#problem-definition)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Text Preprocessing and Feature Selection](#text-preprocessing-and-feature-selection)
5. [Model Development](#modeling-development)
6. [Model Evaluation](#model-evaluation)
7. [Performance Summary](#performance-summary)
8. [Suicide Risk Detection Framework](#suicide-risk-detection-framework)
9. [Limitations and Future Work](#limitations-and-future-work)
10. [How It Can Be Used](#how-it-can-be-used)
11. [References](#references)

---

## Executive Summary

This project employs machine learning models to classify online text posts into four mental health categories: **Suicidal, Depression, Anxiety, and Normal**. The dataset contains 41,174 labeled training posts and 8,436 unlabeled test posts. Using statistical methods and machine learning algorithms, the text data is transformed into numerical representations that allow supervised learning models to detect linguistic patterns associated with different mental health categories. Model performance is evaluated using classification metrics such as precision, recall, and F1 score, with particular attnetion to correctly identifying high-risk categories such as Suicidal posts.

---

## Problem Definition

**Objective**: Given a text post x, predict its mental health category y &isin; {Suicidal, Depression, Anxiety, Normal}

**Research Question**: How effectively can supervised ML models detect linguistic patterns associated with mental illness in online text posts and reliably flag high-risk content for early intervention by mental health support services?

**Motivation**: Mental health signals often appear in online text communication. Automated text classification can assist in identifying patterns associated with psychological distress and may support research in mental health monitoring.

**Important Disclaimer**: The dataset labels represent forum based annotations rather than clinical diagnoses. Therefore, the model learns linguistic patterns associated with mental health conditions rather than diagnosing medical conditions. Predictions should be interpreted as text classification outputs rather than clinical assessments.

---

## Exploratory Data Analysis
### Data Overview

| File | Description |
|----------|---------------------|
| **train.csv** | Labeled dataset used for model training |
| **test.csv** | Unlabeled dataset used for model prediction |

Each record contains:
| Column | Description |
|----------|----------------|
| *id* | Unique identifier for the post |
| *text* | Raw text content of the post |
| *status* | Mental health label (training set only) |

The labels are:
| Class | Description |
|----------|----------------|
| **Suicidal** | Posts expressing suicidal thoughts |
| **Depression** | Posts describing depressive symptoms |
| **Anxiety** | Posts expressing worry or panic |
| **Normal** | Posts not related to mental health distress |

The data consists of two parts:
- **Training set**: 41,174 labeled posts
- **Test set**: 8,436 posts for prediction
  

### Class Distribution

| Class | Label Count | Percentage |
|----------|---------|--------|
| Suicidal | 9,102 | 22.1% |
| Depression | 12,397 | 30.1% |
| Anxiety | 3,394 | 8.2% |
| Normal | 16,281 | 39.5% |
![Class Distribution Visualization](figures/class_distribution.png)

The dataset shows *class imbalance*, with Anxiety class being the smallest category.


### Word Frequency (Word clouds)
- image to be attached

Prominenet words include per category:
- **Suicidal posts**: *die, kill, fuck, life, time*
- **Depression posts**: *feel, know, time, life, people*
- **Anxiety posts**: *feel, know, time, people, panic*
- **Normal posts**: *mom, school, work, friend, day*

Word clouds were generated for each mental health category to visualize frequently occurring words in the dataset. Suicidal posts prominently contain terms related to death and distress, while depression and anxiety posts emphasize emotional and cognitive expressions such as “feel” and “know.” In contrast, normal posts tend to include more neutral everyday topics such as school, work, and family. These observations suggest that linguistic differences exist across categories, supporting the feasibility of using text-based machine learning methods for classification.

### Embedding Visualization (t-SNE / UMAP)
To qualitatively assess whether transformer-based sentence embeddings capture meaningful differences between mental health categories, we projected the high-dimensional embedding vectors into two dimensions using t-SNE/UMAP. The visualization showed partial clustering by class, with suicidal and normal posts exhibiting clearer separation, while depression and anxiety posts displayed greater overlap. This supports the view that embedding representations contain useful semantic information for downstream classification.

---

## Text Preprocessing and Feature Selection
### Preprocessing Steps
- lowercasing text
- removing URLs
- tokenization
- optional stopword removal

### Feature Representation
- TF-IDF: Converts text into sparse vectors representing word importance.
- Embeddings: Sentence embeddings or transformer outputs capture semantic meaning.

### Linguistic Feature Engineering
In addition to transformer-based sentence embeddings, we incorporated several linguistic features inspired by psychological language research. These features included first-person pronoun frequency, negative emotion word usage, absolutist terms, and message length. Prior research has shown that individuals experiencing suicidal ideation often exhibit distinctive linguistic patterns, such as increased self-referential language and absolutist thinking. Combining semantic embeddings with these linguistic features improved the model’s ability to detect suicide-related language.
- punctuation counts
- post length
- sentiment score

---

## Model Development
### Binary Classification Models
- Baseline:
- Improved:
- Stronger:
- Advanced:
  
### Multi-class Classification Models
- Baseline:
- Improved:
- Stronger:
- Advanced:

### Model Training Procedure
- Validation split
- Cross-validation
- Hyperparameter tuning using grid search
- Class weighting
  
---

## Model Evaluation

### Key Metrics
| Metric | Purpose |
|----------|----------------|
| Accruacy | Overall correctness |
| Precision | Control false positives |
| Recall | Detect true cases |
| F1 score | Balance precision and recall |

---

## Performance Summary

Figures, tables, justification of the best chosen model

---

## Suicide Risk Detection Framework

---

## Limitations and Future Work
### Limitations
- Labels are not clinical diagnoses
- Dataset may contain labeling noise
- Users from online platform (i.e. Reddit) may not represent general population
- Model predictions should not be used for medical decisions

### Potential Improvements
- Transformer models
- Larger datasets
- Context-aware models
- User-level conversation modeling
- Fairness and bias analysis

---

## How It Can Be Used

### For Modeling Experts
- **Feature Engineering**: Extract linguistic signals such as word frequencies, sentiment markers, punctuation patterns, and text length as model features.
- **Model Training**: Train supervised classification models (i.e., logistic regression, SVM, or transformer-based models) to predict mental health categories from text.
- **Performance Benchmarking**: Compare alternative models using macro F1 score and per-class recall to ensure balanced performance across categories.
- **Model Interpretation**: Analyze feature importance or attention patterns to identify which linguistic signals contribute most strongly to predictions.
  
### For Platform Moderators and Safety Teams
- **Content Triage**: Flag posts predicted as Suicidal or high-risk categories for human review and moderation.
- **Early Warning Signals**: Identify posts that may indicate psychological distress and prioritize them for support interventions.
- **Moderator Assistance**: Provide automated classification scores to help moderators quickly identify potentially concerning content.
- **Escalation Workflow**: Integrate model predictions into moderation pipelines to guide manual review and support actions.
  
### For Researchers and Mental Health Analysts
- **Language Pattern Analysis**: Study how linguistic features differ across suicidal, depression, anxiety, and normal posts.
- **Large-Scale Text Analysis**: Analyze thousands of posts to identify trends in mental health discussions across online communities.
- **Trend Monitoring**: Track changes in mental health–related language patterns over time or during major societal events.
- **Behavioral Insights**: Examine how emotional tone, vocabulary, and writing style correlate with different mental health states.

### For Data Scientists and NLP Practitioners**
- **Benchmark Dataset**: Use the dataset to evaluate new NLP classification methods for emotionally sensitive text.
- **Model Comparison**: Test different architectures such as TF-IDF models, neural networks, and transformer-based models.
- **Explainability Research**: Apply interpretability methods (i.e., SHAP or attention analysis) to understand model decisions.
- **Class Imbalance Strategies**: Experiment with sampling techniques, class weighting, and threshold tuning for imbalanced datasets.
---

## References
- Kaggle. (2026). Classification of Mental Health Status Dataset. Retrieved from https://www.kaggle.com/competitions/classification-of-mental-health-status/data
- Relevant research papers
- Any tools used

---

