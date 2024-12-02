# Recommendation System

## Introduction
This project implements a collaborative filtering-based recommendation system using the **MovieLens dataset**. It processes datasets, predicts ratings, generates recommendations, and evaluates their performance. Optional tasks include explainability, fairness, controllability, privacy protection, and robustness.

---

## Getting Started

### Prerequisites
- **Python**: Version 3.8 or later.
- **Required Libraries**: Install using the following command:
  ```bash
  pip install pandas numpy scikit-learn

## How to Run the Program

### Step 1: Data Preprocessing
Run the DataPreprocessing.py script to prepare the dataset:
```bash
python DataPreprocess.py
```
Output:
- train_ratings.csv: Processed training dataset
- test_ratings.csv: Processed testing dataset

### Step 2: Rating Prediction
Run the RatingPrediction.py script to predict user-item ratings and evaluate accuracy:
```bash
python RatingPrediction.py
```
Output:
- predicted_ratings.csv: Predicted ratings
- Evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)

### Step 3: Generate Recommendations
Run the ItemRecommendation.py script to generate top-N recommendations for users:
```bash
python ItemRecommendation.py
```
Output:
- Recommendations for users
- Evaluation metrics:
  - Precision
  - Recall
  - F-measure
  - Normalized Discounted Cumulative Gain (NDCG)
 
### Step 4: Optional Tasks
Run the OptionalTasks.py script to analyze the recommendation system:
```bash
python Optionals.py
```
Features:
- Explainability: Provides reasons for recommendations based on genres and tags.
- Fairness: Analyzes diversity and popularity bias in recommendations.
- Controllability: Allows filtering recommendations by specific genres.
- Privacy Protection: Anonymizes sensitive data like movie titles.



## Dataset
Link: https://grouplens.org/datasets/movielens/latest/

Download the MovieLens ml-latest-small dataset from MovieLens:
links.csv
movies.csv
ratings.csv
tags.csv

## Project Results
Slides: https://docs.google.com/presentation/d/1r65Cy6-eKkuGI6xonAQxm5vqnZ0_YV5kaQh9pkjvJF0/edit?usp=sharing


