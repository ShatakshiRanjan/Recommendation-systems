import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load training and testing data
train_df = pd.read_csv('train_ratings.csv')
test_df = pd.read_csv('test_ratings.csv')

# Pivot training data into a user-item matrix
user_item_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=np.nan)

# Normalize user-item matrix by subtracting user mean
user_mean_ratings = user_item_matrix.mean(axis=1)
user_item_matrix_normalized = user_item_matrix.sub(user_mean_ratings, axis=0)

# Compute Pearson similarity between items
def pearson_similarity(matrix):
    matrix = np.nan_to_num(matrix)  # Handle NaN values
    mean_centered = matrix - matrix.mean(axis=0, keepdims=True)
    norm_matrix = np.linalg.norm(mean_centered, axis=0)
    similarity = np.dot(mean_centered.T, mean_centered) / (norm_matrix[:, None] * norm_matrix[None, :])
    np.fill_diagonal(similarity, 0)  # No self-similarity
    return similarity

# Create item similarity matrix
item_similarity = pearson_similarity(user_item_matrix_normalized.values)

# Apply a threshold to filter low-similarity items
threshold = 0.5
item_similarity[item_similarity < threshold] = 0

# Predict ratings using normalized matrix
def predict_ratings(user_item_matrix_normalized, similarity_matrix):
    user_item_matrix = np.nan_to_num(user_item_matrix_normalized)
    similarity_matrix = np.nan_to_num(similarity_matrix)
    
    numerator = np.dot(user_item_matrix, similarity_matrix)
    denominator = np.abs(similarity_matrix).sum(axis=1)
    denominator[denominator == 0] = 1  # Avoid division by zero
    predicted_ratings = numerator / denominator
    return predicted_ratings

# Get predicted ratings
predicted_ratings_matrix = predict_ratings(user_item_matrix_normalized.values, item_similarity)

# Denormalize predictions by adding user mean ratings back
predicted_ratings_matrix = predicted_ratings_matrix + user_mean_ratings.values[:, None]
predicted_ratings_df = pd.DataFrame(predicted_ratings_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Evaluate predictions
def evaluate_predictions(test_df, predicted_ratings_df):
    predictions = []
    actuals = []
    global_mean = train_df['rating'].mean()
    
    for _, row in test_df.iterrows():
        user = row['userId']
        movie = row['movieId']
        actual = row['rating']
        
        if user in predicted_ratings_df.index and movie in predicted_ratings_df.columns:
            predicted = predicted_ratings_df.loc[user, movie]
            if np.isnan(predicted):  # Handle missing predictions
                predicted = user_mean_ratings[user] if user in user_mean_ratings else global_mean
        else:
            predicted = global_mean
        
        predictions.append(predicted)
        actuals.append(actual)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions, squared=False)
    return mae, rmse

# Calculate MAE and RMSE
mae, rmse = evaluate_predictions(test_df, predicted_ratings_df)

# Print results
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
