import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load training and testing datasets
train_df = pd.read_csv('trainData.csv')
test_df = pd.read_csv('testData.csv')

# Pivot training data into a user-item matrix
user_item_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating')

# Normalize user-item matrix by subtracting user mean
user_mean_ratings = user_item_matrix.mean(axis=1)
user_item_matrix_normalized = user_item_matrix.sub(user_mean_ratings, axis=0)

# Compute Pearson similarity
def pearson_similarity(matrix):
    matrix = np.nan_to_num(matrix)
    mean_centered = matrix - matrix.mean(axis=0, keepdims=True)
    norm_matrix = np.linalg.norm(mean_centered, axis=0)
    norm_matrix[norm_matrix == 0] = 1  # Avoid division by zero
    similarity = np.dot(mean_centered.T, mean_centered) / (norm_matrix[:, None] * norm_matrix[None, :])
    np.fill_diagonal(similarity, 0)
    return similarity

item_similarity = pearson_similarity(user_item_matrix_normalized.values)

# Apply dynamic threshold
threshold = np.percentile(item_similarity[item_similarity > 0], 25)  # Set threshold as 25th percentile
item_similarity[item_similarity < threshold] = 0

# Save item similarity matrix
np.save('itemSimilarity.npy', item_similarity)

# Predict ratings
def predict_ratings(user_item_matrix_normalized, similarity_matrix):
    user_item_matrix = np.nan_to_num(user_item_matrix_normalized)
    numerator = np.dot(user_item_matrix, similarity_matrix)
    denominator = np.abs(similarity_matrix).sum(axis=1)
    denominator[denominator == 0] = 1
    return numerator / denominator

predicted_ratings_matrix = predict_ratings(user_item_matrix_normalized.values, item_similarity)

# Denormalize predicted ratings
predicted_ratings_matrix = predicted_ratings_matrix + user_mean_ratings.values[:, None]
predicted_ratings_df = pd.DataFrame(predicted_ratings_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Save predicted ratings
predicted_ratings_df.to_csv('ratingPrediction.csv', index=True)

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
            predicted = predicted if not np.isnan(predicted) else global_mean
        else:
            predicted = global_mean
        
        predictions.append(predicted)
        actuals.append(actual)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return mae, rmse

mae, rmse = evaluate_predictions(test_df, predicted_ratings_df)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
