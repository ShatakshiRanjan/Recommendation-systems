import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_df = pd.read_csv('trainData.csv')
test_df = pd.read_csv('testData.csv')
UserItemMatrix = train_df.pivot_table(index='userId', columns='movieId', values='rating')

userMean = UserItemMatrix.mean(axis=1)
UserItemMatrix_normalized = UserItemMatrix.sub(userMean, axis=0)

# Compute Pearson similarity
def similarityPerson(matrix):
    matrix = np.nan_to_num(matrix)
    mean_centered = matrix - matrix.mean(axis=0, keepdims=True)
    norm_matrix = np.linalg.norm(mean_centered, axis=0)
    norm_matrix[norm_matrix == 0] = 1  # Avoid division by zero
    similarity = np.dot(mean_centered.T, mean_centered) / (norm_matrix[:, None] * norm_matrix[None, :])
    np.fill_diagonal(similarity, 0)
    return similarity

similarityItem = similarityPerson(UserItemMatrix_normalized.values)
threshold = np.percentile(similarityItem[similarityItem > 0], 25)  # Set threshold as 25th percentile
similarityItem[similarityItem < threshold] = 0
np.save('itemSimilarity.npy', similarityItem)

# Predict ratings
def ratingPrediction(UserItemMatrix_normalized, similarity_matrix):
    UserItemMatrix = np.nan_to_num(UserItemMatrix_normalized)
    num = np.dot(UserItemMatrix, similarity_matrix)
    denom = np.abs(similarity_matrix).sum(axis=1)
    denom[denom == 0] = 1
    return num / denom

ratingPredMatrix = ratingPrediction(UserItemMatrix_normalized.values, similarityItem)

ratingPredMatrix = ratingPredMatrix + userMean.values[:, None]
predicted_ratings_df = pd.DataFrame(ratingPredMatrix, index=UserItemMatrix.index, columns=UserItemMatrix.columns)

predicted_ratings_df.to_csv('ratingPrediction.csv', index=True)
def evaluatePrediction(test_df, predicted_ratings_df):
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

mae, rmse = evaluatePrediction(test_df, predicted_ratings_df)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
