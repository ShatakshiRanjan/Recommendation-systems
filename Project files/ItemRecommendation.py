import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load training and testing data
train_df = pd.read_csv('trainData.csv')
test_df = pd.read_csv('testData.csv')
user_item_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating')

# --------------------------------------------------------------------------------
# Collaborative Filtering (Item-Item Similarity)
# --------------------------------------------------------------------------------
def compute_item_similarity(user_item_matrix, similarity_threshold=0.1):
    """
    Compute item-item similarity using cosine similarity and apply a threshold.
    """
    matrix = user_item_matrix.fillna(0).values  # Replace NaN with 0 for similarity computation
    similarity = cosine_similarity(matrix.T)  # Compute similarity between items
    np.fill_diagonal(similarity, 0)  # Avoid self-similarity
    similarity[similarity < similarity_threshold] = 0  # Apply threshold
    return pd.DataFrame(similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

item_similarity = compute_item_similarity(user_item_matrix, similarity_threshold=0.2)

def predict_ratings_collaborative(user_item_matrix, item_similarity):
    """
    Predict ratings using collaborative filtering.
    """
    user_item_filled = user_item_matrix.fillna(0).values
    numerator = np.dot(user_item_filled, item_similarity.values)
    denominator = np.abs(item_similarity.values).sum(axis=1)
    denominator[denominator == 0] = 1  # Avoid division by zero
    predictions = numerator / denominator
    return pd.DataFrame(predictions, index=user_item_matrix.index, columns=user_item_matrix.columns)

collaborative_predictions = predict_ratings_collaborative(user_item_matrix, item_similarity)

# --------------------------------------------------------------------------------
# User-Specific Average Predictions
# --------------------------------------------------------------------------------
user_mean_ratings = user_item_matrix.mean(axis=1)

def predict_ratings_user_mean(user_item_matrix, user_mean_ratings):
    """
    Predict ratings using user-specific averages.
    """
    predictions = user_item_matrix.apply(
        lambda row: row.fillna(user_mean_ratings[row.name]), axis=1
    )
    return predictions

mean_predictions = predict_ratings_user_mean(user_item_matrix, user_mean_ratings)

# --------------------------------------------------------------------------------
# Hybrid Predictions (Blend Collaborative & User Averages)
# --------------------------------------------------------------------------------
def predict_ratings_hybrid(collaborative_predictions, mean_predictions, alpha=0.8, beta=0.2):
    """
    Blend collaborative filtering predictions with user-specific averages.
    """
    return alpha * collaborative_predictions + beta * mean_predictions

hybrid_predictions = predict_ratings_hybrid(collaborative_predictions, mean_predictions, alpha=0.85, beta=0.15)

# --------------------------------------------------------------------------------
# Generate Top-N Recommendations
# --------------------------------------------------------------------------------
def create_recommendations(predicted_ratings_df, train_df, top_n=10):
    recommendations = {}
    for user in predicted_ratings_df.index:
        # Get items the user has not rated in the training set
        rated_items = train_df[train_df['userId'] == user]['movieId'].values
        unrated_items = [item for item in predicted_ratings_df.columns if item not in rated_items]
        
        # Predict ratings for unrated items
        predictions = predicted_ratings_df.loc[user, unrated_items]
        
        # Rank items by predicted ratings and get the top N
        top_items = predictions.nlargest(top_n).index.tolist()
        recommendations[user] = top_items
    return recommendations

recommendations = create_recommendations(hybrid_predictions, train_df)

def evaluate_recommendations(recommendations, test_df, top_n=10):
    precision_list = []
    recall_list = []
    ndcg_list = []

    for user, recommended_items in recommendations.items():
        relevant_items = test_df[test_df['userId'] == user]['movieId'].values
        
        # Calculate Precision and Recall
        relevant_recommended_items = [item for item in recommended_items if item in relevant_items]
        precision = len(relevant_recommended_items) / top_n
        recall = len(relevant_recommended_items) / len(relevant_items) if len(relevant_items) > 0 else 0
        
        # Calculate NDCG
        dcg = sum(
            [1 / np.log2(idx + 2) for idx, item in enumerate(recommended_items) if item in relevant_items]
        )
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(relevant_items), top_n))])
        ndcg = dcg / idcg if idcg > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        ndcg_list.append(ndcg)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f_measure = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_ndcg = np.mean(ndcg_list)

    return avg_precision, avg_recall, avg_f_measure, avg_ndcg

precision, recall, f_measure, ndcg = evaluate_recommendations(recommendations, test_df)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-measure: {f_measure:.4f}")
print(f"NDCG: {ndcg:.4f}")
