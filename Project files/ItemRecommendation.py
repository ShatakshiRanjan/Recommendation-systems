import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

train_df = pd.read_csv('trainData.csv')
test_df = pd.read_csv('testData.csv')
UserItemMatrix = train_df.pivot_table(index='userId', columns='movieId', values='rating')

# --------------------------------------------------------------------------------
# Collaborative Filtering (Item-Item Similarity)
# --------------------------------------------------------------------------------
def itemSimilarity(UserItemMatrix, similarity_threshold=0.1):
    matrix = UserItemMatrix.fillna(0).values  
    similarity = cosine_similarity(matrix.T) 
    np.fill_diagonal(similarity, 0)  
    similarity[similarity < similarity_threshold] = 0  
    return pd.DataFrame(similarity, index=UserItemMatrix.columns, columns=UserItemMatrix.columns)

item_similarity = itemSimilarity(UserItemMatrix, similarity_threshold=0.2)

def collaborativeRatingPredict(UserItemMatrix, item_similarity):
    user_item_filled = UserItemMatrix.fillna(0).values
    numerator = np.dot(user_item_filled, item_similarity.values)
    denominator = np.abs(item_similarity.values).sum(axis=1)
    denominator[denominator == 0] = 1  # Avoid division by zero
    predictions = numerator / denominator
    return pd.DataFrame(predictions, index=UserItemMatrix.index, columns=UserItemMatrix.columns)

collaborative_predictions = collaborativeRatingPredict(UserItemMatrix, item_similarity)

# --------------------------------------------------------------------------------
# User-Specific Average Predictions
# --------------------------------------------------------------------------------
user_mean_ratings = UserItemMatrix.mean(axis=1)

def meanUserRatings(UserItemMatrix, user_mean_ratings):
    predictions = UserItemMatrix.apply(
        lambda row: row.fillna(user_mean_ratings[row.name]), axis=1
    )
    return predictions

mean_predictions = meanUserRatings(UserItemMatrix, user_mean_ratings)

# --------------------------------------------------------------------------------
# Hybrid Predictions (Blend Collaborative & User Averages)
# --------------------------------------------------------------------------------
def ratingHybrid(collaborative_predictions, mean_predictions, alpha=0.8, beta=0.2):
    return alpha * collaborative_predictions + beta * mean_predictions

hybrid_predictions = ratingHybrid(collaborative_predictions, mean_predictions, alpha=0.85, beta=0.15)

# --------------------------------------------------------------------------------
# Generate Top-N Recommendations
# --------------------------------------------------------------------------------
def createReccs(predicted_ratings_df, train_df, top_n=10):
    recommendations = {}
    for user in predicted_ratings_df.index:
        # Get items the user has not rated in the training set
        rated_items = train_df[train_df['userId'] == user]['movieId'].values
        unrated_items = [item for item in predicted_ratings_df.columns if item not in rated_items]
        
        # Predict ratings for unrated items
        predictions = predicted_ratings_df.loc[user, unrated_items]
        
        # Rank items by predicted ratings and get the top N
        topPicks = predictions.nlargest(top_n).index.tolist()
        recommendations[user] = topPicks
    return recommendations

recommendations = createReccs(hybrid_predictions, train_df)

def evalReccs(recommendations, test_df, top_n=10):
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

precision, recall, f_measure, ndcg = evalReccs(recommendations, test_df)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-measure: {f_measure:.4f}")
print(f"NDCG: {ndcg:.4f}")
