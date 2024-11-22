import pandas as pd
import numpy as np

# Load training and testing data
train_df = pd.read_csv('train_ratings.csv')
test_df = pd.read_csv('test_ratings.csv')

# Convert training data into a user-item matrix for predictions
user_item_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=np.nan)

# Predict ratings using a basic average approach (placeholder for the actual model)
user_mean_ratings = user_item_matrix.mean(axis=1)
predicted_ratings_matrix = user_item_matrix.apply(
    lambda x: x.fillna(user_mean_ratings[x.name]), axis=1
)

# Convert predictions back to a DataFrame
predicted_ratings_df = pd.DataFrame(predicted_ratings_matrix.values, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Create top-10 recommendation lists for each user
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

# Generate recommendations
recommendations = create_recommendations(predicted_ratings_df, train_df)

# Evaluate recommendations
def evaluate_recommendations(recommendations, test_df, top_n=10):
    precision_list = []
    recall_list = []
    ndcg_list = []

    for user, recommended_items in recommendations.items():
        # Get the actual items in the test set for this user
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
        
        # Append metrics
        precision_list.append(precision)
        recall_list.append(recall)
        ndcg_list.append(ndcg)

    # Calculate averages
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f_measure = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_ndcg = np.mean(ndcg_list)

    return avg_precision, avg_recall, avg_f_measure, avg_ndcg

# Evaluate the recommendations
precision, recall, f_measure, ndcg = evaluate_recommendations(recommendations, test_df)

# Print results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-measure: {f_measure:.4f}")
print(f"NDCG: {ndcg:.4f}")
