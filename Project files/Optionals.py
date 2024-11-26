import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

train_df = pd.read_csv('trainData.csv')
test_df = pd.read_csv('testData.csv')
moviesCSV = './ml-latest-small/movies.csv'
movies_df = pd.read_csv(moviesCSV)
userItemMatrix = train_df.pivot_table(index='userId', columns='movieId', values='rating')

# Function to compute item similarity
def ItemSimilarity(userItemMatrix):
    matrix = userItemMatrix.fillna(0).values
    similarity = cosine_similarity(matrix.T)
    np.fill_diagonal(similarity, 0)
    return pd.DataFrame(similarity, index=userItemMatrix.columns, columns=userItemMatrix.columns)

item_similarity = ItemSimilarity(userItemMatrix)

# Generate recommendations
def genRecommendation(predicted_ratings_df, train_df, top_n=10):
    recommendations = {}
    for user in predicted_ratings_df.index:
        ratedItems = train_df[train_df['userId'] == user]['movieId'].values
        unratedItems = [item for item in predicted_ratings_df.columns if item not in ratedItems]
        predictions = predicted_ratings_df.loc[user, unratedItems]
        if not predictions.empty:
            top_items = predictions.nlargest(top_n).index.tolist()
            recommendations[user] = top_items
        else:
            recommendations[user] = []
    return recommendations

predRatingMatrix = userItemMatrix.fillna(0).dot(item_similarity)
recommendations = genRecommendation(predRatingMatrix, train_df)

def recommendationToFile(recommendations, movies_df, file_name="recommendations.txt"):
    """
    Save user recommendations to a text file.
    """
    with open(file_name, "w") as file:
        file.write("User Recommendations:\n")
        for user, movie_ids in recommendations.items():
            movie_titles = movies_df[movies_df['movieId'].isin(movie_ids)]['title'].tolist()
            file.write(f"User {user}:\n")
            file.write(f"  Recommended Movies: {', '.join(movie_titles)}\n\n")
    print(f"Recommendations have been saved to {file_name}.")

# Fairness: Genre Diversity
def genre_diversity(recommendations, movies_df):
    diversity = {}
    for user, recs in recommendations.items():
        genres = movies_df[movies_df['movieId'].isin(recs)]['genres'].dropna().str.split('|').explode()
        diversity[user] = genres.value_counts().to_dict()
    return diversity

def genreToFile(diversity, file_name="genre_diversity.txt"):
    with open(file_name, "w") as file:
        file.write("Genre Diversity of Recommendations:\n")
        for user, genres in diversity.items():
            file.write(f"User {user}:\n")
            for genre, count in genres.items():
                file.write(f"  {genre}: {count}\n")
            file.write("\n")
    print(f"Genre diversity has been saved to {file_name}.")

# Explanation function
def explainReccs(user_id, movie_id, train_df, item_similarity, movies_df):
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if movie_row.empty:
        return f"Movie {movie_id} is not found in the dataset."
    
    movie_genres = movie_row['genres'].values[0]
    user_ratings = train_df[train_df['userId'] == user_id]
    user_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].values

    similarRating = []
    for rated_movie in user_rated_movies:
        if rated_movie in item_similarity.columns and movie_id in item_similarity.index:
            similarity_score = item_similarity.loc[movie_id, rated_movie]
            if similarity_score > 0:
                similarRating.append((rated_movie, similarity_score))

    similarRating = sorted(similarRating, key=lambda x: x[1], reverse=True)
    if similarRating:
        similar_movie_ids = [sm[0] for sm in similarRating[:3]]
        similar_movie_titles = movies_df[movies_df['movieId'].isin(similar_movie_ids)]['title'].tolist()
        return (f"Movie {movie_id} is recommended because it shares the genres '{movie_genres}' "
                f"with movies you rated highly (e.g., {', '.join(similar_movie_titles)}).")
    else:
        return (f"Movie {movie_id} is recommended based on its genres '{movie_genres}' "
                "and your overall rating patterns.")

# Privacy Protection: Anonymize data
def anonymize_data(movies_df, train_df, user_ids_to_anonymize=None):
    anonymized_movies = movies_df.copy()
    anonymized_movies['title'] = anonymized_movies['movieId']
    
    anonymized_train = train_df.copy()
    if user_ids_to_anonymize:
        unique_users = user_ids_to_anonymize
    else:
        unique_users = anonymized_train['userId'].unique()
    
    user_id_mapping = {user_id: f"User_{i}" for i, user_id in enumerate(unique_users, start=1)}
    anonymized_train['userId'] = anonymized_train['userId'].map(user_id_mapping)
    
    return anonymized_movies, anonymized_train, user_id_mapping

# Filter recommendations by genre
def filterByGenre(user_id, recommendations, genre_filter, movies_df):
    filtered_recommendations = []
    recommended_movies = recommendations.get(user_id, [])
    for movie_id in recommended_movies:
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if not movie_row.empty:
            genres = movie_row['genres'].values[0]
            if genre_filter in genres.split('|'):
                filtered_recommendations.append(movie_id)
    return filtered_recommendations

# Save anonymized data to a text file
def saveAnonData(anonymized_movies, anonymized_train, file_name="anonymized_data.txt"):
    with open(file_name, "w") as file:
        file.write("Anonymized Movies Data:\n")
        file.write(anonymized_movies[['movieId', 'title']].to_string(index=False))
        file.write("\n\nAnonymized Training Data:\n")
        file.write(anonymized_train.to_string(index=False))
    print(f"Anonymized data has been saved to {file_name}.")

# Main menu for optional tasks
def main():
    while True:
        print("\nChoose a Trustworthiness Enhancement Task:")
        print("1. Transparency and Explainability")
        print("2. Fairness and Unbiases")
        print("3. Controllability")
        print("4. Privacy Protection")
        print("5. Exit")
        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            user_id = int(input("Enter user ID: "))
            movie_id = int(input("Enter movie ID: "))
            explanation = explainReccs(user_id, movie_id, train_df, item_similarity, movies_df)
            print("\nExplanation:\n", explanation)

        elif choice == "2":
            diversity = genre_diversity(recommendations, movies_df)
            genreToFile(diversity)

        elif choice == "3":
            user_id = int(input("Enter user ID: "))
            genre_filter = input("Enter genre to filter (e.g., Comedy, Action): ").strip()
            filtered_recs = filterByGenre(user_id, recommendations, genre_filter, movies_df)
            if filtered_recs:
                filtered_titles = movies_df[movies_df['movieId'].isin(filtered_recs)]['title'].tolist()
                print(f"\nFiltered Recommendations for User {user_id} by Genre '{genre_filter}':\n{', '.join(filtered_titles)}")
            else:
                print(f"\nNo recommendations found for User {user_id} in Genre '{genre_filter}'.")

        elif choice == "4":
            user_ids_to_anonymize = input("Enter user IDs to anonymize (comma-separated) or press Enter to anonymize all: ")
            if user_ids_to_anonymize:
                user_ids_to_anonymize = list(map(int, user_ids_to_anonymize.split(',')))
            else:
                user_ids_to_anonymize = None
            
            anonymized_movies, anonymized_train, user_id_mapping = anonymize_data(movies_df, train_df, user_ids_to_anonymize)
            saveAnonData(anonymized_movies, anonymized_train)
            print("\nAnonymization completed. User ID mapping:")
            for original, anonymized in user_id_mapping.items():
                print(f"  Original User ID: {original} -> Anonymized User ID: {anonymized}")

            recommendationToFile(recommendations, movies_df)

        elif choice == "5":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
