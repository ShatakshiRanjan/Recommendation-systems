import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
train_df = pd.read_csv('trainData.csv')
test_df = pd.read_csv('testData.csv')
moviesCSV = './ml-latest-small/movies.csv'
movies_df = pd.read_csv(moviesCSV)

# Generate user-item matrix
user_item_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating')

# Function to compute item similarity
def compute_item_similarity(user_item_matrix):
    matrix = user_item_matrix.fillna(0).values
    similarity = cosine_similarity(matrix.T)
    np.fill_diagonal(similarity, 0)
    return pd.DataFrame(similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Compute item similarity
item_similarity = compute_item_similarity(user_item_matrix)

# Generate recommendations
def create_recommendations(predicted_ratings_df, train_df, top_n=10):
    recommendations = {}
    for user in predicted_ratings_df.index:
        rated_items = train_df[train_df['userId'] == user]['movieId'].values
        unrated_items = [item for item in predicted_ratings_df.columns if item not in rated_items]
        predictions = predicted_ratings_df.loc[user, unrated_items]
        if not predictions.empty:
            top_items = predictions.nlargest(top_n).index.tolist()
            recommendations[user] = top_items
        else:
            recommendations[user] = []
    return recommendations

# Generate recommendations for the user
predicted_ratings_matrix = user_item_matrix.fillna(0).dot(item_similarity)
recommendations = create_recommendations(predicted_ratings_matrix, train_df)

# Fairness: Genre Diversity
def genre_diversity(recommendations, movies_df):
    """
    Calculate the genre diversity of the recommendations for each user.
    """
    diversity = {}
    for user, recs in recommendations.items():
        # Extract genres for recommended movies
        genres = movies_df[movies_df['movieId'].isin(recs)]['genres'].dropna().str.split('|').explode()
        diversity[user] = genres.value_counts().to_dict()  # Count occurrences of each genre
    return diversity

# Save genre diversity to a text file
def save_genre_diversity_to_file(diversity, file_name="genre_diversity.txt"):
    with open(file_name, "w") as file:
        file.write("Genre Diversity of Recommendations:\n")
        for user, genres in diversity.items():
            file.write(f"User {user}:\n")
            for genre, count in genres.items():
                file.write(f"  {genre}: {count}\n")
            file.write("\n")
    print(f"Genre diversity has been saved to {file_name}.")

# Explanation function
def explain_recommendation(user_id, movie_id, train_df, item_similarity, movies_df):
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if movie_row.empty:
        return f"Movie {movie_id} is not found in the dataset."
    
    movie_genres = movie_row['genres'].values[0]
    user_ratings = train_df[train_df['userId'] == user_id]
    user_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].values

    similar_movies = []
    for rated_movie in user_rated_movies:
        if rated_movie in item_similarity.columns and movie_id in item_similarity.index:
            similarity_score = item_similarity.loc[movie_id, rated_movie]
            if similarity_score > 0:
                similar_movies.append((rated_movie, similarity_score))

    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    if similar_movies:
        similar_movie_ids = [sm[0] for sm in similar_movies[:3]]
        similar_movie_titles = movies_df[movies_df['movieId'].isin(similar_movie_ids)]['title'].tolist()
        return (f"Movie {movie_id} is recommended because it shares the genres '{movie_genres}' "
                f"with movies you rated highly (e.g., {', '.join(similar_movie_titles)}).")
    else:
        return (f"Movie {movie_id} is recommended based on its genres '{movie_genres}' "
                "and your overall rating patterns.")

# Filter recommendations by genre
def filter_recommendations_by_genre(user_id, recommendations, genre_filter, movies_df):
    filtered_recommendations = []
    recommended_movies = recommendations.get(user_id, [])
    for movie_id in recommended_movies:
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if not movie_row.empty:
            genres = movie_row['genres'].values[0]
            if genre_filter in genres.split('|'):
                filtered_recommendations.append(movie_id)
    return filtered_recommendations

# Main menu for optional tasks
def main():
    while True:
        print("\nChoose a Trustworthiness Enhancement Task:")
        print("1. Transparency and Explainability")
        print("2. Fairness and Unbiases")
        print("3. Controllability")
        print("4. Privacy Protection")
        print("5. Robustness and Anti-attacks")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            user_id = int(input("Enter user ID: "))
            movie_id = int(input("Enter movie ID: "))
            explanation = explain_recommendation(user_id, movie_id, train_df, item_similarity, movies_df)
            print("\nExplanation:\n", explanation)

        elif choice == "2":
            # Compute genre diversity for the recommendations
            diversity = genre_diversity(recommendations, movies_df)
            save_genre_diversity_to_file(diversity)

        elif choice == "3":
            user_id = int(input("Enter user ID: "))
            genre_filter = input("Enter genre to filter (e.g., Comedy, Action): ").strip()
            filtered_recs = filter_recommendations_by_genre(user_id, recommendations, genre_filter, movies_df)
            if filtered_recs:
                filtered_titles = movies_df[movies_df['movieId'].isin(filtered_recs)]['title'].tolist()
                print(f"\nFiltered Recommendations for User {user_id} by Genre '{genre_filter}':\n{', '.join(filtered_titles)}")
            else:
                print(f"\nNo recommendations found for User {user_id} in Genre '{genre_filter}'.")

        elif choice == "6":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

