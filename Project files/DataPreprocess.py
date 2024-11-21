import pandas as pd
from sklearn.model_selection import train_test_split

# File paths and Loading CSV files
links_file_path = './ml-latest-small/links.csv'
movies_file_path = './ml-latest-small/movies.csv'
ratings_file_path = './ml-latest-small/ratings.csv'
tags_file_path = './ml-latest-small/tags.csv'

links_df = pd.read_csv(links_file_path)
movies_df = pd.read_csv(movies_file_path)
ratings_df = pd.read_csv(ratings_file_path)
tags_df = pd.read_csv(tags_file_path)

# Merge movies and ratings datasets
ratings_with_movies = pd.merge(ratings_df, movies_df, on='movieId', how='left')

# Merge tags dataset (optional)
tags_aggregated = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
ratings_with_tags = pd.merge(ratings_with_movies, tags_aggregated, on='movieId', how='left')

# Enrich with IMDb links
ratings_enriched = pd.merge(ratings_with_tags, links_df[['movieId', 'imdbId']], on='movieId', how='left')

# Split genres into binary columns
genres_split = ratings_enriched['genres'].str.get_dummies(sep='|')

# Add the genre features to the enriched dataset
ratings_enriched = pd.concat([ratings_enriched, genres_split], axis=1)

# Preview dataset with genres
print("Dataset with genres added:")
print(ratings_enriched[['movieId', 'genres'] + list(genres_split.columns)].head())

# Group by userId and split into training and testing sets
train_list = []
test_list = []

for _, user_data in ratings_enriched.groupby('userId'):
    # Split each user's ratings into 80% train and 20% test
    train, test = train_test_split(user_data, test_size=0.2, random_state=42)
    train_list.append(train)
    test_list.append(test)

# Combine all train and test splits
train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# Verify the split
print(f"Total Ratings: {len(ratings_enriched)}")
print(f"Training Ratings: {len(train_df)} ({len(train_df) / len(ratings_enriched) * 100:.2f}%)")
print(f"Testing Ratings: {len(test_df)} ({len(test_df) / len(ratings_enriched) * 100:.2f}%)")

# Save the train and test datasets
train_df.to_csv('train_ratings.csv', index=False)
test_df.to_csv('test_ratings.csv', index=False)

print("Training and testing datasets saved.")



