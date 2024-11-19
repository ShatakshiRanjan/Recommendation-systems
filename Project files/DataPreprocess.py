import pandas as pd

# File paths
links_file_path = './ml-latest-small/links.csv'
movies_file_path = './ml-latest-small/movies.csv'
ratings_file_path = './ml-latest-small/ratings.csv'
tags_file_path = './ml-latest-small/tags.csv'

# Load CSV files
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

# Preview the enriched dataset
print(ratings_enriched.head())

# Split genres into binary columns
genres_split = ratings_with_movies['genres'].str.get_dummies(sep='|')

# Add the genre features to the enriched dataset
ratings_enriched = pd.concat([ratings_with_movies, genres_split], axis=1)

# Preview dataset with genres
print(ratings_enriched[['movieId', 'genres'] + list(genres_split.columns)].head())


