import pandas as pd

# File paths (adjust these paths as needed)
links_file_path = 'ml-latest-small/links.csv'
movies_file_path = 'ml-latest-small/movies.csv'
ratings_file_path = 'ml-latest-small/ratings.csv'
tags_file_path = 'ml-latest-small/tags.csv'

# Load CSV files into pandas DataFrames
links_df = pd.read_csv(links_file_path)
movies_df = pd.read_csv(movies_file_path)
ratings_df = pd.read_csv(ratings_file_path)
tags_df = pd.read_csv(tags_file_path)

# Display the first few rows of each DataFrame (optional)
print("Links Data:")
print(links_df.head())

print("\nMovies Data:")
print(movies_df.head())

print("\nRatings Data:")
print(ratings_df.head())

print("\nTags Data:")
print(tags_df.head())

