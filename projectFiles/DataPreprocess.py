import pandas as pd
from sklearn.model_selection import train_test_split

# File paths (adjust these paths as needed)
links_file_path = './ml-latest-small/links.csv'
movies_file_path = './ml-latest-small/movies.csv'
ratings_file_path = './ml-latest-small/ratings.csv'
tags_file_path = './ml-latest-small/tags.csv'

# Load CSV files into pandas DataFrames
links_df = pd.read_csv(links_file_path)
movies_df = pd.read_csv(movies_file_path)
ratings_df = pd.read_csv(ratings_file_path)
tags_df = pd.read_csv(tags_file_path)

# Function to split each user's data
def split_user_ratings(user_data):
    train, test = train_test_split(user_data, test_size=0.2, random_state=42)
    return train, test

# Group data by 'userId' and split ratings
train_list = []
test_list = []

for _, user_data in ratings_df.groupby('userId'):
    train, test = split_user_ratings(user_data)
    train_list.append(train)
    test_list.append(test)

# Combine all train and test splits
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

# Reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Training Dataset:\n", train_df.head())
print("Testing Dataset:\n", test_df.head())

print(f"Total Ratings: {len(ratings_df)}")
print(f"Training Ratings: {len(train_df)} ({len(train_df) / len(ratings_df) * 100:.2f}%)")
print(f"Testing Ratings: {len(test_df)} ({len(test_df) / len(ratings_df) * 100:.2f}%)")



