import pandas as pd
from sklearn.model_selection import train_test_split

linksCSV = './ml-latest-small/links.csv'
moviesCSV = './ml-latest-small/movies.csv'
ratingsCSV = './ml-latest-small/ratings.csv'
tagsCSV = './ml-latest-small/tags.csv'

links = pd.read_csv(linksCSV)
movies = pd.read_csv(moviesCSV)
ratings = pd.read_csv(ratingsCSV)
tags = pd.read_csv(tagsCSV)

# Merge datasets
RatingMoviesComb = pd.merge(ratings, movies, on='movieId', how='left')
tagsAggregated = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
TagsRatingComb = pd.merge(RatingMoviesComb, tagsAggregated, on='movieId', how='left')
ratingMerge = pd.merge(TagsRatingComb, links[['movieId', 'imdbId']], on='movieId', how='left')
splitGenre = ratingMerge['genres'].str.get_dummies(sep='|')
ratingMerge = pd.concat([ratingMerge, splitGenre], axis=1)
ratingMerge.dropna(subset=['rating', 'movieId'], inplace=True)

trainList = []
testList = []
for _, userData in ratingMerge.groupby('userId'):
    train, test = train_test_split(userData, test_size=0.2, random_state=42)
    trainList.append(train)
    testList.append(test)

# Combine splits
train_df = pd.concat(trainList).reset_index(drop=True)
test_df = pd.concat(testList).reset_index(drop=True)

train_df.to_csv('trainData.csv', index=False)
test_df.to_csv('testData.csv', index=False)
print(f"Total Ratings: {len(ratingMerge)}")
print(f"Training Ratings: {len(train_df)} ({len(train_df) / len(ratingMerge) * 100:.2f}%)")
print(f"Testing Ratings: {len(test_df)} ({len(test_df) / len(ratingMerge) * 100:.2f}%)")



