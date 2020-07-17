import pandas as pd

metadata = pd.read_csv('data/movies_metadata.csv', low_memory=False)

# C is average rating for all movies in dataset
C = metadata['vote_average'].mean()

# m is set to 90 percentile of vote counts, to filter those without enough number of votes
m = metadata['vote_count'].quantile(0.90)

# filter qualified movies that have more vote count than m
qualified_movies = metadata.copy().loc[metadata['vote_count'] >= m]


def weighted_rating(x, m=m, c=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m)) * R + (m / (v + m)) * c


# make score field that contains weighted ratings
qualified_movies['score'] = qualified_movies.apply(weighted_rating, axis=1)

# sort movies by score
qualified_movies = qualified_movies.sort_values('score', ascending=False)

# print top 20 movies from recommendation value
print(qualified_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))