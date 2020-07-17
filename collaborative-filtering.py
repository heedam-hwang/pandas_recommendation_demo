import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('data/ratings_small.csv')
movies = pd.read_csv('data/movies_small.csv')

ratings.drop('timestamp', axis=1, inplace=True)
movies.drop('genres', axis=1, inplace=True)

# filter items which have ratings lower than 4.0
ratings['rating'] = ratings['rating'].astype('float')
ratings['userId'] = ratings['userId'].astype('int')
ratings = ratings.loc[ratings['rating'] >= 4.0]

# merge two dataset to create movie-user table
merge_table = pd.merge(ratings, movies, on='movieId')

# pivot table to get userId-movieId data
movie_user_table = merge_table.pivot_table('rating', index='userId', columns='title').fillna(0.0)
movie_user_table = movie_user_table.transpose()
similarity_matrix = cosine_similarity(movie_user_table)
item_based_cf = pd.DataFrame(data=similarity_matrix, index=movie_user_table.index, columns=movie_user_table.index)


# get recommendation for target title, 3 items for each item the user has preferred
def get_item_based_cf(title):
    return item_based_cf[title].sort_values(ascending=False)[:3]


while True:
    target_user_id = int(input("insert user id to get recommendations(0: break, 1~671: valid) :: "))
    if target_user_id == 0:
        break
    preferred_items = [x for x in merge_table.loc[merge_table['userId'] == target_user_id, 'title']]
    recommended_items = []

    for item in preferred_items:
        print(get_item_based_cf(item))
