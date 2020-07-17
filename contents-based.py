import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

credits_data = pd.read_csv('data/credits.csv')
keywords_data = pd.read_csv('data/keywords.csv')
metadata = pd.read_csv('data/movies_metadata.csv', low_memory=False)

# remove bad rows
metadata = metadata.drop([19730, 29503, 35587])

# convert id into integer
keywords_data['id'] = keywords_data['id'].astype('int')
credits_data['id'] = credits_data['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# merge keywords and credits into metadata

metadata = metadata.merge(credits_data, on='id')
metadata = metadata.merge(keywords_data, on='id')

features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names[:3] if len(names) > 3 else names

    return []


metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


metadata['soup'] = metadata.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(metadata['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


print(get_recommendations('The Dark Knight Rises'))