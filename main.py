import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load dataset
movies = pd.read_csv("movies.csv")

# fill missing values
movies = movies.fillna('')

# combine important columns
movies["tags"] = movies["genres"] + movies["keywords"]

# convert text to numbers
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies["tags"]).toarray()

# similarity
similarity = cosine_similarity(vectors)

# function
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(movies.iloc[i[0]].title)

# test
movie_name = input("Enter movie name: ")
recommend(movie_name)