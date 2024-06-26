# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Step 2: Load and Clean the Data
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")

data = pd.merge(ratings, books, on='ISBN')
data = pd.merge(data, users, on='User-ID')
data = data[data['Book-Rating'] > 0]
user_counts = data['User-ID'].value_counts()
book_counts = data['ISBN'].value_counts()
data = data[data['User-ID'].isin(user_counts[user_counts >= 200].index)]
data = data[data['ISBN'].isin(book_counts[book_counts >= 100].index)]

# Step 3: Create the User-Book Matrix
user_book_matrix = data.pivot(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)
user_book_matrix_sparse = csr_matrix(user_book_matrix.values)

# Step 4: Train the KNN Model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_matrix_sparse)

# Step 5: Create the Recommendation Function
def get_recommends(book_title):
    book_index = list(user_book_matrix.columns).index(book_title)
    distances, indices = model_knn.kneighbors(user_book_matrix.iloc[:, book_index].values.reshape(1, -1), n_neighbors=6)
    
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        book_name = user_book_matrix.columns[indices.flatten()[i]]
        distance = distances.flatten()[i]
        recommended_books.append([book_name, distance])
    
    return [book_title, recommended_books]

# Step 6: Test the Function
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
