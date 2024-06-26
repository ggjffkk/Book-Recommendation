import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the dataset
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")

# Merge the data
data = pd.merge(ratings, books, on='ISBN')
data = pd.merge(data, users, on='User-ID')

# Filter data
data = data[data['Book-Rating'] > 0]  # Consider only non-zero ratings
user_counts = data['User-ID'].value_counts()
book_counts = data['ISBN'].value_counts()

data = data[data['User-ID'].isin(user_counts[user_counts >= 200].index)]
data = data[data['ISBN'].isin(book_counts[book_counts >= 100].index)]
