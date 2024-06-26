model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_matrix_sparse)
