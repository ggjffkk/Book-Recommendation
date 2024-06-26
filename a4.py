def get_recommends(book_title):
    book_index = list(user_book_matrix.columns).index(book_title)
    distances, indices = model_knn.kneighbors(user_book_matrix.iloc[:, book_index].values.reshape(1, -1), n_neighbors=6)
    
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        book_name = user_book_matrix.columns[indices.flatten()[i]]
        distance = distances.flatten()[i]
        recommended_books.append([book_name, distance])
    
    return [book_title, recommended_books]
