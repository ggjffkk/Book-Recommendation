user_book_matrix = data.pivot(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)
user_book_matrix_sparse = csr_matrix(user_book_matrix.values)
