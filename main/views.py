from django.shortcuts import render
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# Create your views here.

def home(request):
    return render(request, 'main.html')


def contact(request):
    return render(request, 'contact.html')


def prediction(request):
    query_index = request.GET['query']
    books = pd.read_csv('D:\\Python project\\booksforall\\main\\BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                     'imageUrlL']
    users = pd.read_csv('D:\\Python project\\booksforall\\main\\BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    users.columns = ['userID', 'Location', 'Age']
    ratings = pd.read_csv('D:\\Python project\\booksforall\\main\\BX-Book-Ratings.csv', sep=';', error_bad_lines=False,
                          encoding="latin-1")
    ratings.columns = ['userID', 'ISBN', 'bookRating']
    counts1 = ratings['userID'].value_counts()
    ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
    counts = ratings['bookRating'].value_counts()
    ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]
    # merge the two datasets on ISBN
    combine_book_rating = pd.merge(ratings, books, on='ISBN')
    columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    combine_book_rating.head()
    # to know the count of the ratings of the books, to eliminate the useless data
    combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])

    book_ratingCount = (combine_book_rating.
        groupby(by=['bookTitle'])['bookRating'].
        count().
        reset_index().
        rename(columns={'bookRating': 'totalRatingCount'})
    [['bookTitle', 'totalRatingCount']]
        )
    book_ratingCount.head()
    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle',
                                                             right_on='bookTitle', how='left')
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    # taking users from US and Canada only, because of the training time
    combined = rating_popular_book.merge(users, left_on='userID', right_on='userID', how='left')

    us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
    us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)
    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
    us_canada_user_rating_pivot = us_canada_user_rating.pivot(index='bookTitle', columns='userID',
                                                              values='bookRating').fillna(0)
    inf = {}
    for u in range(1, 746):
        inf[us_canada_user_rating_pivot.index[u]] = u
    us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(us_canada_user_rating_matrix)
    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[inf[query_index], :].values.reshape(1, -1),
                                              n_neighbors=6)

    results = []
    ss = 'Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[inf[query_index]])
    for i in range(1, len(distances.flatten())):
        result = '{0}: {1}, with distance of {2}:'.format(i,
                                                          us_canada_user_rating_pivot.index[indices.flatten()[i]],
                                                          distances.flatten()[i])
        results.append(result)
    context = {'hh': ss, 'results': results}
    return render(request, 'main.html', context)
