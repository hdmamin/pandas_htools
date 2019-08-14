import pandas_htools
import pytest


def test_count_equals_one(df20):
    assert df20.filter_by_count('a', '=', 1).equals(df20), 'Column has no ' \
        'duplicate values and we filter by condition count==1, so return the' \
        'whole df.'


def test_ge_400(df_store):
    assert df_store.filter_by_count('City', '>=', 400).City.nunique() == 5, \
        '5 cities occur at least 400 times in the dataframe.'


def test_less_than_3(df_store):
    assert df_store.filter_by_count('City', '<', 3).City.shape[0] == 196, \
        'Should have 70 rows of singly occurring cities and 126 rows for 63' \
        'twice-occurring cities.'


def test_greater_than_normed(df_store):
    subset = df_store.filter_by_count('Quantity', '>', .1, True)
    assert subset.shape[0] == 7232, 'Filtered numeric column should have' \
        '7232 rows when considering categories that made up more than 10% ' \
        'of the df.'


# Maybe can convert to one class using parametrize()?
# class TestFilterByCount:
#
#     def test_count_equals_one(self, df20):
#         assert df20.filter_by_count('a', '=', 1).equals(df20), 'Column has no ' \
#             'duplicate values and we filter by condition count==1, so return the' \
#             'whole df.'
#
#     def test_ge_400(self, df_store):
#         assert df_store.filter_by_count('City', '>=', 400).City.nunique() == 5, \
#             '5 cities occur at least 400 times in the dataframe.'
#
#     def test_less_than_3(self, df_store):
#         assert df_store.filter_by_count('City', '<', 3).City.shape[0] == 196, \
#             'Should have 70 rows of singly occurring cities and 126 rows for 63' \
#             'twice-occurring cities.'
#
#     def test_greater_than_normed(self, df_store):
#         subset = df_store.filter_by_count('Quantity', '>', .1, True)
#         assert subset.shape[0] == 7232, 'Filtered numeric column should have' \
#             '7232 rows when considering categories that made up more than 10% ' \
#             'of the df.'