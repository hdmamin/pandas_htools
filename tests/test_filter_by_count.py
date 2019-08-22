import pandas_htools
import pytest


def test_filter_by_count_equals_one(df20):
    out = df20.filter_by_count('a', '=', 1)
    assert out.equals(df20), 'Column has no duplicate values and we filter ' \
        'by condition count==1, so return the whole df.'


def test_filter_by_count_greater_equals_400(df_store):
    out = df_store.filter_by_count('City', '>=', 400)
    assert out.City.nunique() == 5, \
        '5 cities occur at least 400 times in the dataframe.'


def test_less_than_3(df_store):
    out = df_store.filter_by_count('City', '<', 3)
    assert out.shape[0] == 196, 'Should have 70 rows of singly occurring ' \
        'cities and 126 rows for 63 twice-occurring cities.'


def test_filter_by_count_greater_than_normed(df_store):
    out = df_store.filter_by_count('Quantity', '>', .1, True)
    assert out.shape[0] == 7232, 'Filtered numeric column should have' \
        '7232 rows when considering categories that made up more than 10% ' \
        'of the df.'


def test_filter_by_count_zero_results(df_mta):
    out = df_mta.filter_by_count('# Vehicles - ETC (E-ZPass)', '>', .5, True)
    assert out.shape == (0, 6)
