import pytest

import pandas_htools


def test_top_categories_ncats(df_store):
    out = df_store.top_categories('State', n_categories=5)
    assert out.shape == (5207, 21)
    assert out.State.nunique() == 5
    assert (set(df_store.State.value_counts().head(5).index.values) ==
            set(out.State.unique()))


def test_top_categories_threshold(df_store):
    out = df_store.top_categories('Quantity', threshold=1_000)
    assert out.Quantity.unique().tolist() == [2, 3, 5, 4]
    assert out.shape[0] == 7232
    assert df_store.shape[0] == 9994


def test_top_categories_nulls(df_nulls):
    out = df_nulls.top_categories('userId', n_categories=2)
    assert out.userId.unique().tolist() == [4, 52]


def test_top_categories_both_args(df_store):
    with pytest.raises(AssertionError):
        out = df_store.top_categories('Ship Mode', n_categories=3, threshold=6)


def test_top_categories_both_args(df_store):
    with pytest.raises(AssertionError):
        out = df_store.top_categories('Ship Mode')
