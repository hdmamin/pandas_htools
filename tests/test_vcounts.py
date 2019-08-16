import pandas_htools


def test_vcounts_integer(df_rep):
    """Compute vcounts on a column of integers."""
    out = df_rep.a.vcounts()
    assert all(df_rep.a.value_counts().values == out.a_raw_count.values),\
        'Raw count column should match un-normalized value_counts().'
    assert all(df_rep.a.value_counts(normalize=True).values ==
               out.a_normed_count.values), \
        'Normed count column should match normalized value_counts().'


def test_vcounts_str(df_mta):
    """Compute vcounts on a column of strings."""
    out = df_mta.Direction.vcounts()
    for dir_ in ('I', 'O'):
        assert df_mta.Direction.value_counts()[dir_] == \
            out.loc[out.Direction == dir_, 'Direction_raw_count'].values[0]
        assert df_mta.Direction.value_counts(normalize=True)[dir_] ==\
            out.loc[out.Direction == dir_, 'Direction_normed_count'].values[0]


def test_vcounts_sort(df_store):
    """Compute vcounts while passing in a sort parameter."""
    categories = ['Office Supplies', 'Furniture', 'Technology']
    out = df_store.Category.vcounts(sort=True, ascending=False)
    assert all(out.Category.values == categories)
    out = df_store.Category.vcounts(sort=True, ascending=True)
    assert all(out.Category.values == categories[::-1])


def test_vcounts_normalize(df_rep):
    """Compute vcounts while passing in a normalize parameter (should be
    ignored).
    """
    assert df_rep.c.vcounts().equals(df_rep.c.vcounts(normalize=True))


def test_vcounts_all_none(df_none):
    """Confirm zero rows returned when the column contains nothing but None."""
    out = df_none.a.vcounts()
    assert out.shape == (0, 3)
    out = df_none.a.vcounts(dropna=False)
    assert out.shape == (1, 3)
