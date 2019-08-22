import pandas_htools


def test_grouped_mode_int(df_mta):
    """Find the mode of an integer column after grouping by another integer
    column.
    """
    out = df_mta.grouped_mode('Plaza ID', '# Vehicles - Cash')
    for i in df_mta['Plaza ID'].unique():
        plaza_mode = df_mta.loc[df_mta['Plaza ID'] == i,
                                '# Vehicles - Cash'].mode().iloc[0]
        assert out.loc[i] == plaza_mode


def test_grouped_mode_float(df_store):
    """Find mode of a column of floats after grouping by a column of strings.
    """
    out = df_store.grouped_mode('Category', 'Profit')
    for cat in df_store.Category.unique():
        cat_mode = df_store.loc[df_store.Category == cat, 'Profit'].mode()[0]
        assert out.loc[cat] == cat_mode


def test_grouped_mode_multi_x(df_store):
    """Find mode after grouping by multiple columns."""
    out = df_store.grouped_mode(['Ship Mode', 'Region'], 'Sub-Category')
    assert len(out.index.levels) == 2
    assert out.index.names == ['Ship Mode', 'Region']
    assert all(out.values == ['Binders', 'Binders', 'Binders', 'Binders',
                              'Binders', 'Paper', 'Paper', 'Binders', 'Paper',
                              'Paper', 'Binders', 'Paper', 'Binders',
                              'Binders', 'Binders', 'Binders'])


def test_grouped_mode_nulls(df_nulls):
    """Find mode when nulls exist, either in the grouping column or the output
    column.
    """
    # Grouping column has nulls.
    out = df_nulls.grouped_mode('movieId', 'rating')
    assert out.shape[0] == 3

    # Target column has nulls.
    out = df_nulls.grouped_mode('rating', 'movieId')
    assert out.shape[0] == 5
