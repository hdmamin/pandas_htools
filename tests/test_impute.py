import pandas_htools


def test_impute_mean(df_nulls):
    """Impute missing ratings using the column mean."""
    avg = df_nulls.rating.mean()
    out = df_nulls.impute('rating')
    assert out.shape[1] == df_nulls.shape[1] + 1
    assert out.rating_isnull.sum() == 7
    assert out.rating.iloc[0] == avg


def test_impute_no_nulls(df20):
    """Try to impute when there are no null values."""
    ncols = df20.shape[1]
    out = df20.impute('a', method='mode', inplace=False, dummy=False)
    assert out.equals(df20)
    assert out.shape[1] == ncols


def test_impute_all_none(df_none):
    """Try to impute when every value in column is None (not np.nan).
    """
    out = df_none.impute('a')
    assert out.a_isnull.mean() == 1.0
    assert out.a.isnull().mean() == 1.0


def test_impute_multi(df_nulls):
    """Repeatedly impute nulls in multiple columns."""
    ncols = df_nulls.shape[1]
    for col in df_nulls.columns:
        df_nulls = df_nulls.impute(col, method='mean', inplace=False,
                                   dummy=True)
    assert df_nulls.shape[1] == ncols * 2
    assert df_nulls.isnull().sum().sum() == 0


def test_impute_inplace(df_nulls):
    """Fill null user ID's using median imputation."""
    df_new = df_nulls.impute('userId', method='median')
    df_nulls.impute('userId', method='median', inplace=True)
    assert df_new.equals(df_nulls)


def test_impute_not_inplace(df_nulls):
    """Perform mode imputation on rating column. Remember that userId has been
    imputed in place already due to the previous test.
    """
    out = df_nulls.impute('rating', method='mode', inplace=True)
    assert out is None
