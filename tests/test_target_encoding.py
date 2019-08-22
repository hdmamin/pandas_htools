from sklearn.model_selection import train_test_split
import pandas_htools


def test_target_encode_null_targets(df_nulls):
    """Perform mean target encoding where the target includes some null values.
    """
    out = df_nulls.target_encode(['movieId'], 'rating', state=1)
    assert out.shape[1] == 4
    assert df_nulls.shape[1] == 3
    assert out.movieId__mean_enc.isnull().sum() == 0


def test_target_encode_null_both(df_nulls):
    """Perform mean target encoding where both the feature column and the
    target column contain some nulls.
    """
    out = df_nulls.target_encode(['userId'], 'rating')
    assert out.shape[1] == 4
    assert df_nulls.shape[1] == 3
    assert out.userId__mean_enc.isnull().sum() == 0


def test_target_encode_2_x_cols(df_store):
    """Perform median target encoding after grouping by 2 columns."""
    out = df_store.target_encode(['Segment', 'Category'], 'Profit',
                                 stat='median')
    assert out.loc[(out.Segment == 'Consumer') & (out.Category == 'Furniture'),
                   'Segment_Category__median_enc'].nunique() <= 5
    assert out.shape[0] == df_store.shape[0]
    assert out.shape[1] == df_store.shape[1] + 1
    assert out.Segment_Category__median_enc.isnull().sum() == 0


def test_target_encode_multi_x_cols(df_mta):
    """Perform variance target encoding after grouping by 3 columns."""
    out = df_mta.sample(n=5_000, replace=False)\
                .target_encode(['Plaza ID', 'Hour', 'Direction'],
                               '# Vehicles - Cash',
                               n=2,
                               stat='var')
    assert out.shape[0] == 5_000
    assert out.shape[1] == df_mta.shape[1] + 1
    assert out['Plaza ID_Hour_Direction__var_enc'].isnull().sum() == 0


def test_target_encode_validation(df_store):
    """Perform mean target encoding in place on both the training and
    validation sets.
    """
    train, val = train_test_split(df_store, train_size=.8)
    train.target_encode(['Sub-Category'], 'Discount', 3, inplace=True,
                        df_val=val)
    assert train.shape[1] == df_store.shape[1] + 1
    assert val.shape[1] == df_store.shape[1] + 1
    assert train['Sub-Category__mean_enc'].isnull().sum() == 0
    assert val['Sub-Category__mean_enc'].isnull().sum() == 0


def test_target_encode_inplace(df_nulls):
    """Perform skew target encoding in place when both the grouping column
    and the target column contain some nulls.
    """
    df_nulls.target_encode(['userId'], 'rating', 6, stat='skew', inplace=True)
    assert df_nulls.shape[1] == 4
    assert df_nulls.userId__skew_enc.isnull().sum() == 0
