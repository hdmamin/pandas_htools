import pandas_htools


def test_target_encode_null_targets(df_nulls):
    out = df_nulls.target_encode('movie_Id', 'rating', state=1)
    assert out.shape[1] == 4
    assert df_nulls.shape[1] == 3
    assert out.rating_enc.isnull().sum() == 0


def test_target_encode_null_both(df_nulls):
    out = df_nulls.target_encode('user_Id', 'rating')
    assert True


def test_target_encode_multi_x(df_store):
    assert True


def test_target_encode_std():
    assert True


def test_target_encode_inplace(df_nulls):
    assert True
