import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def df20():
    return pd.DataFrame(np.arange(100).reshape((20, 5)), columns=list('abcde'))


@pytest.fixture(scope='module')
def df_mta():
    """Use module scope because some target_encode() tests edit in place."""
    return pd.read_csv('~/data/MTA_Hourly.csv')


@pytest.fixture(scope='module')
def df_nulls():
    """Use module scope because some impute() tests edit in place."""
    df = pd.read_csv('~/msan_mod3a/msan630/hw1/tiny_training2.csv')
    df.loc[df.index % 2 == 0, 'rating'] = np.nan
    df.loc[df.userId % 2 == 1, 'userId'] = np.nan
    return df


@pytest.fixture(scope='session')
def df_none():
    return pd.DataFrame([{'a': None} for i in range(10)])


@pytest.fixture(scope='session')
def df_rep(df20):
    return pd.concat((df20, df20.head(10), df20.head(5)), axis=0)\
             .sample(frac=1, replace=False)


@pytest.fixture(scope='session')
def df_store():
    return pd.read_excel('~/data/superstore.xls')


@pytest.fixture
def df(request):
    return request.getfixturevalue(request.param)
