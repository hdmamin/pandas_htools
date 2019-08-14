import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def df20():
    return pd.DataFrame(np.arange(100).reshape((20, 5)), columns=list('abcde'))


@pytest.fixture(scope='session')
def df_rep():
    return pd.concat((df20, df20.head(10), df.head(5)), axis=0)\
             .sample(frac=1, replace=False)


@pytest.fixture(scope='session')
def df_store():
    return pd.read_excel('~/data/superstore.xls')


@pytest.fixture
def df(request):
    return request.getfixturevalue(request.param)
