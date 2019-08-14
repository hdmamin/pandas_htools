import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def df20():
    return pd.DataFrame(np.arange(100).reshape((20, 5)), columns=list('abcde'))


@pytest.fixture(scope='session')
def df_store():
    return pd.read_excel('~/data/superstore.xls')


# @pytest.fixture(scope='session', params=[(df20, 0), (df_store, 0)])
# def df_sizes(request):
#     return request.param


@pytest.fixture
def df(request):
    return request.getfixturevalue(request.param)
