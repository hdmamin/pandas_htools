from pandas.core.frame import DataFrame
import pandas_htools
import pytest


@pytest.mark.parametrize('df',
                         ['df20', 'df_store'],
                         indirect=True)
class TestEnds:

    def test_ends_default_rows(self, df):
        output = df.ends()
        assert output.shape[0] == 6, f'ends() should return 6 rows.'
        assert isinstance(output, DataFrame), 'ends() should return data frame'

    def test_ends_negative_rows(self, df):
        with pytest.raises(AssertionError):
            df.ends(-1)

    def test_ends_more_rows(self, df):
        output = df.ends(9)
        assert output.shape[0] == 18, 'ends() should return 18 rows'

    def test_ends_fewer_rows(self, df):
        output = df.ends(1)
        assert output.shape[0] == 2, 'ends(1) should return 2 rows'

    def test_ends_whole_df(self, df):
        output = df.ends(200_000)
        assert output.equals(df), 'When n*2 is >= number of rows in'\
            'df, ends should return the entire df.'
