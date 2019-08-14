import pandas_htools
import pytest


@pytest.mark.parametrize('df',
                         ['df20', 'df_store'],
                         indirect=True)
class TestEnds:

    def test_ends_default_rows(self, df):
        assert df.ends().shape[0] == 6, f'ends() should return 6 rows.'

    def test_ends_negative_rows(self, df):
        with pytest.raises(AssertionError):
            df.ends(-1)

    def test_ends_more_rows(self, df):
        assert df.ends(9).shape[0] == 18, 'ends() should return 18 rows'

    def test_ends_fewer_rows(self, df):
        assert df.ends(1).shape[0] == 2, 'ends(1) should return 2 rows'

    def test_ends_whole_df(self, df):
        assert df.ends(200_000).equals(df), 'When n*2 is >= number of rows in'\
            'df, ends should return the entire df.'
