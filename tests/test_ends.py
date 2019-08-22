from pandas.core.frame import DataFrame
import pytest

import pandas_htools


@pytest.mark.parametrize('df',
                         ['df20', 'df_store'],
                         indirect=True)
class TestEnds:

    def test_ends_default_rows(self, df):
        """Test with default parameters."""
        output = df.ends()
        assert output.shape[0] == 6, f'ends() should return 6 rows.'
        assert isinstance(output, DataFrame), 'ends() should return data frame'

    def test_ends_negative_rows(self, df):
        """Try passing in negative number of rows."""
        with pytest.raises(AssertionError):
            df.ends(-1)

    def test_ends_more_rows(self, df):
        """Set the number of rows to be greater than the default amount."""
        output = df.ends(9)
        assert output.shape[0] == 18, 'ends() should return 18 rows'

    def test_ends_fewer_rows(self, df):
        """Set the number of rows to be less than the default amount."""
        output = df.ends(1)
        assert output.shape[0] == 2, 'ends(1) should return 2 rows'

    def test_ends_whole_df(self, df):
        """Set the number of rows such that 2*n is greater than the number of
        rows in the dataframe.
        """
        output = df.ends(200_000)
        assert output.equals(df), 'When n*2 is >= number of rows in'\
            'df, ends should return the entire df.'
