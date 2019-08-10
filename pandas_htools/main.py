import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def ends(df, n=5):
    """Display the first and last few rows of a dataframe.

    Parameters
    -----------
    df: pd.DataFrame
    n: int
        Number of rows to return in the head and tail, respectively. The total
        number of rows returned will be equal to 2*n.

    Returns
    --------
    pd.DataFrame
    """
    return pd.concat((df.head(n), df.tail(n)), axis=0)
