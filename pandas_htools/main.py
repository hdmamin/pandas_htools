import operator
import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def ends(df, n=5):
    """Display the first and last few rows of a dataframe.

    Parameters
    -----------
    n: int
        Number of rows to return in the head and tail, respectively. The total
        number of rows returned will be equal to 2*n.

    Returns
    --------
    pd.DataFrame
    """
    if n < 0:
        raise ValueError('n must be positive.')

    if df.shape[0] < 2 * n:
        return df
    return pd.concat((df.head(n), df.tail(n)), axis=0)


@pf.register_dataframe_method
def filter_by_count(df, col, method, value):
    """Filter a dataframe to return a subset of rows determined by their
    value_counts(). For example, we can return rows with users who appear
    at least 5 times in the dataframe, or with users who appear less than 10
    times, or who appear greater than or equal to 100 times.

    Parameters
    -----------
    col: str
        Name of dataframe column to filter by.
    method: str
        Symbol specifying which operation to use for filtering.
        One of ('=', '<', '>', '<=', '>=').
    value: int, float
        Numeric value that each row in `col` will be compared against.

    Returns
    --------
    pd.DataFrame

    Examples
    ---------
    Return rows containing users who appear at least 5 times:
    df.filter_by_count('user_id', 5, '>=')

    Return rows containing users who appear only once:
    df.filter_by_count('user_id', 1, '=')
    """
    operation = {'=': operator.eq,
                 '>': operator.gt,
                 '<': operator.lt,
                 '>=': operator.ge,
                 '<=': operator.le
                }
    counts = df[col].value_counts().loc[lambda x: operation[method](x, value)]
    return df[df[col].isin(counts.index)]


@pf.register_dataframe_method
def top_cats(df, col, categories=None, threshold=None):
    """Filter a dataframe to return rows containing the most common categories.
    The user can either specify the number of categories to include or set
    a threshold for the minimum number of occurrences. One of `categories` and
    `threshold` should be None, while the other should be an integer.

    Parameters
    -----------
    col: str
        Name of column to filter on.
    categories: int, None
        Optional - # of categories to include (i.e. top 5 most common
        categories).
    threshold: int, None
        Optional - Value count threshold to include (i.e. all categories that
        occur at least 10 times).
    """
    if categories is not None:
        top = df[col].value_counts(ascending=False).head(categories).index
        return df[df[col].isin(top)]
    if threshold is not None:
        return df.groupby(col).filter(lambda x: len(x) >= threshold)


@pf.register_series_method
def vcounts(df_col):
    """Return both the raw and normalized value_counts of a series.

    Examples
    ---------
    df.colname.vcounts()
    """
    return pd.merge(df_col.value_counts(), df_col.value_counts(normalize=True),
                    'left', left_index=True, right_index=True,
                    suffixes=['', '_normed'])
