from functools import partial
import operator
import pandas as pd
import pandas_flavor as pf
from sklearn.model_selection import KFold


@pf.register_dataframe_method
def ends(df, n=3):
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
    assert n > 0, 'n must be positive.'

    if df.shape[0] < 2 * n:
        return df
    return pd.concat((df.head(n), df.tail(n)), axis=0)


@pf.register_dataframe_method
def filter_by_count(df, col, method, value, norm=False):
    """Filter a dataframe to return a subset of rows determined by their
    value_counts(). For example, we can return rows with users who appear
    at least 5 times in the dataframe, or with users who appear less than 10
    times, or who appear exactly once.

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
    df.filter_by_count('user_id', '>=', 5)

    Return rows containing users who appear only once:
    df.filter_by_count('user_id', '=', 1)

    Return rows containing users who make up less than 20% of rows:
    df.filter_by_count('user_id', '<', .2, True)
    """
    operation = {'=': operator.eq,
                 '>': operator.gt,
                 '<': operator.lt,
                 '>=': operator.ge,
                 '<=': operator.le
                }
    counts = df[col].value_counts(norm).loc[lambda x:
                                            operation[method](x, value)]
    return df[df[col].isin(counts.index)]


@pf.register_dataframe_method
def grouped_mode(df, xs, y):
    """Return the most common value in column y for each value or combination
    of values of xs. Note that this can be slow, especially when passing in
    multiple x columns.

    Parameters
    -----------
    xs: list[str]
        One or more column names to group by.
    y: str
        Column to calculate the modes from.

    Returns
    --------
    pd.Series
    """
    return df.groupby(xs)[y].agg(lambda x: pd.Series.mode(x)[0])


@pf.register_dataframe_method
def impute(df, col, method='mean', inplace=False, dummy=True):
    """Fill null values in the specified column, then optionally add an
    additional column specifying whether the first column was originally null.
    This can be useful in certain machine learning problems if the fact that a
    value is missing may indicate something about the example.

    For instance, we might try to predict student test scores, where one
    feature column records the survey results of asking the student's parent to
    rate their satisfaction with the teacher on a scale of 1-5. If the value is
    missing, that means the parent didn't take the survey, and therefore may
    not be very involved with the student's academics. This could be highly
    relevant information that we don't want to discard, which we would if we
    simply imputed the missing value and made no record of it.

    Parameters
    -----------
    col: str
        Name of df column to fill null values for.
    method: str
        One of ('mean', 'median', 'mode'). More complex methods, such as
        building a model to predict the missing values based on other features,
        must be done manually.
    inplace: bool
        Specify whether to perform the operation in place (default False).
    dummy: bool
        Specify whether to add a dummy column recording whether the value was
        initially null (default True).

    Returns
    --------
    pd.DataFrame
    """
    if not inplace:
        df = df.copy()

    # If adding a dummy column, it must be created before imputing null values.
    if dummy:
        df[col + '_isnull'] = df[col].isnull() * 1
    fill_val = getattr(df[col], method)()
    df[col].fillna(fill_val, inplace=True)

    if not inplace:
        return df


@pf.register_dataframe_method
def target_encode(df, x, y, n, mode='mean', shuffle=True, state=None,
                  inplace=False, df_val=None):
    """Compute target encoding based on one or more feature columns.

    Parameters
    -----------
    x: str, list[str]
        Name of columns to group by.
    y: str
        Name of target variable column.
    n: int
        Number of folds for regularized version. Must be >1.
    mode: str
        Specifies the type of aggregation to use on the target column.
        Typically this would be mean or occasionally median, but all the
        standard dataframe aggregation functions are available:
        ('mean', 'median', 'min', 'max', 'std', 'var', 'skew').
    shuffle: bool
        Specifies whether to shuffle the dataframe when creating folds of the
        data. This would be important, for instance, if the dataframe is
        ordered by a user_id, where each user has multiple rows. Here, a lack
        of shuffling means that all of a user's rows are likely to end up in
        the same fold. This effectively eliminates the value of creating the
        folds in the first place.
    state: None, int
        If state is an integer and shuffle is True, the folds produced by
        KFold will be repeatable. If state is None (the default) and shuffle
        is True, shuffling will be different every time.
    inplace: bool
        Specifies whether to do the operation in place. The inplace version
        does not return anything. When inplace==False, the dataframe is
        returned.
    df_val: None, pd.DataFrame
        Validation set (optional). If provided, naive (i.e. un-regularized)
        target encoding will be performed using the labels from the original
        (i.e. training) df. NOTE: Inplace must be True when passing in df_val,
        because we only return the original df.

    Returns
    --------
    pd.DataFrame or None
    """
    assert df_val is None or inplace, 'To encode df_val, inplace must be True.'

    if not inplace:
        df = df.copy()
    new_col = f'{y}_enc'
    df[new_col] = global_agg = getattr(df[y], mode)()

    # Compute target encoding on n-1 folds and map back to nth fold.
    for train_idx, val_idx in KFold(n, shuffle, state).split(df):
        enc = getattr(df.iloc[train_idx, :].groupby(x)[y], mode)()
        df.loc[:, new_col].iloc[val_idx] = df.loc[:, x].iloc[val_idx].map(enc)

    # Encode validation set in place if it is passed in. No folds are used.
    if df_val is not None:
        enc = getattr(df.groupby(x)[y], mode)()
        df_val[new_col] = df_val[x].map(enc).fillna(global_agg)

    if not inplace:
        return df


@pf.register_dataframe_method
def top_categories(df, col, categories=None, threshold=None):
    """Filter a dataframe to return rows containing the most common categories.
    This can be useful when a column has many possible values, some of which
    are extremely rare, and we want to consider only the ones that occur
    relatively frequently.

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

    Returns
    --------
    pd.DataFrame
    """
    if categories is not None:
        top = df[col].value_counts(ascending=False).head(categories).index
        return df[df[col].isin(top)]
    if threshold is not None:
        return df.groupby(col).filter(lambda x: len(x) >= threshold)


@pf.register_series_method
def vcounts(df_col, **kwargs):
    """Return both the raw and normalized value_counts of a series.

    Parameters
    -----------
    Most parameters in value_counts() are available (i.e. `sort`, `ascending`,
    `dropna`), with the obvious exception of `normalize` since that is handled
    automatically.

    Returns
    --------
    pd.DataFrame

    Examples
    ---------
    df.colname.vcounts()
    """
    if 'normalize' in kwargs.keys():
        del kwargs['normalize']
    df = pd.merge(df_col.value_counts(**kwargs),
                  df_col.value_counts(normalize=True, **kwargs),
                  'left', left_index=True, right_index=True,
                  suffixes=['_raw_count', '_normed_count'])\
           .reset_index()
    return df.rename({'index': df.columns[1].split('_')[0]}, axis=1)
