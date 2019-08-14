import pandas_htools


def test_vcounts_integer(df_rep):
    output = df_rep.a.vcounts()
    assert output.a.equals(df_rep.a.value_counts()), 'Un-normalized counts should be'
    assert output.a_normed.equals(df_rep.a.value_counts(normalize=True))
