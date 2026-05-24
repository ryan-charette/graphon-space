from graphon_space.sampling import SamplingConfig, bin_feasible_envelope, sample_feasible_region


def test_sampling_records_have_required_columns():
    df = sample_feasible_region(SamplingConfig(model="triangle", kmax=2, samples=20, seed=7))

    assert {"e", "t", "t_triangle", "t_2star", "t_tilde", "entropy", "c", "P"}.issubset(df.columns)
    assert len(df) == 20


def test_bin_feasible_envelope():
    df = sample_feasible_region(SamplingConfig(model="two-star", kmax=2, samples=20, seed=9))
    envelope = bin_feasible_envelope(df, e_bins=5)

    assert {"e_mid", "min", "max", "count"}.issubset(envelope.columns)
    assert envelope["count"].sum() == len(df)
