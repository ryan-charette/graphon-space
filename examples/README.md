# Historical screening comparison

`compare_screening_scores.py` preserves the distinct scoring formula from
`Graphons Fall 2023/graphons_take2.py`. The current package uses a denominator
of `A^2 + B^2`; that local prototype used `(A^3 - B^3)^(2/3)` with the same
entropy numerator. These normalizations can give different decisions when
compared with `H''(e)`.

From the repository root, after installing the package dependencies:

```sh
python -m examples.compare_screening_scores --edge 0.3 --a 0.1 --b 0.01
```

This example reuses the package's validated score and compares the two
normalizations. It does not establish which screening criterion is
mathematically justified, prove an entropy advantage, or alter the package's
search behavior. The prototype's duplicated optimizer, incorrectly weighted
entropy calculation, and plots are omitted. The other 2023 tripodal scripts
are already substantially represented by `graphon_space.constructive`.
