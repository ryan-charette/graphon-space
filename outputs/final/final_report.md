# Final Graphon-Space Numerical Run

## Data Products

- Triangle samples: 60,000 rows, k=1..6.
- Edge-2-star samples: 60,000 rows, k=1..6.
- Triangle boundary records: 62.
- Edge-2-star boundary records: 62.
- Triangle optimize-grid records: 27, successful family records: 2.
- Edge-2-star optimize-grid records: 75, successful family records: 3.
- Edge-2-star stability records: 39, successful family records: 21.

## Validation Metrics

- Triangle max-boundary median absolute error vs e^(3/2): 4.396e-12.
- Triangle max-boundary worst absolute error vs e^(3/2): 3.626e-10.
- Edge-2-star max-boundary median absolute error vs known envelope: 7.841e-14.
- Edge-2-star max-boundary worst absolute error vs known envelope: 2.785e-10.

## Target Comparisons

- Triangle target e=0.35, t=0.02 winner: bipodal (S=0.450187, class=symmetric-bipodal).
- Edge-2-star target e=0.5, t=0.29 winner: bipodal (S=0.509197, class=nonsymmetric-bipodal).

## Notes

- The optimize-grid runs are intentionally coarse first-pass maps; dense phase maps require much longer SLSQP runs.
- The figures overlay known ER and upper-boundary curves where available.