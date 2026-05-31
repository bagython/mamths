`uv run <suite>.py`

# compare bisection and fixed point convergence

[main.py](main.py)

```
fixed point ~ bisection: interval=(1, 4) x0=2.5 alpha=-0.01850
  bisection  : root=2.9999999999 err=1.16e-10 iters=32
  fixed point: root=2.9999999993 err=6.75e-10 iters=30
fixed point much slower: interval=(0, 5) x0=4.0 alpha=-0.00500
  bisection  : root=3.0000000002 err=1.75e-10 iters=33
  fixed point: root=3.0000000061 err=6.12e-09 iters=128
fixed point much faster: interval=(1, 6) x0=4.5 alpha=-0.03704
  bisection  : root=2.9999999998 err=1.75e-10 iters=33
  fixed point: root=3.0000000000 err=0.00e+00 iters=7
```

# Jacoby naive vs vectorized

[jacobi.py](jacobi.py)

```
3x3 sanity check:
  naive  (14 iter): [1.9999998732422768, -1.0000001589738368, 5.999999695774749]
  matrix (14 iter): [1.9999998732422768, -1.0000001589738368, 5.999999695774749]

100x100 benchmark, eps=1e-06:
  naive  : 561 iterations in 0.2068s
  matrix : 561 iterations in 0.0024s
  speedup: 85.9x
  max |x_naive - x_matrix|: 0.00e+00
```

# Root finding algorithms experiments

# Sparse matrix solver

# PageRank experiments

# SLE algorithms testing suite

# LinReg with Gradient Descent
