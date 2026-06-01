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

# Root finding algorithms experiments

# Sparse matrix solver

# PageRank experiments

# SLE algorithms testing suite

# LinReg with Gradient Descent
