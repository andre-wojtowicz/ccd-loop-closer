# Cyclic Coordinate Descent algorithm for protein loop closure

This program implements Cyclic Coordinate Descent algorithm on the basis of the article *Cyclic coordinate descent: A robotics algorithm for protein loop closure* by Canutescu AA and Dunbrack RL Jr. The implementation uses Python and Numpy.

The program has some internal tests:

```
$ python CCDLoopCloser.py

loop    eff. so far     result  rmsd    iterations
0       100.0%          SUCC    0.0901  10
1       100.0%          SUCC    0.0988  53
2       66.67%          MAX_IT  0.9603  250
3       75.0%           SUCC    0.0998  47
4       80.0%           SUCC    0.0937  36
5       83.33%          SUCC    0.0787  18
6       85.71%          SUCC    0.0999  132
7       87.5%           SUCC    0.0977  27
8       88.89%          SUCC    0.098   28
9       90.0%           SUCC    0.0957  7
10      90.91%          SUCC    0.095   5
(...)
```

You may read about the algorithm in my contest [article](http://dx.doi.org/10.13140/2.1.1898.6249).
