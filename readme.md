# InvariantRiskMinimization.jl
Implementation of the Invariant Risk Minimization [1] in Julia.
## About
The repository provides a basic implementation of the Invariant Risk Minimization (IRM). The code assumes binary classification with labels from the set {-1, 1} and mean square error loss.

The core function is `irm`, which initializes and trains a linear classifier using IRM. For further details refer to the Help mode in the Julia's REPL:
```julia
julia>?irm
irm(envdatasets, niter, λ, η; <keyword arguments>)

Train a linear classifier using the Invariant Risk Minimization [^1] method.
...
```

The repository also contains some unit tests in the `test.jl` file. Some of the tests check that code changes did not change the output of the algorithm for the specified input. If you change some implementation details, these could fail, but it does not necessarily mean something is wrong - it only means that the code will yield different results.

[1] ARJOVSKY, Martin, et al. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.