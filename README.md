# Turbocharging Newton-Schulz with AOL Rescaling and Triton Kernels

Our attempt to improve the speed of the newton schulz algorithm, starting from the dion implementation.

Disclaimer: this work is still in progress, especially we want to highlight that this approach change the 
underlying algorithm. So extra verification should be done before integrating it in optimizers like dion/muon.

## Changes

For this implementation we started from the [dion implementation of newton schulz](https://github.com/microsoft/dion)
which has a great triton implementation of the newton schulz algorithm.

### triton kernel for ns_line_3:

We noticed that the ns_line_3 function was taking a lot of time, so we wrote a triton kernel to avoid multiple
loadings of the same data. This give a marginal speedup on small matrices, where loading data is the bottleneck.

### Fewer iterations:

We remove the previous normalization to switch to AOL rescaling
Which is further explained in the paper: https://arxiv.org/pdf/2208.03160

This consists in computing W@W^t using ns_line_1 and then computing the
scaling factors: fast_inv_sqrt(reduce_sum(abs(WW^t), axis=-1)) which is a vector

Since the main operation to compute those correspond to ns_line_1,
we can fuse it with the first newton schulz iterate. Furthermore this gives a better
starting point for the newton schulz iterations as the matrix is closer to orthogonal

Thanks to this, we can save one iteration of newton schulz. However, the non linear nature of AOL prevents us from
using Jiacheng's approach to compute new polynomials factors. So we proposed two approaches to optimize our NS coeffs,
via polynomial approximation of AOL's effect or genetic algorithms. This is done in the directory `hp_opt`.

### Current work:

First, we need to make sure our implementation is not changing the nature of the underlying iterative computation.
Indeed, AOL rescaling has an effect of the polar factor of the matrix $U.V^T$ we are trying to compute. However,
when matrices are relatively close to being column-orthogonal (which is mostly the case in high dimensions), 
meaning this effect is small compared to the approximation error of the baseline NS algorithm.

This is checked empirically in a GPT-2 like training setup in the `gradient-exploration` directory.

### Empirical validation:

In order to validate the suitability of this approach for orthogonality based optimizers, we run benchmarks 
on both the nanogpt and cifar speedrun setups, as implemented in `speedrun-nanogpt` and `speedrun-cifar` directories
respectively.

## Current results:

Using a L40S GPU, we obtain a decent speedup:

![speedup graph](assets/speedup_evaluation.png)

When tested on random uniform matrices, the matrices seems closer to orthogonal:

![orthogonality graph](assets/svs_2048x2048.png)

On the cifar speedrun setup, we obtain similar final accuracies:

| Model / Run | Mean Accuracy | Std Dev | Training Time Mean (s) | Iterations |
| --- | --- | --- | --- | --- |
| NS | 0.9401 | 0.0009 | 2.66 | 20 |
| AOL + NS | 0.9401 | 0.0016 | 2.64 | 20 |

This minor speedup is expected as the model is small, however, it does validate the equal capability of our approach to optimize
the model. Also, we did not modify any hyperparameter from the baseline NS run to replicate this result.

## Citation

```
@misc{lin2025flash,
  author       = {Thibaut Boissin and Thomas Massena},
  title        = {flash-newton-schulz: AOL rescaling and triton kernel for newton schulz},
  year         = {2025},
  url          = {https://github.com/thib-s/flash-newton-schulz}
}
```
