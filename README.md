# Turbocharging Muon via AOL Preconditioning

![Figure 1: Using Turbo-Muon allows speedups on large and small matrices compared to naive Muon implementations and even to better
optimized versions such as Dion (denoted as `Muon+` here).](assets/banner_turbo_muon.png)

Orthogonality-Based Optimization via optimizers like Muon relies on iterative orthogonalisation procedures. 
These steps are designed to approximate the closest orthogonal matrix to the gradient in Frobenius norm before
the weight update. This mechanism is the main driver of performance on speed training tasks and also shows benefits
on large scale tasks. 

In this repository, we provide a codebase for accelerating the Muon optimizer via gradient preconditioning. 
Our work leverages the AOL parametrization method to ensure we reach better approximations of the orthogonalized gradients
in less iterative steps. Moreover, the main cost of the AOL parametrization method can be fused with the original iterative 
scheme of Newton-Schulz orthogonalisation. 

Here, we provide a standalone torch implementation of our Turbo-Muon variant along with custom triton kernels to double the 
speed of a naive implementation on medium to large matrices (c.f, Fig 1). Our standalone version on the Muon optimizer is provided
in the `turbo_muon.py` file, whereas the custom triton kernels are provided in the `newton_schulz_triton.py` file.

## Building upon existing Triton kernels

For this implementation, we started from the brilliant [implementation of Newton-Schulz](https://github.com/microsoft/dion) from the Dion paper,
which provides custom triton kernels for the NS algorithm.

### Triton Kernel for `ns_line_3`:

We noticed that the `ns_line_3` function was taking a lot of time, so we wrote a triton kernel to avoid multiple
loadings of the same data. This gave a marginal speedup on small matrices, where loading data is the bottleneck.

### Fewer iterations:

We remove the previous normalization to switch to AOL rescaling of Prach and Lampert.
Which is further explained in this [paper](https://arxiv.org/pdf/2208.03160).

This preconditioning leverages the existing computation of W@W^t in `ns_line_1` and then computes the
scaling factors via: `fast_inv_sqrt(reduce_sum(abs(WW^t), axis=-1))` which yields a vector of column rescaling which we
apply both to W and W@W^t. This introduces twofold benefits: first, this ensures that the NS iterations will now converge 
since the rescaled matrix now has singular values that are all inferior to one, secondly, this also makes the gradient closer
to orthogonality.

Thanks to this, we can save one iteration of Newton-Schulz approximation. However, the non linear nature of AOL prevents us from
using Jiacheng's approach to compute new polynomials factors. So we proposed two approaches to optimize our NS coeffs,
via polynomial approximation of AOL's effect or genetic algorithms. This is done in the directory `hp_opt`. We found both approaches 
to yield marginal gains and therefore use standard Muon coefficients which yield very good performance.

### Current work:

In our work, we provide several contributions:

- First, we show that AOL preconditioning makes NS converge to better estimates of the PolarFactor in usual
practical training regimes.
- Then, we also demonstrate the slight bias AOL preconditioning introduces recovers a steepest descent direction as per
the theoretical framework of [Old Optimizer, New Norm](https://arxiv.org/abs/2409.20325).
- Finally, we validate the better speed and convergence of our method by improving the speedrunning records that use the Muon
optimizer on both the `modded-nano-gpt` tasks and on `CIFAR-10` classification.

### Directories and Structure:

In the repository, we included standalone code implementations for both the torch optimizer that uses Muon with or without 
AOL preconditioning, along with custom triton kernels that allow for blazingly fast orthogonalization.

In the `paper_exps` directory, you will find our research code regarding:
- The better orthogonalization performance of AOL preconditioning (in `paper_exps/figures`).
- An empirical exploration of the slight bias introduced by preconditioning on real GPT like transformer gradients (in `paper_exps/gradient_exploration`).
- An exploration of polynomial and genetic NS coefficient optimization strategies (in `paper_exps/hp_opt`).
- The speedrunning improvements on both the CIFAR-10 dataset (`paper_exps/speedrun-cifar`) and the modded-nanogpt task (`paper_exps/speedrun-nanogpt`).

## Citation

Our paper will be made public soon. In the meantime, please consider citing this repository.

```
@misc{boissin2025turbocode,
  author       = {Thibaut Boissin and Thomas Massena},
  title        = {flash-newton-schulz: AOL rescaling and triton kernel for newton schulz},
  year         = {2025},
  url          = {https://github.com/thib-s/flash-newton-schulz}
}
```
