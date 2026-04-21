# Foldy Lax Model
## Background

```@meta
CurrentModule = FoldyLax
```

To test the dictionary learning algorithms, `DictionaryLearning.jl` provides utilities for 
generating Green's matrices for the Foldy-Lax model of scattered wave-fields. In this 
section we give an overview of this model and what is implemented in this package.

We regard an emmiter (a source or scatterer) as a pair ``(x, v) \in \mathbb{R}^3 \times \mathbb{C}^p`` where ``x`` is the location of the emitter and ``v`` the signals discrete Fourier transform at prescribed frequencies ``f_1, \cdots, f_p``. 

For some ``a,b \in \mathbb{R}^3`` define ``f(a,b) : \mathbb{C}^p \to \mathbb{C}^p`` to be the linear transformation such that 

```math
f(a,b)e_s = \frac{\exp\left( \frac{2\pi f_s}{c_0} i\left\lVert a - b\right\rVert_2\right)}{4\pi \left\lVert a - b\right\rVert_2}e_s
```

This is the *free-space propogator*. The signal recieved by a reciever at ``a`` from the emmiter ``(b_i, \hat b_i)`` is given by ``f(a, b)\hat b_i``.

Let ``r_1, \cdots, r_n \in \mathbb{R}^3`` be the coordinates of our recievers. Let ``\xi_1, \cdots, \xi_m\in \mathbb{R}^3`` be the scatterers, ``\hat\xi_1, \cdots, \hat\xi_m \in \mathbb{C}^p`` their signal, and ``0 \leq \tau_1, \cdots, \tau_m \leq 1`` be their scattering amplitudes. Let ``z_1, \cdots, z_k \in \mathbb{R}^3`` be the coordinates of the sources, and ``\hat z_1, \cdots \hat z_k \in \mathbb{C}^p`` their signal.

```math
\hat{r}_i = \sum_{j=1}^k f(r_i, z_j)\hat z_j + \sum_{j=1}^m f(r_i, \xi_j)\tau_j \hat \xi_j \qquad i = 1, \cdots, n \tag{1}
```

```math
\hat{\xi}_i = \sum_{j=1}^k f(\xi_i, z_j)\hat z_j + \sum_{\substack{j=1\\ j \neq i}}^m f(\xi_i, \xi_j) \tau_j \hat \xi_j \qquad i = 1, \cdots, m \tag{2}
```

Let ``\hat r = \hat{r}_1 \oplus \cdots \oplus \hat{r}_n``, ``\hat z = \hat{z}_1 \oplus \cdots \oplus \hat{z}_k``, and ``\hat \xi = \hat \xi_1 \oplus \cdots \oplus \hat \xi_m``. We would like to solve for the order-3 "Green's tensor" 
```math
\mathcal{G} : \underbrace{\mathbb{C}^p \oplus \cdots \oplus \mathbb{C}^p}_{\text{$k$ times}} \to \underbrace{\mathbb{C}^p \oplus \cdots \oplus \mathbb{C}^p}_{\text{$n$ times}}
``` 

such that ``\mathcal{G}(\hat z) = \hat r``. Forming ``\mathcal{G}`` efficiently is the goal of the `DictionaryLearning.FoldyLax`. Since ``\mathcal{G}(e^T_s\hat z) = e^T_s\hat r`` it suffices to construct the linear transformations ``G_s = \mathcal{G}_{:,:,s}`` independently.

Define ``M^{[\xi z]}_s \in M_\mathbb{C}(m, k),`` ``M^{[rz]}_s \in M_\mathbb{C}(n,k)``, ``M^{[\xi \xi]}_s \in M_{\mathbb{C}}(m, m),`` and ``M^{[r \xi]}_s \in M_\mathbb{C}(n, m)`` so that

```math
\begin{align*}
e^T_i M^{[\xi z]}_s e_j &= e_s^T f(\xi_i, z_j)\\
e^T_i M^{[r z]}_s e_j &= e_s^T f(r_i, z_j)\\
e^T_i M^{[\xi \xi]}_s e_j &= \begin{cases} e_s^T f(\xi_i, \xi_j)\tau_j & i \neq j\\ -1 & i = j \end{cases}\\
e^T_i M^{[r \xi]}_s e_j &= e_s^T f(r_i, \xi_j) \tau_j
\end{align*}
```

Solving (2) gives the excited waves in terms of the source waves: 
```math
\hat \xi_s = - \left(M^{[\xi \xi]}_s\right)^{-1} M^{[\xi z]}_s \hat z_s. \tag{3}
```
Similarly, we may rewrite (1) as 
```math
\hat r_s = M^{[rz]}_s \hat z_s + M^{[r \xi]}_s  \hat\xi_s.
``` 
Substituting into (3) lends 
```math
\hat r_s = \left[M^{[rz]}_s - M^{[r \xi]}_s \left(M^{[\xi \xi]}_s\right)^{-1} M^{[\xi z]}_s\right]\hat z_s. \tag{4}
``` 

The quantity inside the brackets are the components of the tensor ``\mathcal{G}`` we seek to compute.

`FoldyLax.jl` provides two functions for computing ``\mathcal{G}``. The first is `compM!` which is used to compute the matrices ``M^{[\cdot]}_s`` for ``s = 1, \cdots, p`` all at once. The second is `compG!` which computes the tensor ``\mathcal{G}``. In order to speed up the computation when the sources and recievers are changing over time, `compG!` expects the ``M^{[\xi\xi]}_s``-s to be factorized in order to speed up the least-squares solve from (3).

Please see the documentation for those two function as well as `\examples\foldy-lax-example.ipynb`. All functions support both CPUs and CUDA, but are optimized for the latter.
The following are functions provided by the `FoldyLax` submodule.

## Functions

See the example in `/examples/foldy-lax/foldy-lax-example.ipynb` for how to use
the following functions.

```@docs
compM!
```

```@docs
compG!
```