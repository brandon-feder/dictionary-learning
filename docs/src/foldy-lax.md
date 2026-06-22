# Foldy Lax Model
## Background

To test the dictionary learning algorithms, `DictionaryLearning.jl` provides utilities for 
generating Green's matrices for the Foldy-Lax model of scattered wave-fields. In this 
section we give an overview of this model and what is implemented in this package.

For some ``a,b \in \mathbb{R}^3`` define ``f(a,b) : \mathbb{C}^p \to \mathbb{C}^p`` to be the linear transformation such that 

```math
f(a,b)e_s = \frac{\exp\left( \frac{2\pi f_s}{c_0} i\left\lVert a - b\right\rVert_2\right)}{4\pi \left\lVert a - b\right\rVert_2}e_s
```

This is the *free-space propogator*. ``f(a, b_i)`` takes a signal emitted by a source at ``b`` to the signal recieved by a reciever at ``a``.

Let ``r_1, \cdots, r_n \in \mathbb{R}^3`` be the coordinates of our recievers. Let ``\xi_1, \cdots, \xi_m\in \mathbb{R}^3`` be the scatterers, ``\hat\xi_1, \cdots, \hat\xi_m \in \mathbb{C}^p`` their signal, and ``0 \leq \tau_1, \cdots, \tau_m \leq 1`` be their scattering amplitudes. Let ``z_1, \cdots, z_k \in \mathbb{R}^3`` be the coordinates of the sources, and ``\hat z_1, \cdots \hat z_k \in \mathbb{C}^p`` their signal.

```math
\hat{r}_i = \sum_{j=1}^k f(r_i, z_j)\hat z_j + \sum_{j=1}^m f(r_i, \xi_j)\tau_j \hat \xi_j \qquad i = 1, \cdots, n \tag{1}
```

```math
\hat{\xi}_i = \sum_{j=1}^k f(\xi_i, z_j)\hat z_j + \sum_{\substack{j=1\\ j \neq i}}^m f(\xi_i, \xi_j) \tau_j \hat \xi_j \qquad i = 1, \cdots, m \tag{2}
```

Given $\hat z_i$ as well as coordinates $r_i$, $z_i$, and $\xi_i$ in general positions, equations (1) and (2) constitude a determined system of linear equations.

For $s = 1, \cdots, p$ define
``\mathcal{G}_s : \mathbb{C}^{k} \to \mathbb{C}^{m}`` 
to be the linear transformation such that 
```math
\mathcal{G}_s \begin{bmatrix}
    e_s^T \hat z_1\\
    \vdots\\
    e_s^T \hat z_k
\end{bmatrix} = \begin{bmatrix}
    e_s^T \hat r_1\\
    \vdots\\
    e_s^T \hat r_m
\end{bmatrix}.
```

Then, define ``\mathcal{G} = \mathcal{G} = \begin{bmatrix}\mathcal{G}_1^T, \cdots, \mathcal{G}_s^T\end{bmatrix}^T``. This is the dictionary we would like to learn. Forming ``\mathcal{G}`` efficiently is the goal of the of the functions below. It suffices to construct the ``\mathcal{G}_s`` independently and in parallel. With this in mind, define 
```math
\hat z^s = \begin{bmatrix}\hat z_1, \cdots, \hat z_k\end{bmatrix}^T e_s\qquad\text{and}\qquad \hat r^s = \begin{bmatrix}\hat r_1, \cdots, \hat r_m\end{bmatrix}^T e_s
```

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
\hat \xi^s = - \left(M^{[\xi \xi]}_s\right)^{-1} M^{[\xi z]}_s \hat z^s. \tag{3}
```
Similarly, we may rewrite (1) as 
```math
\hat r^s = M^{[rz]}_s \hat z_s + M^{[r \xi]}_s  \hat\xi^s.
``` 
Substituting into (3) lends 
```math
\hat r^s = \left[M^{[rz]}_s - M^{[r \xi]}_s \left(M^{[\xi \xi]}_s\right)^{-1} M^{[\xi z]}_s\right]\hat z^s. \tag{4}
``` 

The quantity inside the brackets is $\mathcal{G}_s$.

## Functions

See the example in `/examples/foldy-lax/foldy-lax-example.ipynb` for how to use the following.

```@docs
FoldyLaxStruct
```

```@docs
FoldyLaxWorkStruct
```

```@docs
foldylax!
```