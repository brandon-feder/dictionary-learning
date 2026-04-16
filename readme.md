### Suspace Recovery

The subspace recovery algorithm consists of four steps:
1. Computing the correlation covariance matrix $\Sigma$
2. Computing the $r$ weighted correlation covariance matrices $\Sigma_j$
3. Computing the $r$ projections $\Sigma_j^\text{proj} = \Sigma_j - \text{proj}_\Sigma(\Sigma_j)$
4. Recovering the $r$ subspaces by computing the top `s` eigenvalues of each $\Sigma_j^\text{proj}$

We explain how these are each computed in `SubspaceRecovery.jl`.


<!-- ### Computing Correlation Covariance
The most expensive step of doing subspace recovery seems like 
computing
$$\hat{\Sigma}_j = \frac{1}{N}\sum_{i=1}^N \langle y_i, y_j\rangle y_i y_i^T.$$
We describe how to compute all these matrices in $\mathcal{O}(Nd^4)$ time.

Note that 
$$\langle y_i, y_j\rangle = y_i^Ty_jy_j^Ty_i = \text{tr}(y_i^Ty_jy_j^Ty_i).$$
From the cyclic property of trace,
$$=\text{tr}(y_i^Ty_iy_j^Ty_j) = \langle y_i^Ty_i, y_j^Ty_j\rangle _F$$
From properties of the Frobenius inner-product,
$$=\text{vec}(y_i^Ty_i)^T\text{vec}(y_j^Ty_j)$$
Let $v_i = \text{vec}(y_i^Ty_i)$. Then we seek to compute 
$$\hat{\Sigma}_j = \frac{1}{N}\sum_{i=1}^N \left(v_j^Tv_i\right)v_i = \frac{1}{N}\sum_{i=1}^N \left(v_i^Tv_j\right)v_i$$
Since $(v_i^Tv_j)$ is a scalar,
$$= \frac{1}{N}\sum_{i=1}^N v_i\left(v_i^Tv_j\right) = \frac{1}{N}\left(\sum_{i=1}^N v_iv_i^T\right)v_j$$

This can be computed much faster. -->