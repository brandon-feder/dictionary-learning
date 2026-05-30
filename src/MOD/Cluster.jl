function mod_cluster!(
    D::StridedMatrix{T1}, mss::MODSampleStruct{T1, T2}
) where T1 where T2
    (; Ds, m, k) = mss

    # stack and flatten samples into a single matrix
    Ds_ = reshape(stack(Ds), m, :)

    # cluster
    res = kmedoids(adapt(CPU(), 1 .- abs.(Ds_' * Ds_)), k)

    # for each cluster
    for i in 1:k
        # M contains the vectors in this cluster
        idxs = findall(isequal(i), res.assignments)
        M = Ds_[:, idxs]

        # compute eigenvectors of M
        eig = eigen(Hermitian(M*M'))
        eigvindx = sortperm(adapt(CPU(), eig.values), by=λ->-λ)

        # column is top eigenvector
        view(D, :, i) .= view(eig.vectors, :, eigvindx[1])
    end

    return D
end