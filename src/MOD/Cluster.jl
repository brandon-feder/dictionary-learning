function mod_cluster_kmed!(
    D::StridedMatrix{T1}, msss::Vector{<:MODSampleStruct}
) where T1
    @assert allequal(size.(getproperty.(msss, :D)))
    (; m, k) = msss[1]

    Ds = reshape(stack(getproperty.(msss, :D)), m, :)

    # cluster
    dists = adapt(CPU(), 1 .- abs.(Ds'*Ds))
    res = kmedoids(dists, k)

    # for each cluster
    for i in 1:k
        # M contains the vectors in this cluster
        idxs = findall(isequal(i), res.assignments)
        M = Ds[:, idxs]

        # compute eigenvectors of M
        eig = eigen(Hermitian(M*M'))
        eigvindx = sortperm(adapt(CPU(), eig.values), by=λ->-λ)

        # column is top eigenvector
        view(D, :, i) .= view(eig.vectors, :, eigvindx[1])
    end

    return D
end

function mod_cluster_dbscan!(
    D::StridedMatrix{T1}, msss::Vector{<:MODSampleStruct},
    radius::T2
) where T1 where T2
    @assert allequal(size.(getproperty.(msss, :D), 1))
    (; m) = msss[1]
    k = size(D, 2)

    Ds = reshape(stack(getproperty.(msss, :D)), m, :)
    normalize!.(eachcol(Ds))
    dists = adapt(CPU(), 1 .- abs.(Ds'*Ds))

    res = dbscan(dists, radius; metric=nothing, min_cluster_size=2)

    if length(res.clusters) != k
        throw(ErrorException("Expecting $k clusters but $(length(res.clusters)) were found"))
    end

    # for each cluster
    for i in 1:k
        # M contains the vectors in this cluster
        idxs = findall(isequal(i), res.assignments)
        M = Ds[:, idxs]

        # compute eigenvectors of M
        eig = eigen(Hermitian(M*M'))
        eigvindx = sortperm(adapt(CPU(), eig.values), by=λ->-λ)

        # column is top eigenvector
        view(D, :, i) .= view(eig.vectors, :, eigvindx[1])
    end
end

function mod_cluster!(
    D::StridedMatrix{T1}, msss::Vector{<:MODSampleStruct},
    alg::Symbol=:dbscan; dbscan_radius::T2=0.01
) where T1 where T2
    @assert T2 == real(T1)
    @assert alg ∈ [:dbscan, :kmed]

    if alg == :kmed
        mod_cluster_kmed!(D, msss)
    elseif alg == :dbscan
        mod_cluster_dbscan!(D, msss, dbscan_radius)
    end
end