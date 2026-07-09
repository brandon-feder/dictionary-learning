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
    D::StridedMatrix{T1}, msss::Vector{<:MODSampleStruct}, min_cluster_size::Int,
    parallel::Bool
) where T1
    @assert allequal(size.(getproperty.(msss, :D), 1))
    (; m) = msss[1]
    k = size(D, 2)
    metric = ComplexAngularDist_Faked_Normalized()

    # collect data to cluster
    Ds = adapt(CPU(), reshape(stack(getproperty.(msss, :D)), m, :))
    Ds_f64 = reinterpret(Float64, Ds)
    normalize!.(eachcol(Ds_f64))
    
    # select centers
    sidx = dbscanpp_init(
        AllInit(), 
        Ds_f64, 
        metric,
        parallel
    )
    nntree = BallTree(Ds_f64, metric)

    # binary search for dbscan radious
    r = 1.0
    res = nothing
    for i in 1:20
        res = dbscanpp(Ds_f64, nntree, r;
            m=min_cluster_size,
            init=sidx,
            metric,
            parallel
        )
        
        nc = nclusters(res)
        if nc == k
            break
        elseif nc > k
            r  += 2.0^-i 
        else
            r -= 2.0^-i 
        end
    end

    if nclusters(res) != k
        throw(ErrorException("Expecting $k clusters but $(length(res.clusters)) were found"))
    end

    # for each cluster
    for i in 1:k
        # M contains the vectors in this cluster
        idxs = findall(isequal(i), assignments(res))
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
    alg::Symbol=:dbscan; min_cluster_size::Int=2, parallel::Bool=true
) where T1
    @assert alg ∈ [:dbscan, :kmed]

    if alg == :kmed
        mod_cluster_kmed!(D, msss)
    elseif alg == :dbscan
        mod_cluster_dbscan!(D, msss, min_cluster_size, parallel)
    end
end