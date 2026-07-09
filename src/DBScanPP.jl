abstract type DBScanPPInit end

struct AllInit <: DBScanPPInit end

struct KCenterInit <: DBScanPPInit
    k::Int # number of points in center
end

struct UniformInit <: DBScanPPInit
    k::Int # number of points in center
end

@kwdef mutable struct DBscanPPTiming
    init::Real = 0.0 # time to choose centers
    core::Real = 0.0 # time to construct core points
    graph::Real = 0.0 # time to create clustering graph
    comps::Real = 0.0 # time to compute connected components
end

function dbscanpp_init(
    init::KCenterInit, X::AbstractMatrix{T}, metric::Metric,
    parallel::Bool=true
) where T
    @assert get_backend(X) == CPU()
    n = size(X, 2)
    k = init.k
    @assert 1 <= k <= n
    floop_ex = parallel ? ThreadedEx() : SequentialEx()

    sidx = Vector{Int}(undef, k)
    sidx[1] = rand(1:n)
    d = Vector{Float64}(undef, n)
    
    mindist = fill(Inf, n)
    for i in 2:k
        c = view(X, :, sidx[i-1])

        @floop floop_ex for i in 1:n
            d[i] = evaluate(metric, view(X, :, i), c)
        end
        
        @inbounds @simd for j in 1:n
            mindist[j] = min(mindist[j], d[j])
        end
        sidx[i] = argmax(mindist)
    end
    return sidx
end

function dbscanpp_init(
    init::AllInit, X::AbstractMatrix{T}, metric::Metric,
    parallel::Bool=true
) where T
    @assert get_backend(X) == CPU()
    return collect(1:size(X, 2))
end

function dbscanpp_init(
    init::UniformInit, X::AbstractMatrix{T}, metric::Metric,
    parallel::Bool=true
) where T
    @assert get_backend(X) == CPU()
    return randperm(size(X, 2))[1:init.k]
end

function dbscanpp(
    X::AbstractMatrix{T}, nntree::NNTree, ε::Real; m::Int=1, init::Union{DBScanPPInit, AbstractVector{Int}}=AllInit(),
    metric::Metric=Euclidean(), parallel = true,
    timing::DBscanPPTiming=DBscanPPTiming()
) where T
    n = size(X, 2)
    @assert get_backend(X) == CPU()
    @assert 1 <= m <= n
    floop_ex = parallel ? ThreadedEx() : SequentialEx()
    
    # choose centers
    if !isa(init, AbstractVector)
        timing.init = @elapsed begin
            sidx = dbscanpp_init(init, X, metric, parallel)
        end
    else
        sidx = init
        @assert length(sidx) == length(unique(sidx))
        @assert length(sidx) > 0
        @assert minimum(sidx) > 0
        @assert maximum(sidx) <= n
    end
    ns = length(sidx)
    
    # find core points
    timing.core = @elapsed begin
        @floop floop_ex for si in sidx
            @init begin
                idxs = Vector{Int}(undef, m)
                dists = Vector{Float64}(undef, m)
            end
            knn!(idxs, dists, nntree, view(X, :, si), m)
            res = dists[end] < ε ? si : Any[]
            @reduce(cidx = append!(Any[], res))
        end
    end

    # create clustering graph
    timing.graph = @elapsed begin
        @floop floop_ex for ci in cidx
            @init begin
                idxs = Vector{Int64}(undef, 0)
            end
            c = view(X, :, ci)
            empty!(idxs)
            inrange!(idxs, nntree, c, ε)
            el_local = [Graphs.SimpleEdge(ci, i) for i in idxs]
            @reduce(el = append!(Any[], el_local))
        end
        g = SimpleGraphFromIterator(el)
    end

    # get connected components
    timing.comps += @elapsed begin
        comps = connected_components(g)
    end

    # create DBScanResult objects
    @floop floop_ex for comp in comps
        if length(comp) >= m
            core = intersect(comp, cidx)
            isempty(core) && continue
            boundary = setdiff(comp, core)
            res_local = [DbscanCluster(length(comp), core, boundary)]
        else
            res_local = []
        end
        @reduce(clstr_list = append!(DbscanCluster[], res_local))
    end

    return DbscanResult(clstr_list, n)
end