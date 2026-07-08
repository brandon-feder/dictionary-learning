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
    init::KCenterInit, X::Matrix{T}, metric::Metric,
    paralell::Bool=true
) where T
    n = size(X, 2)
    k = init.k
    @assert 1 <= k <= n

    sidx = Vector{Int}(undef, k)
    sidx[1] = rand(1:n)

    mindist = fill(Inf, n)
    for i in 2:k
        c = view(X, :, sidx[i-1])
        d = colwise(metric, X, c)
        @inbounds @simd for j in 1:n
            mindist[j] = min(mindist[j], d[j])
        end
        sidx[i] = argmax(mindist)
    end
    return sidx
end

function dbscanpp_init(
    init::AllInit, X::Matrix{T}, metric::Metric,
    paralell::Bool
) where T
    return collect(1:size(X, 2))
end

function dbscanpp_init(
    init::UniformInit, X::Matrix{T}, metric::Metric,
    paralell::Bool
) where T
    return randperm(size(X, 2))[1:init.k]
end

function dbscanpp(
    X::Matrix{T}, nntree::NNTree, ε::Real; m::Int=1, init::Union{DBScanPPInit, AbstractVector{Int}}=AllInit(),
    metric::Metric=Euclidean(), paralell = true,
    timing::DBscanPPTiming=DBscanPPTiming()
) where T
    n = size(X, 2)
    floop_ex = paralell ? ThreadedEx() : SequentialEx()
    
    # check arguments
    @assert 1 <= m <= n
    
    # choose centers
    if !isa(init, AbstractVector)
        timing.init = @elapsed begin
            sidx = dbscanpp_init(init, X, metric, paralell)
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
        core = intersect(comp, cidx)
        isempty(core) && continue
        boundary = setdiff(comp, core)
        res_local = DbscanCluster(length(comp), core, boundary)
        @reduce(clstr_list = append!(DbscanCluster[], [res_local]))
    end

    return DbscanResult(clstr_list, n)
end