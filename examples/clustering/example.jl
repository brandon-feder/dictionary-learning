using SparseArrays
using LinearAlgebra
using DataStructures
using PProf
using StatsBase
using Combinatorics
using Graphs
using DataStructures
using Clustering
using KrylovKit
using Random

import Plots
import StatsPlots


using DictionaryLearning

include("../debug.jl")

include("clustering-test.jl")


function gendata(k::Int, n::Int, s::Int)
    """
    Generate an `k x n` matrix consisting of 
    columns with sparsity `s` and with non-zero
    entries taking values 1 or -1.
    """

    # construct COO format of SparseMatrix
    C = repeat(1:n, inner=[s])
    R = Vector{Int64}(undef, s*n)
    V = rand((-1.0, 1.0), s*n)

    # choose support in each column
    perm = collect(1:k)
    for j in 1:n

        # choose support
        shuffle!(perm)
        sup = view(perm, 1:s)
        copyto!(view(R, s*(j-1)+1:s*j), sup)
    end

    # create sparse matrix
    X = sparse(R, C, V, k, n)

    return X
end

function gendict(m::Int, k::Int)
    """
    Generate an `M x K` dictionary consisting of 
    columns which are distributed about the M-sphere
    uniformly at random.
    """

    D = randn(m, k)
    normalize!.(eachcol(D))
    return D
end

function main()
    m::Int = 50
    k::Int = 50
    n::Int = m^3
    s::Int = 3
    t::Int = 500

    println("$INF m = $m")
    println("$INF k = $k")
    println("$INF n = $n")
    println("$INF s = $s")
    println("$INF t = $t")

    # generate data
    D = gendict(m, k)
    X = gendata(k, n, s)

    # compute correlation covariance
    Σ = Matrix{Float64}(undef, m, m)
    work = Vector{Float64}(undef, m*k)
    println("$INF computing Σ")
    compΣ!(Σ, D, X, work)

    # compute G
    G = Matrix{Float64}(undef, n, t)
    work = Vector{Float64}(undef, k*k + k*t)
    println("$INF computing G")
    compG!(G, D, X, 1:t, work)

    # compute V
    YT = X' * D'
    V = Matrix{Float64}(undef, n, m^2)
    println("$INF computing V")
    compV!(V, YT)

    # compute T
    Z = Array{Float64}(undef, m, m, t)
    println("$INF computing Z")
    compZ!(Z, G, V)

    # recover subspaces
    S = Array{Float64}(undef, m, s, t)
    println("$INF computing subspaces")
    compS!(S, Z, Σ, s)

    # get true subspaces
    St = similar(S)
    println("$INF computing true subspaces")
    truesubs!(St, D, X[:, 1:t], s)

    # plot info about subspaces
    p1 = Plots.plot(
        plot_sd_err(S, St), plot_sd_by_sup(S, X, s, t), 
        layout=(1,2), size=(1000, 500)
    )
    display(p1)

    # get τ from user
    τ = 0.0
    while true
        try    
            print("choose τ: ")
            τ = parse(Float64, readline())
            break
        catch e
            if !isa(e, ArgumentError)
                break
            end
        end
    end
    # τ = 1.5
    println("$INF τ = $τ")

    # form graphs
    println("$INF forming graphs")
    g = formgraph(S, τ)
    gt = truetruegraph(X, t, s)
    println("$TAB$INF # total edges: $(ne(g))")
    println("$TAB$INF # edges different: $(nedgedif(g, gt))")

    # compute true simplicies
    println("$INF computing simplicies")

    # used to plot recovery
    p = Plots.plot()

    # recovery from true simplicies
    simps_true = truesimps(X, t)
    Dr = similar(D)
    recdict!(Dr, S, simps_true)
    err = maximum(abs.(D'*Dr), dims=2)
    StatsPlots.dotplot!(err, alpha=0.9, label="true", normalize=:probability)

    # recover from edge graph spectral using true graph
    simps_eg = cluster_edge(gt, k, s)
    Dr = similar(D)
    recdict!(Dr, S, simps_eg)
    err = maximum(abs.(D'*Dr), dims=2)
    StatsPlots.dotplot!(err, alpha=0.9, label="edge true", normalize=:probability)

    # recover from edge graph spectral
    simps_eg = cluster_edge(g, k, s)
    Dr = similar(D)
    recdict!(Dr, S, simps_eg)
    err = maximum(abs.(D'*Dr), dims=2)
    StatsPlots.dotplot!(err, alpha=0.9, label="edge", normalize=:probability)

    # # recover simplicies using spectral clustering on true graph
    simps_spec = cluster_spec(gt, k, s)
    Dr = similar(D)
    recdict!(Dr, S, simps_spec)
    err = maximum(abs.(D'*Dr), dims=2)
    StatsPlots.dotplot!(err, alpha=0.9, label="spectral true", normalize=:probability)

    # recover simplicies using spectral clustering
    simps_spec = cluster_spec(g, k, s)
    Dr = similar(D)
    recdict!(Dr, S, simps_spec)
    err = maximum(abs.(D'*Dr), dims=2)
    StatsPlots.dotplot!(err, alpha=0.9, label="spectral", normalize=:probability)

    thresh = 10
    niters = 10*ceil(Int, k * log(k)^2)
    println("$INF thresh = $thresh")
    println("$INF niters = $niters")

    # recovery using overlapping algorithm with true graph
    simps_over = cluster_overlap(gt, k, s, (t * s) / (4 * k), ceil(Int, k * log(k)^2))
    Dr = similar(D)
    recdict!(Dr, S, simps_over)
    err = maximum(abs.(D'*Dr), dims=2)
    StatsPlots.dotplot!(err, alpha=0.9, label="overlapping true", normalize=:probability)

    # recovery using overlapping algorithm
    simps_over = cluster_overlap(g, k, s, (t * s) / (4 * k), ceil(Int, k * log(k)^2))
    Dr = similar(D)
    recdict!(Dr, S, simps_over)
    err = maximum(abs.(D'*Dr), dims=2)
    StatsPlots.dotplot!(err, alpha=0.9, label="overlapping", normalize=:probability)

    display(p)

    # recover dictionary
    # println("$INF recovering dictionary")
end

function plot_sd_err(asubs::Array, tsubs::Array)
    # compute distance from true subspaces
    sd = Vector{Float64}(undef, size(asubs, 3))
    subdists!(sd, tsubs, asubs)

    println(mean(sd))

    p = Plots.histogram(
        sd, 
        title="subspace approximation errors", 
        xlabel="error",
        legend=false
    )
    
    return p
end

function plot_sd_by_sup(S::AbstractArray, X::AbstractMatrix, s::Int, t::Int)
    # the distance between subspaces that share support
        #  of size `n` will be stored in `samps[n]`
    samps = DefaultDict{Int64, Vector{Float64}}(()->Float64[])
    
    # collect all pairs of subspaces
    l1 = Vector{Int}(undef, binomial(t, 2))
    l2 = similar(l1)
    for (i, (j, k)) in enumerate(combinations(1:t,2))
        l1[i] = j
        l2[i] = k
    end

    # pair up subspaces
    X1 = view(X, :, l1)
    X2 = view(X, :, l2)

    S1 = view(S, :, :, l1)
    S2 = view(S, :, :, l2)

    # determine support of each pair
    nss = Vector{Int}(undef, binomial(t, 2))
    nsupshared!(nss, X1, X2)

    # determine subspace distances
    sds = Vector{Float64}(undef, binomial(t, 2))
    subdists!(sds, S1, S2)

    # sort subdists based on support
    for (ss, sd) in zip(nss, sds)
        push!(samps[ss], sd)
    end

    # create plot
    p = Plots.plot(
        title="subspace distance by support shared",
        xlabel="distance"
    )
    for i in 0:s-1
        if length(samps[i]) > 0
            Plots.histogram!(samps[i], label="s=$i", alpha=0.9)
        end
    end

    return p
end

main()