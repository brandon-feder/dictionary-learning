function formgraph(
    S::AbstractArray{T}, τ::Real
) where T
    m, s, t = size(S)

    # create graph and add nodes
    g = SimpleGraph()
    add_vertices!(g, t)

    # collect all pairs of subspaces
    l1 = Vector{Int}(undef, binomial(t, 2))
    l2 = similar(l1)
    for (i, (j, k)) in enumerate(combinations(1:t,2))
        l1[i] = j
        l2[i] = k
    end

    # compute subspace distances
    sds = Vector{Float64}(undef, binomial(t, 2))
    subdists!(sds, view(S, :, :, l1), view(S, :, :, l2))

    # add edges between near subspaces
    for (sd, i, j) in zip(sds, l1, l2)
        if sd < τ
            add_edge!(g, i, j)
        end
    end

    return g
end

function truetruegraph(
    X::AbstractSparseMatrix{T}, t::Int, s::Int
) where T
    # create graph and add nodes
    g = SimpleGraph()
    add_vertices!(g, t)

    # collect all pairs of subspaces
    l1 = Vector{Int}(undef, binomial(t, 2))
    l2 = similar(l1)
    for (i, (j, k)) in enumerate(combinations(1:t,2))
        l1[i] = j
        l2[i] = k
    end

    # add edges
    for (i,j) in zip(l1,l2)
        xi = X[:,i]
        xj = X[:,j]
        supi = findnz(xi)[1]
        supj = findnz(xj)[1]
        if length(intersect(supi,supj)) > 0
            add_edge!(g, (i,j))
        end
    end

    return g
end

function nedgedif(g1::AbstractGraph, g2::AbstractGraph)
    A1 = adjacency_matrix(g1)
    A2 = adjacency_matrix(g2)

    return round(Int64, 0.5*norm(A1 - A2)^2)
end

function truesimps(
    X::AbstractSparseMatrix{T}, t::Int
) where T
    k, _ = size(X)

    # create simplexes
    simps = Simplices()
    for (i, xi) in enumerate(eachcol(view(X, :, 1:t)))
        sup = findnz(xi)[1]
        push!(simps, i, sup)
    end

    return simps
end
function toedgegraph(g::AbstractGraph)
    nedges = ne(g)
    edgelst = collect(edges(g))

    bufs = [Int[] for _ in 1:nedges]
    @floop ThreadedEx() for i in 1:nedges
        for j in (i+1):nedges
            if edgelst[i].src == edgelst[j].src || 
                edgelst[i].src == edgelst[j].dst || 
                edgelst[i].dst == edgelst[j].src || 
                edgelst[i].dst == edgelst[j].dst
                    push!(bufs[i], j)
            end
        end
    end

    eg = SimpleGraph(nedges)
    for (i, bufi) in enumerate(bufs)
        for j in bufi
            add_edge!(eg, i, j)
        end
    end

    return eg
end

function cluster(g::AbstractGraph, k::Int, s::Int, back::Backend=CPU(), alg::Symbol=:edge)
    # only implemented algorithm
    @assert alg == :edge

    t, nedges = nv(g), ne(g)
    edgelst = collect(edges(g))
    eg = toedgegraph(g)

    # get adjacency matrix
    A = Float64.(adapt(back, adjacency_matrix(eg)))

    # symetric noramlized laplacian
    d = degree(eg)
    Disqt = adapt(back, Diagonal([x > 0 ? 1.0/sqrt(x) : 0.0 for x in d]))
    L = I - Disqt * (A * Disqt)

    # compute features
    x₀ = adapt(back, randn(nedges)) # initial guess
    res = eigsolve(x -> L*x, x₀, k+10, :SR, krylovdim=2*k, issymmetric=true)
    U = hcat(res[2][2:k+1]...)

    # normalize rows
    U ./= sqrt.(sum(abs2, U, dims=2))
    U_cpu = adapt(CPU(), U)

    # do clustering
    km = kmeans(U_cpu', k)
    agnmts = assignments(km)

    # map edges back to nodes
    clstcounts = [zeros(Int, k) for _ in 1:t]
    for (i, ei) in enumerate(edgelst)
        u, v = src(ei), dst(ei)
        cid = agnmts[i]
        clstcounts[u][cid] += 1
        clstcounts[v][cid] += 1
    end
    
    simps = Simplices()
    for i in 1:t
        supi = partialsortperm(clstcounts[i], 1:s, rev=true)
        push!(simps, i, supi)
    end

    return simps
end