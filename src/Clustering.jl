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

    return norm(A1 - A2)^2
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