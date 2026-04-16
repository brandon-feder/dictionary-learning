function cluster_spec(g::AbstractGraph, k::Int, s::Int)
    t = nv(g)

    # get adjacency matrix
    A = Float64.(adjacency_matrix(g))

    # symetric noramlized laplacian
    d = degree(g)
    Disqt = Diagonal([x > 0 ? 1.0/sqrt(x) : 0.0 for x in d])
    L = Symmetric(I - Disqt * A * Disqt)

    # comoute features
    x₀ = randn(t) # initial guess
    res = eigsolve(L, x₀, k, :SR, krylovdim=2*k)

    U = hcat(res[2]...)

    # normalize rows
    U ./= norm(eachrow(U))

    # do clustering
    km = kmeans(U', k)
    cents = km.centers

    # form simplicies
    simps = Simplices()
    
    for (i,ui) in enumerate(eachrow(U))
        # compute distances to each centeroid
        dists = map(vi -> norm(vi .- ui), eachcol(cents))

        # assign to s closest clusters
        sinds = partialsortperm(dists, 1:s)
        push!(simps, i, sinds)
    end

    return simps
end

function cluster_overlap(g::AbstractGraph, k::Int, s::Int, thresh::Real, niters::Int)
    t = nv(g)
    
    # candidate simplices
    cand_simps = Dict{Set{Int}, Tuple{Int, Int}}()
    
    # get iterator to pairs of edges
    all_edges = collect(edges(g))

    # main loop
    for _ in 1:niters
        edge = rand(all_edges)
        u, v = src(edge), dst(edge)
        
        # get common neighbors
        cmn_uv = intersect(Set(neighbors(g, u)), Set(neighbors(g, v)))
        
        # buld Suv
        s_uv = Set{Int}([u, v])
        for w in vertices(g)
            ncmn_uvw = length(intersect(neighbors(g, w), cmn_uv))
            if ncmn_uvw >= thresh
                push!(s_uv, w)
            end
        end
        
        # store unique sets found
        if !haskey(cand_simps, s_uv)
            cand_simps[s_uv] = (u, v)
        end
    end
    
    # # 6. Deletion Step: Filter non-identifying sets [cite: 453, 469]
    # We delete S_uv if u, v are contained in a strictly smaller set S_ab
    final_simps = DefaultDict{Int, Set{Int}}(()->Set{Int}())
    all_found = collect(keys(cand_simps))
    
    i = 1
    for (s_uv, (u, v)) in cand_simps
        is_minimal = true
        for s_ab in all_found
            if s_uv == s_ab
                continue
            end
            
            # Check if {u, v} ⊆ S_ab AND |S_ab| < |S_uv|
            if u in s_ab && v in s_ab && length(s_ab) < length(s_uv)
                is_minimal = false
                break
            end
        end
        
        if is_minimal
            final_simps[i] = s_uv
            i += 1
        end
    end

    simps = Simplices(final_simps, DefaultDict{Int, Set{Int}}(()->Set{Int}()))

    return simps
end

function cluster_edge(g::AbstractGraph, k::Int, s::Int)
    t, nedges = nv(g), ne(g)
    edgelst = collect(edges(g))
    eg = SimpleGraph(nedges)
    
    # build adjacency for the edge graph
    for i in 1:nedges
        for j in (i+1):nedges
            if edgelst[i].src == edgelst[j].src || 
               edgelst[i].src == edgelst[j].dst || 
               edgelst[i].dst == edgelst[j].src || 
               edgelst[i].dst == edgelst[j].dst
                add_edge!(eg, i, j)
            end
        end
    end

    # get adjacency matrix
    A = Float64.(adjacency_matrix(eg))

    # symetric noramlized laplacian
    d = degree(eg)
    Disqt = Diagonal([x > 0 ? 1.0/sqrt(x) : 0.0 for x in d])
    L = Symmetric(I - Disqt * A * Disqt)

    # comoute features
    x₀ = randn(nedges) # initial guess
    res = eigsolve(L, x₀, k+10, :SR, krylovdim=2*k)
    U = hcat(res[2][2:k+1]...)

    println(res[1][2:k+5])

    # normalize rows
    U ./= norm(eachrow(U))

    # do clustering
    km = kmeans(U', k)
    agnmts = assignments(km)

    # map edges back to nodes
    clst_counts = [zeros(Int, k) for _ in 1:t]
    for (i, ei) in enumerate(edgelst)
        u, v = src(ei), dst(ei)
        cid = agnmts[i]
        clst_counts[u][cid] += 1
        clst_counts[v][cid] += 1
    end
    
    simps = Simplices()
    for i in 1:t
        supi = partialsortperm(clst_counts[i], 1:s, rev=true)
        push!(simps, i, supi)
    end

    return simps
end


# function function cluster_overlapping_nodes(g::AbstractGraph, t::Int, s::Int, k::Int)
#     # 1. Construct the Line Graph
#     # In L(G), nodes represent edges of G
#     lg, edge_map = line_graph_with_map(g)
#     num_edges = ne(g)
    
#     # 2. Spectral Embedding of the Line Graph
#     # We compute the normalized Laplacian of the Line Graph
#     L_mat = normalized_laplacian(lg)
#     # Get the eigenvectors (excluding the first trivial one)
#     # We need enough dimensions to represent k clusters, usually k or k-1
#     vals, vecs = eigen(Matrix(L_mat))
#     embedding = vecs[:, 2:k+1]' # Shape: (k, num_edges)

#     # 3. K-means clustering on edges
#     results = kmeans(embedding, k)
#     edge_assignments = assignments(results) # Vector of length num_edges
    
#     # 4. Map edge clusters back to nodes
#     # node_cluster_counts[node_idx][cluster_idx] = count
#     node_cluster_counts = [zeros(Int, k) for _ in 1:t]
    
#     for (i, edge_idx) in enumerate(edge_map)
#         u, v = src(edge_idx), dst(edge_idx)
#         cluster_id = edge_assignments[i]
#         node_cluster_counts[u][cluster_id] += 1
#         node_cluster_counts[v][cluster_id] += 1
#     end
    
#     # 5. Final Assignment: Pick top 's' clusters for each node
#     final_assignments = Vector{Vector{Int}}(undef, t)
#     for i in 1:t
#         # Sort indices by count descending
#         top_clusters = partialsortperm(node_cluster_counts[i], 1:s, rev=true)
#         final_assignments[i] = top_clusters
#     end
    
#     return final_assignments
# end

# """
# Helper to create line graph and keep track of edge indexing
# """
# function line_graph_with_map(g::AbstractGraph)
#     edges_list = collect(edges(g))
#     m = length(edges_list)
#     lg = SimpleGraph(m)
    
#     # Build adjacency for the line graph
#     for i in 1:m
#         for j in (i+1):m
#             # If edges share a vertex, connect them in Line Graph
#             if edges_list[i].src == edges_list[j].src || 
#                edges_list[i].src == edges_list[j].dst || 
#                edges_list[i].dst == edges_list[j].src || 
#                edges_list[i].dst == edges_list[j].dst
#                 add_edge!(lg, i, j)
#             end
#         end
#     end
#     return lg, edges_list
# end