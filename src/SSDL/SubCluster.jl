struct SSDLSubClusterStruct{T1, T2}
    back::Backend
    m::Int
    s::Int
    t::Int
    S::StridedArray{T1} # m × s × t
    D::Symmetric{T2, <: StridedMatrix{T2}} # t × t
    τ::T2
    g::Ref{<: SimpleGraph}
    ge::Ref{<: SimpleGraph}

    function SSDLSubClusterStruct(
        back::Backend, m::Int, s::Int, t::Int,
        S::StridedArray{T1}, D::Symmetric{T2, <: StridedMatrix{T2}},
        τ::T2, g::Ref{<: SimpleGraph}, ge::Ref{<: SimpleGraph}
    ) where T1 where T2
        @assert size(S) == (m, s, t)
        @assert size(D) == (t, t)
        @assert backsagree(S, D)
        @assert get_backend(S) == back
        @assert τ >= 0.0

        return new{T1, T2}(back, m, s, t, S, D, τ, g, ge)
    end
end

function SSDLSubClusterStruct(
    ssds::AbstractSSDLSubDistStruct{T1, T2},
    τ::Real
) where T1 where T2
    (; back, m, s, t, S, D) = ssds

    g = SimpleGraph(t)
    ge = SimpleGraph()

    return SSDLSubClusterStruct(back, m, s, t, S, D, τ, Ref(g), Ref(ge))
end

struct SSDLSubClusterWorkStruct{T}
    D_cpu::Symmetric{T, <: StridedMatrix{T}} # t × t

    function SSDLSubClusterWorkStruct(
        D_cpu::Symmetric{T, <: StridedMatrix{T}}
    ) where T
        @assert get_backend(D_cpu) == CPU()
        
        return new{T}(D_cpu)
    end
end

function SSDLSubClusterWorkStruct(
    sscs::SSDLSubClusterStruct{T1, T2}
) where T1 where T2
    (; back, t) = sscs
    D_cpu = Symmetric(Matrix{T2}(undef, t, t))

    return SSDLSubClusterWorkStruct(D_cpu)
end

function get_edge_graph!(g::AbstractGraph, ge::AbstractGraph)
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

    for (i, bufi) in enumerate(bufs)
        for j in bufi
            add_edge!(ge, i, j)
        end
    end

    return ge
end

function ssdl_sub_cluster!(
    sscs::SSDLSubClusterStruct{T1, T2},
    sscws::SSDLSubClusterWorkStruct{T2},
) where T1 where T2
    (; g, ge, t, τ, D) = sscs
    (; D_cpu) = sscws
    
    g = g[]
    ge = ge[]
    
    # move stuff to CPU
    copyto!(D_cpu.data, D.data)

    # build graph
    for i in 1:t
        for j in i+1:t
            sd = D_cpu[j, i]
            if sd < τ
                add_edge!(g, i, j)
            end
        end
    end
    
    # build edge graph
    add_vertices!(ge, ne(g))
    get_edge_graph!(g, ge)
end