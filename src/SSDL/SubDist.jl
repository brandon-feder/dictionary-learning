abstract type AbstractSSDLSubDistStruct{T1, T2} end

struct SSDLSubDistStruct{T1, T2} <: AbstractSSDLSubDistStruct{T1, T2}
    back::Backend
    m::Int
    s::Int
    t::Int
    nrm::Symbol
    S::StridedArray{T1} # m × s × t
    D::Symmetric{T2, <: StridedMatrix{T2}} # t × t

    function SSDLSubDistStruct(
        back::Backend, m::Int, s::Int, t::Int, nrm::Symbol,
        S::StridedArray{T1}, D::Symmetric{T2, <: StridedMatrix{T2}}
    ) where T1 where T2
        @assert T2 == real(T1)
        @assert back == get_backend(S)
        @assert DictionaryLearning.backsagree(S, D)
        @assert size(S) == (m, s, t)
        @assert size(D) == (t, t)
        @assert nrm ∈ [:fnorm, :opnorm]

        return new{T1, T2}(back, m, s, t, nrm, S, D)
    end
end

struct SSDLTrueSubDistStruct{T1, T2} <: AbstractSSDLSubDistStruct{T1, T2}
    back::Backend
    m::Int
    k::Int
    s::Int
    t::Int
    nrm::Symbol
    S::StridedArray{T1} # m × s × t
    D::Symmetric{T2, <: StridedMatrix{T2}} # t × t
    X::AbstractSparseMatrix{T1} # k × t
    C::Symmetric{Int, <: StridedMatrix{Int}} # t × t

    function SSDLTrueSubDistStruct(
        back::Backend, m::Int, k::Int, s::Int, t::Int, nrm::Symbol,
        S::StridedArray{T1}, D::Symmetric{T2, <: StridedMatrix{T2}}, 
        X::AbstractSparseMatrix{T1}, C::Symmetric{Int, <: StridedMatrix{Int}}
    ) where T1 where T2
        @assert T2 == real(T1)
        @assert back == get_backend(S)
        @assert DictionaryLearning.backsagree(S, D)
        @assert get_backend(C) == CPU() 
        @assert get_backend(X) == CPU()
        @assert size(S) == (m, s, t)
        @assert size(X) == (k, t)
        @assert size(D) == (t, t)
        @assert size(C) == (t, t)
        @assert nrm ∈ [:fnorm, :opnorm]

        return new{T1, T2}(back, m, k, s, t, nrm, S, D, X, C)
    end
end

struct SSDLSubDistWorkStruct{T1, T2}
    W::StridedArray{T1} # m × m × t
    S2::StridedArray{T1} # m × s × t
    d::StridedVector{T2} # t

    function SSDLSubDistWorkStruct(
        W::StridedArray{T1}, S2::StridedArray{T1},
        d::StridedVector{T2}
    ) where T1 where T2
        m, s, t = size(S2)

        @assert size(W) == (m, m, t)
        @assert size(S2) == (m, s, t)
        @assert length(d) == t
        @assert T2 == real(T1)

        return new{T1, T2}(W, S2, d)
    end
end

function SSDLSubDistStruct(
    ssrs::DictionaryLearning.AbstractSSDLSubRecStruct{T1},
    nrm::Symbol
) where T1
    (; back, m, s, t, S) = ssrs
    
    D = Symmetric(adapt(back, Matrix{real(T1)}(undef, t, t)))

    return SSDLSubDistStruct(back, m, s, t, nrm, S, D)
end

function SSDLTrueSubDistStruct(
    stsrs::DictionaryLearning.SSDLFakeSubRecStruct{T1},
    nrm::Symbol
) where T1
    (; back, m, k, s, t, S, X) = stsrs
    
    D = Symmetric(adapt(back, Matrix{real(T1)}(undef, t, t)))
    C = Symmetric(Matrix{Int}(undef, t, t))
    Xt = X[:, 1:t]

    return SSDLTrueSubDistStruct(back, m, k, s, t, nrm, S, D, Xt, C)
end

function SSDLSubDistWorkStruct(
    ssds::AbstractSSDLSubDistStruct{T1, T2}
) where T1 where T2
    (; back, m, s, t) = ssds

    W = adapt(back, Array{T1}(undef, m, m, t))
    S2 = adapt(back, Array{T1}(undef, m, s, t))
    d = adapt(back, Vector{T2}(undef, t))
    
    return SSDLSubDistWorkStruct(W, S2, d)
end

function sub_dist!(
    ssds::AbstractSSDLSubDistStruct{T1, T2},
    ssdws::SSDLSubDistWorkStruct{T1}
) where T1 where T2
    (; m, t, nrm, S, D) = ssds
    (; W, S2, d) = ssdws
    CUBLAS.gemm_strided_batched!('N', 'C', 1.0, S, S, 0.0, W)

    for i in 1:t
        S_ = view(S, :, :, i+1:t)
        S2_ = view(S2, :, :, i+1:t)
        Wi = view(W, :, :, i)
        D_ = view(parent(D), i, i+1:t) # upper triangle
        d_ = view(d, i+1:t)

        copyto!(S2_, S_)
        CUBLAS.gemm_strided_batched!(
            'N', 'N', 1.0, reshape(Wi, m, m, 1), S_, -1.0, S2_
        )

        if nrm == :fnorm
            batched_norm!(d_, S2_)
        elseif nrm == :opnorm
            batched_opnorm!(d_, S2_)
        end

        D_ .= d_
    end
    
    # clear diagonal
    D.data[1:t+1:t^2] .= 0.0
end

function sub_dist!(
    ssds::SSDLTrueSubDistStruct{T1, T2},
    ssdws::SSDLSubDistWorkStruct{T1}
) where T1 where T2
    (; t, C, X) = ssds

    invoke(
        sub_dist!, 
        Tuple{
            AbstractSSDLSubDistStruct{T1, T2}, 
            SSDLSubDistWorkStruct{T1}
        }, 
        ssds, ssdws
    )

    for i in 1:t
        nzi = findnz(view(X, :, i))[1]
        for j in i+1:t
            nzj = findnz(view(X, :, j))[1]
            C.data[i, j] = length(intersect(nzi, nzj))
        end
    end

    # clear diagonal
    C.data[1:t+1:t^2] .= 0.0
end