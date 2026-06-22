abstract type AbstractSSDLSubDistStruct{T1, T2} end

struct SSDLSubDistStruct{T1, T2} <: AbstractSSDLSubDistStruct{T1, T2}
    back::Backend
    m::Int
    s::Int
    t::Int
    dist::Symbol
    S::StridedArray{T1} # m × s × t
    D::Symmetric{T2, <: StridedMatrix{T2}} # t × t

    function SSDLSubDistStruct(
        back::Backend, m::Int, s::Int, t::Int, dist::Symbol,
        S::StridedArray{T1}, D::Symmetric{T2, <: StridedMatrix{T2}}
    ) where T1 where T2
        @assert T2 == real(T1)
        @assert back == get_backend(S)
        @assert DictionaryLearning.backsagree(S, D)
        @assert size(S) == (m, s, t)
        @assert size(D) == (t, t)
        @assert dist ∈ [:spa, :fnorm]

        return new{T1, T2}(back, m, s, t, dist, S, D)
    end
end

struct SSDLTrueSubDistStruct{T1, T2} <: AbstractSSDLSubDistStruct{T1, T2}
    back::Backend
    m::Int
    k::Int
    s::Int
    t::Int
    dist::Symbol
    S::StridedArray{T1} # m × s × t
    D::Symmetric{T2, <: StridedMatrix{T2}} # t × t
    X::AbstractSparseMatrix{T1} # k × t
    C::Symmetric{Int, <: StridedMatrix{Int}} # t × t

    function SSDLTrueSubDistStruct(
        back::Backend, m::Int, k::Int, s::Int, t::Int, dist::Symbol,
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
        @assert dist ∈ [:spa, :fnorm]

        return new{T1, T2}(back, m, k, s, t, dist, S, D, X, C)
    end
end

function SSDLSubDistStruct(
    ssrs::DictionaryLearning.AbstractSSDLSubRecStruct{T1},
    dist::Symbol=:spa
) where T1
    (; back, m, s, t, S) = ssrs
    
    D = Symmetric(adapt(back, Matrix{real(T1)}(undef, t, t)))

    return SSDLSubDistStruct(back, m, s, t, dist, S, D)
end

function SSDLTrueSubDistStruct(
    stsrs::DictionaryLearning.SSDLFakeSubRecStruct{T1},
    dist::Symbol
) where T1
    (; back, m, k, s, t, S, X) = stsrs
    
    D = Symmetric(adapt(back, Matrix{real(T1)}(undef, t, t)))
    C = Symmetric(Matrix{Int}(undef, t, t))
    Xt = X[:, 1:t]

    return SSDLTrueSubDistStruct(back, m, k, s, t, dist, S, D, Xt, C)
end

function sub_dist_true!(
    ssds::SSDLTrueSubDistStruct{T1, T2}
) where T1 where T2
    (; t, C, X) = ssds

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

@kernel function pairwise_opnorm_kernel!(
    D, @Const(S), ::Val{s}
) where s
    m, _, t = size(S)

    # get global index
    gi::Int = @index(Global)

    # get index in upper triangle
    i::Int = t-floor(Int, (sqrt(8*(t*(t+1)/2-gi)+1)-1)/2)
    j::Int = gi-div((i-1)*(2*t-i+2),2)+(i-1)

    # scratch space
    C = @MMatrix zeros(eltype(S), s, s)

    # Compute C = A' * B
    for x in 1:m
        for v in 1:s
            for u in 1:s
                @inbounds C[u, v] += conj(S[x, u, i]) * S[x, v, j]
            end
        end
    end

    # convert to SMatrix for allocation-free
    C_static = SMatrix(C)
    
    # do power iteration
    if iszero(C_static)
        op_norm = zero(real(eltype(S)))
    else
        # initial guess vector
        v = @SVector ones(eltype(S), s)
        v = v ./ norm(v)
        
        for _ in 1:25
            u = C_static * v
            u_norm = norm(u)
            u = u_norm > 0 ? u ./ u_norm : u 
            
            v_new = C_static' * u
            v_norm = norm(v_new)
            v = v_norm > 0 ? v_new ./ v_norm : v_new
        end
        
        op_norm = norm(C_static * v)
    end

    # Write to the output matrix D
    D[i, j] = acos(op_norm)
end

function sub_dist_spa!(
    ssds::AbstractSSDLSubDistStruct{T1, T2}
) where T1 where T2
    (; back, dist, s, t, S, D) = ssds
    @assert dist == :spa
    
    # 16x16 is a standard, efficient workgroup size for 2D grids
    kernel! = pairwise_opnorm_kernel!(back, (256,))
    kernel!(D.data, S, Val(s), ndrange=(div(t*(t+1),2),))
    KernelAbstractions.synchronize(back)
end


@kernel function pairwise_fnorm_kernel!(
    D, @Const(S), @Const(C)
)
    m, s, t = size(S)

    # get global index
    gi::Int = @index(Global)

    # get index in upper triangle
    i::Int = t-floor(Int, (sqrt(8*(t*(t+1)/2-gi)+1)-1)/2)
    j::Int = gi-div((i-1)*(2*t-i+2),2)+(i-1)
    
    acc1 = zero(real(eltype(D)))
    @inbounds begin
        for u in 1:m
            for v in 1:s
                acc2 = zero(eltype(D))
                for w in 1:m
                    acc2 = acc2 + conj(C[w, u, i])*S[w,v,j]
                end
                acc1 = acc1 + abs(acc2)^2
            end
        end
    end
    D[i,j] = sqrt(acc1)
end

function sub_dist_fnorm!(
    ssds::DictionaryLearning.AbstractSSDLSubDistStruct{T1, T2}
) where T1 where T2
    (; back, dist, t, m, S, D) = ssds
    @assert dist == :fnorm

    # compute I - Ai*Ai' for all Ai
    C = S ⊠ batched_adjoint(S)
    C .= adapt(back, I(m)) .- C

    kernel! = pairwise_fnorm_kernel!(back, (256,))
    kernel!(D.data, S, C, ndrange=(div(t*(t+1),2),))
    KernelAbstractions.synchronize(back)
end

function sub_dist!(
    ssds::AbstractSSDLSubDistStruct{T1, T2},
) where T1 where T2
    (; dist) = ssds

    if dist == :spa
        sub_dist_spa!(ssds)
    elseif dist == :fnorm
        sub_dist_fnorm!(ssds)
    end

    if isa(ssds, SSDLTrueSubDistStruct{T1, T2})
        sub_dist_true!(ssds)
    end
end