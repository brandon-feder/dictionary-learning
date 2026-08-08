mutable struct GELMAStruct{T1 <: Union{Real, Complex}, T2 <: Real}
    const back::Backend
    const m::Int
    const k::Int
    const n::Int
    const D::StridedMatrix{T1}
    const Y::StridedMatrix{T1}
    const dt::T2
    const τ::T2
    const ε::T2
    const max_iters::Int
    const sparse_ls_cutoff::T2
    err_hist::Vector{T2}
    sparsity_hist::Vector{T2}
    elap_hist::Vector{T2}

    function GELMAStruct(
        back::Backend, m::Int, k::Int, n::Int,
        D::StridedMatrix{T1}, Y::StridedMatrix{T1},
        dt::T2, τ::T2, ε::T2, max_iters::Int, sparse_ls_cutoff::T2,
        err_hist::Vector{T2}, sparsity_hist::Vector{T2}, 
         elap_hist::Vector{T2}
    ) where T1 where T2
        @assert get_backend(Y) == back
        @assert dt > 0.0 && τ > 0.0 && ε >= 0.0 && max_iters > 0
        @assert get_backend(D) == back
        @assert size(D) == (m, k)
        @assert size(Y) == (m, n)
        @assert sparse_ls_cutoff <= 1.0

        return new{T1, T2}(
            back, m, k, n, D, Y, dt, τ, ε, max_iters,
            sparse_ls_cutoff, err_hist, sparsity_hist, elap_hist
        )
    end
end

struct GELMAWorkStruct{T <: Union{Real, Complex}}
    Xadj::StridedMatrix{T}  # n × k
    Z::StridedMatrix{T}     # m × n

    # these can overlap
    W1::StridedMatrix{T}    # m × n
    A::StridedMatrix{T}     # k × k

    function GELMAWorkStruct(
        Xadj::StridedMatrix{T},
        Z::StridedMatrix{T}, W1::StridedMatrix{T}, 
        A::StridedMatrix{T}
    ) where T
        n, k = size(Xadj)
        m = size(Z, 1)

        @assert backsagree(Xadj, Z, W1, A)
        @assert size(W1) == (m, n)
        @assert size(Xadj) == (n, k)
        @assert size(Z) == (m, n)

        return new{T}(Xadj, Z, W1, A)
    end
end

function GELMAStruct(
    D::StridedMatrix{T1}, Y::AbstractMatrix{T1}, k::Int, tau::T2;
    dt::T2=0.1, eps::T2=1e-2, max_iters::Int = 1000,
    sparse_ls_cutoff::T2 = 0.05
) where T1 <: Union{Real, Complex} where T2 <: Real
    m, n = size(Y)
    back = get_backend(Y)
    err_hist = Vector{T2}(undef, 0)
    sparsity_hist = Vector{T2}(undef, 0)
    elap_hist = Vector{T2}(undef, 0)

    return GELMAStruct(
        back, m, k, n, D, Y, dt, tau, eps,
        max_iters, sparse_ls_cutoff, err_hist, 
        sparsity_hist, elap_hist
    )
end

function GELMAWorkStruct(
    ms::GELMAStruct{T1, T2}
) where T1 where T2
    (; back, k, Y) = ms
    m, n = size(Y)
    
    Xadj = adapt(back, Matrix{T1}(undef, n, k))
    Z = adapt(back, Matrix{T1}(undef, m, n))
    work = adapt(back, Vector{T1}(undef, max(m*n, k*k)))
    W1 = reshape(view(work, 1:m*n), m, n)
    A = reshape(view(work, 1:k*k), k, k)

    return GELMAWorkStruct(Xadj, Z, W1, A)
end

function gelma_iter!(
    ms::GELMAStruct{T1, T2}, mws::GELMAWorkStruct{T1},
    Xadjsp::Union{Nothing, AbstractSparseMatrix{T1}}
) where T1 where T2
    (; back, D, Y, τ, dt) = ms
    (; Xadj, Z, W1, A) = mws
    
    function η(x::T, a::T) where T
        if x > a
            return x - a
        elseif -a <= x <= a
            return zero(T)
        else
            return x + a
        end
    end

    # W1 <- Y - DX
    copyto!(W1, Y)
    if Xadjsp != nothing
        mul!(W1, D, Xadjsp', -1.0, 1.0)
    else
        mul!(W1, D, Xadj', -1.0, 1.0)
    end

    if !all(isfinite.(W1))
        throw(ErrorException(
            "X became rank defficient; Try using more samples or"*
            "decreasing sparsity regularization parameter."
        ))
    end

    # X <- D'(Y - DX + Z)*dt + X
    W1 .+= Z
    mul!(Xadj, W1', D, dt, 1.0)
    W1 .-= Z

    # Z <- W * dt + Z
    axpy!(dt, W1, Z)

    # X <- sgn(X)η(|X|-τ)
    Xadj .= η.(abs.(Xadj), τ * dt) .* sign.(Xadj)
end

function gelma!(
    ms::GELMAStruct{T1, T2}, mws::GELMAWorkStruct{T1}
) where T1 where T2
    (; m, n, k, D, Y, ε, 
        max_iters, err_hist, sparsity_hist,
        elap_hist, sparse_ls_cutoff) = ms
    (; W1, Xadj, Z) = mws

    empty!(err_hist)
    empty!(sparsity_hist)
    empty!(elap_hist)

    Ynrm = norm(Y)

    # normalize
    normalize!.(eachcol(D))

    # initial guess
    randn!(Xadj)
    fill!(Z, 0.0)

    itn = 0 # tracks which iteration
    Xadjsp = nothing
    while true
        # # whether or not to use sparse LS solve
        usesp = !(length(sparsity_hist) == 0 || 
            sparsity_hist[end] > sparse_ls_cutoff)

        elap = @elapsed begin
            if usesp
                Xadjsp = sparse(Xadj)
            end
            
            gelma_iter!(ms, mws, Xadjsp)
        end
        
        # normalized error
        err = norm(W1) / Ynrm

        # add to history
        push!(err_hist, err)
        push!(sparsity_hist, count(!iszero, Xadj))
        push!(elap_hist, elap)

        # increment iteration
        itn += 1

        # determine whether to terminate
        if itn > max_iters || err < ε
            break
        end
    end
end