mutable struct MODSampleStruct{T1 <: Union{Real, Complex}, T2 <: Real}
    const back::Backend
    const m::Int
    const k::Int
    const n::Int
    const Y::StridedMatrix{T1}
    Ds::Vector{StridedMatrix{T1}}
    const dt::T2
    const τ::T2
    const ε::T2
    const max_iters::Int
    const sparse_ls_cutoff::T2
    nsamples::Int
    err_hist::Vector{Vector{T2}}
    sparsity_hist::Vector{Vector{T2}}
    elap_hist::Vector{Vector{T2}}

    function MODSampleStruct(
        back::Backend, m::Int, k::Int, n::Int,
        Y::StridedMatrix{T1}, Ds::Vector{StridedMatrix{T1}},
        dt::T2, τ::T2, ε::T2, max_iters::Int, sparse_ls_cutoff::T2,
        nsamples::Int, err_hist::Vector{Vector{T2}}, sparsity_hist::Vector{Vector{T2}}, 
         elap_hist::Vector{Vector{T2}}
    ) where T1 where T2
        @assert get_backend(Y) == back
        @assert dt > 0.0 && τ > 0.0 && ε >= 0.0 && max_iters > 0
        @assert nsamples >= 0
        @assert length(Ds) == nsamples
        for D in Ds
            @assert get_backend(D) == back
            @assert size(D) == (m, k)
        end
        @assert size(Y) == (m, n)
        @assert length(err_hist) == nsamples
        @assert length(sparsity_hist) == nsamples
        @assert length(elap_hist) == nsamples
        @assert sparse_ls_cutoff <= 1.0

        return new{T1, T2}(
            back, m, k, n, Y, Ds, dt, τ, ε, max_iters,
            sparse_ls_cutoff, nsamples, err_hist, sparsity_hist, elap_hist
        )
    end
end

struct MODSampleWorkStruct{T <: Union{Real, Complex}}
    Xadj::StridedMatrix{T}          # n × k
    Z::StridedMatrix{T}     # m × n

    # these can overlap
    W1::StridedMatrix{T}    # n × k
    W2::StridedMatrix{T}    # m × n
    A::StridedMatrix{T}     # k × k

    function MODSampleWorkStruct(
        Xadj::StridedMatrix{T},
        Z::StridedMatrix{T}, W1::StridedMatrix{T}, 
        W2::StridedMatrix{T}, A::StridedMatrix{T}
    ) where T
        n, k = size(Xadj)
        m = size(Z, 1)

        @assert backsagree(Xadj, Z, W1, W2, A)
        @assert size(W1) == (n, k)
        @assert size(W2) == (m, n)
        @assert size(Xadj) == (n, k)
        @assert size(Z) == (m, n)

        return new{T}(Xadj, Z, W1, W2, A)
    end
end

function MODSampleStruct(
    Y::AbstractMatrix{T1}, k::Int, tau::T2;
    dt::T2=0.1, eps::T2=1e-2, max_iters::Int = 1000,
    sparse_ls_cutoff::T2 = 0.05
) where T1 <: Union{Real, Complex} where T2 <: Real
    m, n = size(Y)
    back = get_backend(Y)
    Ds = Vector{StridedMatrix{T1}}(undef, 0)
    err_hist = Vector{Vector{T2}}(undef, 0)
    sparsity_hist = Vector{Vector{T2}}(undef, 0)
    elap_hist = Vector{Vector{T2}}(undef, 0)

    return MODSampleStruct(
        back, m, k, n, Y, Ds, dt, tau, eps,
        max_iters, sparse_ls_cutoff, 0, err_hist, 
        sparsity_hist, elap_hist
    )
end

function MODSampleWorkStruct(
    ms::MODSampleStruct{T1, T2}
) where T1 where T2
    (; back, k, Y) = ms
    m, n = size(Y)
    
    Xadj = adapt(back, Matrix{T1}(undef, n, k))
    Z = adapt(back, Matrix{T1}(undef, m, n))
    work = adapt(back, Vector{T1}(undef, max(n*k, m*n, k*k)))
    W1 = reshape(view(work, 1:n*k), n, k)
    W2 = reshape(view(work, 1:m*n), m, n)
    A = reshape(view(work, 1:k*k), k, k)

    return MODSampleWorkStruct(Xadj, Z, W1, W2, A)
end