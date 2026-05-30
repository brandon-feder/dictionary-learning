abstract type AbstractSSDLSubRecStruct{T} end

struct SSDLSubRecStruct{T} <: AbstractSSDLSubRecStruct{T}
    back::Backend
    m::Int
    n::Int
    k::Int
    s::Int
    t::Int
    Y::StridedMatrix{T} # observations
    S::StridedArray{T} # subspaces
    
    function SSDLSubRecStruct(
        back::Backend, m::Int, n::Int, k::Int, s::Int,
        t::Int, Y::StridedMatrix{T}, S::StridedArray{T}
    ) where T
        @assert back == get_backend(Y)
        @assert backsagree(Y, S)
        @assert size(Y) == (m, n)
        @assert size(S) == (m, s, t)
        @assert k > 0 && t > 0

        return new{T}(back, m, n, k, s, t, Y, S)
    end
end

struct SSDLFakeSubRecStruct{T} <: AbstractSSDLSubRecStruct{T}
    back::Backend
    m::Int
    n::Int
    k::Int
    s::Int
    t::Int
    D::StridedMatrix{T} # dictioanry
    X::AbstractSparseMatrix{T} # sample support
    S::StridedArray{T} # subspaces

    function SSDLFakeSubRecStruct(
        back::Backend, m::Int, n::Int, k::Int, s::Int,
        t::Int, D::StridedMatrix{T}, 
        X::AbstractSparseMatrix{T}, S::StridedArray{T}
    ) where T
        @assert get_backend(D) == back
        @assert get_backend(X) == CPU()
        @assert backsagree(D, S)
        @assert size(D) == (m, k)
        @assert size(X) == (k, n)
        @assert size(S) == (m, s, t)
        @assert k > 0 && t > 0

        return new{T}(back, m, n, k, s, t, D, X, S)
    end
end

function SSDLSubRecStruct(
    Y::StridedMatrix{T}, k::Int, s::Int, t::Int
) where T
    back = get_backend(Y)
    m, n = size(Y)

    S = adapt(back, Array{T}(undef, m, s, t))

    return SSDLSubRecStruct(back, m, n, k, s, t, Y, S)
end

function SSDLFakeSubRecStruct(
    D::StridedMatrix{T}, 
    X::AbstractSparseMatrix{T},
    s::Int, t::Int
) where T
    back = get_backend(D)
    m, k = size(D)
    n = size(X, 2)

    S = adapt(back, Array{T}(undef, m, s, t))
    
    return SSDLFakeSubRecStruct(back, m, n, k, s, t, D, X, S)
end

function ssdl_subrec!(
    ssrs::SSDLSubRecStruct{T}
) where T
    (; back, m, n, t, s, Y, S) = ssrs

    # allocate workspace
    Σ = adapt(back, Matrix{T}(undef, m, m))
    w = adapt(back, Matrix{T}(undef, 1, n))
    W = adapt(back, Matrix{T}(undef, m, n))
    Z = adapt(back, Matrix{T}(undef, m, m))
    tau = adapt(back, Vector{T}(undef, s))
    
    # Σ <- 1/n YY'
    mul!(Σ, Y, Y', 1/n, 0.0)

    Σnrmsq = norm(Σ)^2

    for j in 1:t
        yj = view(Y, :, j)
        Sj = view(S, :, :, j)

        # compute weights for j and i=1, ..., n
        mul!(w, yj', Y)

        # scale Y by squared weights
        W .= Y .* abs2.(w)

        # compute correlation projection
        mul!(Z, W, Y', 1/n, 0.0)

        Z .-= (dot(Z, Σ)/Σnrmsq) * Σ

        # compute leading s eigenvectors
        eig = eigen(Hermitian(Z))
        eigvindx = sortperm(adapt(CPU(), eig.values), by=λ->-λ)
        
        Sj .= view(eig.vectors, :, eigvindx[1:s])

        # orthonormalize
        LAPACK.geqrf!(Sj, tau)
        LAPACK.orgqr!(Sj, tau)
    end
end

function ssdl_subrec!(
    ssrs::SSDLFakeSubRecStruct{T}
) where T
    (; back, m, n, k, t, s, D, X, S) = ssrs
    
    # allocate workspace
    Σ = adapt(back, Matrix{T}(undef, m, m))
    w1 = adapt(back, Matrix{T}(undef, 1, k))
    w2 = adapt(back, Matrix{T}(undef, 1, n))
    W = adapt(back, Matrix{T}(undef, m, k))
    Z = adapt(back, Matrix{T}(undef, m, m))
    X_back = adapt(back, X)
    tau = adapt(back, Vector{T}(undef, s))

    # Σ <- 1/n YY'
    Σ .= 1/n * D*(X_back*X_back')*D'

    Σnrmsq = norm(Σ)^2

    for j in 1:t
        xj = adapt(back, X[:,j])
        yj = D*xj
        Sj = view(S, :, :, j)

        # compute weights for j and i=1, ..., n
        mul!(w1, yj', D)
        mul!(w2, w1, X_back)

        # # scale Y by squared weights
        # W .= D*X .* abs2.(w)
        w2diag = adapt(CUDABackend(), T.(spdiagm(abs2.(vec(w2)))))
        mul!(W, D, 1/n * X_back * w2diag * X_back')
        mul!(Z, W, D')
        Z .-= (dot(Z, Σ)/Σnrmsq) * Σ

        # compute leading s eigenvectors
        eig = eigen(Hermitian(Z))
        eigvindx = sortperm(adapt(CPU(), eig.values), by=λ->-λ)
        
        Sj .= view(eig.vectors, :, eigvindx[1:s])

        # orthonormalize
        LAPACK.geqrf!(Sj, tau)
        LAPACK.orgqr!(Sj, tau)
    end
end

function ssdl_subrec_true!(
    stsrs::SSDLFakeSubRecStruct{T}
) where T
    (; back, t, s, D, X, S) = stsrs

    tau = adapt(back, Vector{T}(undef, s))

    for i in 1:t
        Si = view(S, :, :, i)
        xi = view(X, :, i)

        # find support of vector
        sup, _ = findnz(xi)

        # get basis for subspace
        copyto!(Si, view(D, :, sup))

        # orthonormalize
        LAPACK.geqrf!(Si, tau)
        LAPACK.orgqr!(Si, tau)
    end
end