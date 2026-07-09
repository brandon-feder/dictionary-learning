function get_white(
    Y::StridedMatrix{T}, k::Int, s::Int
) where T
    m, n = size(Y)
    Yh = copy(Y')
    qrYadj = qr(Yh)
    R = copy(qrYadj.R')

    svdR = svd(R)
    U = svdR.U
    svls = svdR.S * sqrt(k/(n*s))

    Winv = U * Diagonal(svls) * U'

    if k < m
        svls[k+1:end] .= 0.0
        svls[1:k] .^= -1
    else
        svls .^= -1
    end

    W = U * Diagonal(svls) * U'

    return W, Winv
end

function get_white(D::StridedMatrix{T}) where T
    return get_white(D, size(D, 2), 1)
end

function get_white_fast(
    Y::StridedMatrix{T}, k::Int, s::Int
) where T
    m, n = size(Y)
    A = Y*Y'

    eigA = eigen(Hermitian(A))
    U = eigA.vectors
    evls = eigA.values

    evls_rev = view(evls, length(evls):-1:1)
    evls_rev[min(m,k)+1:end] .= 0.0
    evls_rev_ = view(evls_rev, 1:min(m,k))

    evls_rev_[1:min(m,k)] .= sqrt.(evls_rev_) * sqrt(k/(n*s))

    Winv = U * Diagonal(evls) * U'

    evls_rev_ .^= -1

    W = U * Diagonal(evls) * U'

    return W, Winv
end

function get_white_fast(D::StridedMatrix{T}) where T
    return get_white_fast(D, size(D, 2), 1)
end