function mod_sample_init_iter!(
    ms::DictionaryLearning.MODSampleStruct{T1, T2}, mws::DictionaryLearning.MODSampleWorkStruct{T1},
    usesp::Bool
) where T1 where T2
    (; back, D, Y) = ms
    (; Xadj, Z, W1, A) = mws

    # We would like to solve the LS problem
    #           X'D' = Y'
    # for D. We do this differently depending
    # on the value of `usesp` indicating whether
    # Xadjsp is sparse enough or not.
    #   (case 1) If `usesp == false` then
    #       we can solve directly using ldiv!
    #   (case 2) If `usesp == true` then we
    #       solve instead the problem
    #           XX'D' = XY'
    #       for D since this is easier on GPUs.

    if usesp
        Xadjsp = sparse(Xadj)

        # densify
        if back == CPU()
            A .= Matrix(Xadjsp' * Xadjsp)
        else
            A .= CuMatrix(Xadjsp' * Xadjsp)
        end

        # essentially do the same thing as ldiv!
        # but avoid scalar indexing.
        mul!(D, Y, Xadjsp)
        Aqr = qr!(A)
        rmul!(D, Aqr.Q)
        rdiv!(D, UpperTriangular(Aqr.factors)')
    else
        copyto!(W1, Xadj)
        ldiv!(adjoint(D), qr!(W1), adjoint(Y))
    end

    # normalize columns of D
    D ./= sqrt.(sum(abs2, D, dims=1))

    if usesp
        return Xadjsp
    else
        return nothing
    end
end

function mod_sample_iter!(
    ms::DictionaryLearning.MODSampleStruct{T1, T2}, mws::DictionaryLearning.MODSampleWorkStruct{T1},
    Xadjsp::Union{Nothing, AbstractSparseMatrix{T1}}
) where T1 where T2
    (; back, D, Y, τ, dt) = ms
    (; Xadj, Z, W1, W2, A) = mws
    
    function η(x::T, a::T) where T
        if x > a
            return x - a
        elseif -a <= x <= a
            return zero(T)
        else
            return x + a
        end
    end

    # W2 <- Y - DX
    copyto!(W2, Y)
    if Xadjsp != nothing
        mul!(W2, D, Xadjsp', -1.0, 1.0)
    else
        mul!(W2, D, Xadj', -1.0, 1.0)
    end

    if !all(isfinite.(W2))
        throw(ErrorException(
            "X became rank defficient; Try using more samples or"*
            "decreasing sparsity regularization parameter."
        ))
    end

    # X <- D'(Y - DX + Z)*dt + X
    W2 .+= Z
    mul!(Xadj, W2', D, dt, 1.0)
    W2 .-= Z

    # Z <- W * dt + Z
    axpy!(dt, W2, Z)

    # X <- sgn(X)η(|X|-τ)
    Xadj .= η.(abs.(Xadj), τ * dt) .* sign.(Xadj)
end

function mod_sample!(
    ms::DictionaryLearning.MODSampleStruct{T1, T2}, mws::DictionaryLearning.MODSampleWorkStruct{T1},
    D0::Union{Nothing, StridedMatrix{T1}}=nothing, X0adj::Union{Nothing, StridedMatrix{T1}}=nothing
) where T1 where T2
    (; back, m, k, n, Y, D, ε, 
        max_iters, err_hist, sparsity_hist,
        elap_hist, sparse_ls_cutoff) = ms
    (; W2, Xadj, Z) = mws

    if D0 != nothing
        @assert size(D0) == (m, k)
        @assert X0adj != nothing
        @assert size(X0adj) == (n, k)
    end

    empty!(err_hist)
    empty!(sparsity_hist)
    empty!(elap_hist)

    Ynrm = norm(Y)
    
    # do first initialization
    elap = @elapsed begin
        if D0 == nothing
            randn!(Xadj)
            mod_sample_init_iter!(ms, mws, false)
        else
            copyto!(Xadj, X0adj)
            copyto!(D, D0)
        end
    end

    # initialize Z
    fill!(Z, 0.0)

    itn = 0 # tracks which iteration
    Xadjsp = nothing
    while true
        elap += @elapsed begin
            mod_sample_iter!(ms, mws, Xadjsp)
        end
        
        # normalized error
        err = norm(W2) / Ynrm

        # add to history
        push!(err_hist, err)
        push!(sparsity_hist, count(!iszero, Xadj) / (m*n))
        push!(elap_hist, elap)

        # increment iteration
        itn += 1

        # determine whether to terminate
        if itn > max_iters || err < ε
            break
        end
        
        # whether or not to use sparse LS solve
        usesp = !(length(sparsity_hist) == 0 || 
            sparsity_hist[end] > sparse_ls_cutoff)
        
        # reinitialize before next iteration
        elap = @elapsed begin
            Xadjsp = mod_sample_init_iter!(ms, mws, usesp)
        end
    end
    
    if back == CUDABackend() && Xadjsp != nothing
        CUDA.unsafe_free!(Xadjsp)
    end
end