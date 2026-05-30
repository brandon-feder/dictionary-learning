function mod_sample_iter!(
    ms::MODSampleStruct{T1, T2}, mws::MODSampleWorkStruct{T1},
    usesp::Bool
) where T1 where T2
    (; back, Ds, Y, τ, dt) = ms
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

    D = Ds[end]

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

    # W2 <- Y - DX
    copyto!(W2, Y)
    if usesp
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
    ms::MODSampleStruct{T1, T2}, mws::MODSampleWorkStruct{T1}
) where T1 where T2
    (; back, m, k, n, Y, Ds, ε, 
        max_iters, err_hist, sparsity_hist,
        elap_hist, sparse_ls_cutoff) = ms
    (; W2, Xadj, Z) = mws

    Ynrm = norm(Y)

    # initial guess
    randn!(Xadj)
    fill!(Z, 0.0)

    # add new vector for sample's histroy
    push!(Ds, adapt(back, Matrix{T1}(undef, m, k)))
    push!(err_hist, T2[])
    push!(sparsity_hist, T2[])
    push!(elap_hist, T2[])
    ms.nsamples += 1

    itn = 0 # tracks which iteration
    while true
        # whether or not to use sparse LS solve
        usesp = !(length(sparsity_hist[end]) == 0 || 
            sparsity_hist[end][end] > sparse_ls_cutoff)
        
        # do next iteration
        elap = @elapsed begin
            mod_sample_iter!(ms, mws, usesp)
        end
        
        # normalized error
        err = norm(W2) / Ynrm

        # add to history
        push!(err_hist[end], err)
        push!(sparsity_hist[end], count(!iszero, Xadj) / (m*n))
        push!(elap_hist[end], elap)

        # increment iteration
        itn += 1

        # determine whether to terminate
        if itn > max_iters || err < ε
            break
        end
    end
end