function foldylax!(
    fls::FoldyLaxStruct{T}, flws::FoldyLaxWorkStruct{T}
) where T
    function dsts!(
        ::Val{N}, W::StridedMatrix{Complex{T}},
        A::StridedArray{T}, B::StridedArray{T},
    ) where {N,T<:Real}
        map!(W, CartesianIndices(W)) do idx
            i, j = Tuple(idx)
            s = zero(T)
            @inbounds @simd for k in 1:N
                s += (A[k,i] - B[k,j])^2
            end
            sqrt(s)
        end

        return W
    end

    (;m, n, k, p, d, src, sct, rec, frq, τ, G) = fls
    (;Mξξ, Mξz, Mrz, Mrξadj, Mξξfac, work) = flws

    wns = reshape(frq .* (2π / 3e8), 1, 1, p)
    wst = WorkStackTrack(work)
    
    # solve for Mrz
    withwork(wst, n, k) do W
        dsts!(Val(d), W, rec, src)
        Mrz .= reshape(W, n, k, 1) .* wns
        Mrz .= exp.(1im .* Mrz) ./ (4π .* W)
    end

    # deal with edge case
    if m == 0
        for s in 1:p
            G_ = view(G, (s-1)*n+1:s*n, :)
            G_ .= view(Mrz, :, :, s)
        end
        return fls
    end 

    τ_ = reshape(τ, 1, m, 1)
    τ_adj = permutedims(τ_, (2,1,3))

    # solve for Mξξ; set diagonals to -1
    withwork(wst, m, m) do W
        dsts!(Val(d), W, sct, sct)
        Mξξ .= reshape(W, m, m, 1) .* wns
        Mξξ .= exp.(1im .* Mξξ) ./ (4π .* W) .* τ_
        view(reshape(Mξξ, m^2, :), 1:(m+1):m*m, :) .= -1
    end

    # solve for Mξz
    withwork(wst, m, k) do W
        dsts!(Val(d), W, sct, src)
        Mξz .= reshape(W, m, k, 1) .* wns
        Mξz .= exp.(1im .* Mξz) ./ (4π .* W)
    end

    # solve for Mrξadj, adjoint of Mrξ
    withwork(wst, m, n) do W
        dsts!(Val(d), W, sct, rec)
        Mrξadj .= W .* wns
        Mrξadj .= exp.(1im .* Mrξadj) ./ (4π .* W) .* τ_adj
    end
    
    # compute factorizations
    Mξξfac[] = [qr!(view(Mξξ, :, :, s)) for s in 1:p]

    for s in 1:p
        # get current slice
        Mξz_ = view(Mξz, :, :, s)
        Mξξ_ = Mξξfac[][s]
        Mrz_ = view(Mrz, :, :, s)
        Mrξadj_ = view(Mrξadj, :, :, s)
        G_ = view(G, (s-1)*n+1:s*n, :)

        ldiv!(adjoint(Mξξ_), Mrξadj_)
        Mrξadj_ = adjoint(Mrξadj_)

        G_ .= Mrz_ .- Mrξadj_ * Mξz_
    end

    return fls
end

function foldylax_update!(
    fls::FoldyLaxStruct{T}, flws::FoldyLaxWorkStruct{T}
) where T
    function dsts!(
        ::Val{N}, W::StridedMatrix{Complex{T}},
        A::StridedArray{T}, B::StridedArray{T},
    ) where {N,T<:Real}
        map!(W, CartesianIndices(W)) do idx
            i, j = Tuple(idx)
            s = zero(T)
            @inbounds @simd for k in 1:N
                s += (A[k,i] - B[k,j])^2
            end
            sqrt(s)
        end

        return W
    end

    (;m, n, k, p, d, src, rec, sct, frq, τ, G) = fls
    (;Mξξ, Mξz, Mrz, Mrξadj, Mξξfac, work) = flws

    wns = reshape(frq .* (2π / 3e8), 1, 1, p)
    wst = WorkStackTrack(work)
    
    # solve for Mrz
    withwork(wst, n, k) do W
        dsts!(Val(d), W, rec, src)
        Mrz .= reshape(W, n, k, 1) .* wns
        Mrz .= exp.(1im .* Mrz) ./ (4π .* W)
    end

    # deal with edge case
    if m == 0
        for s in 1:p
            G_ = view(G, (s-1)*n+1:s*n, :)
            G_ .= view(Mrz, :, :, s)
        end
        return fls
    end 

    τ_ = reshape(τ, 1, m, 1)
    τ_adj = permutedims(τ_, (2,1,3))

    # solve for Mξz
    withwork(wst, m, k) do W
        dsts!(Val(d), W, sct, src)
        Mξz .= reshape(W, m, k, 1) .* wns
        Mξz .= exp.(1im .* Mξz) ./ (4π .* W)
    end

    # solve for Mrξadj, adjoint of Mrξ
    withwork(wst, m, n) do W
        dsts!(Val(d), W, sct, rec)
        Mrξadj .= W .* wns
        Mrξadj .= exp.(1im .* Mrξadj) ./ (4π .* W) .* τ_adj
    end

    for s in 1:p
        # get current slice
        Mξz_ = view(Mξz, :, :, s)
        Mξξ_ = Mξξfac[][s]
        Mrz_ = view(Mrz, :, :, s)
        Mrξadj_ = view(Mrξadj, :, :, s)
        G_ = view(G, (s-1)*n+1:s*n, :)

        ldiv!(adjoint(Mξξ_), Mrξadj_)
        Mrξadj_ = adjoint(Mrξadj_)

        G_ .= Mrz_ .- Mrξadj_ * Mξz_
    end

    return fls
end