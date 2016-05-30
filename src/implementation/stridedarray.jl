# implementation/stridedarray.jl
#
# High-level implementation of tensor operations for StridedArray from Julia
# Base Library. Checks dimensions and converts to StridedData before passing
# to low-level (recursive) function.

"""`add!(α, A, conjA, β, C, indCinA)`

Implements `C = β*C+α*permute(op(A))` where `A` is permuted according to `indCinA` and `op` is `conj` if `conjA=Val{:C}` or the identity map if `conjA=Val{:N}`. The indexable collection `indCinA` contains as nth entry the dimension of `A` associated with the nth dimension of `C`.
"""
function add!{CA}(α, A::StridedArray, ::Type{Val{CA}}, β, C::StridedArray, indCinA)
    _size(A, indCinA) == _size(C) || throw(DimensionMismatch())

    dims, stridesA, stridesC, minstrides = add_strides(size(C), _permute(_strides(A),indCinA), _strides(C))
    dataA = StridedData(A, stridesA, Val{CA})
    offsetA = 0
    dataC = StridedData(C, stridesC)
    offsetC = 0

    if α == 0
        β == 1 || _scale!(dataC,β,dims)
    elseif α == 1 && β == 0
        add_rec!(Val{1}, dataA, Val{0}, dataC, dims, offsetA, offsetC, minstrides)
    elseif α == 1 && β == 1
        add_rec!(Val{1}, dataA, Val{1}, dataC, dims, offsetA, offsetC, minstrides)
    elseif β == 0
        add_rec!(α, dataA, Val{0}, dataC, dims, offsetA, offsetC, minstrides)
    elseif β == 1
        add_rec!(α, dataA, Val{1}, dataC, dims, offsetA, offsetC, minstrides)
    else
        add_rec!(α, dataA, β, dataC, dims, offsetA, offsetC, minstrides)
    end
    return C
end

"""`trace!(α, A, conjA, β, C, indCinA, cindA1, cindA2)`

Implements `C = β*C+α*partialtrace(op(A))` where `A` is permuted and partially traced, according to `indCinA`, `cindA1` and `cindA2`, and `op` is `conj` if `conjA=Val{:C}` or the identity map if `conjA=Val{:N}`. The indexable collection `indCinA` contains as nth entry the dimension of `A` associated with the nth dimension of `C`. The partial trace is performed by contracting dimension `cindA1[i]` of `A` with dimension `cindA2[i]` of `A` for all `i in 1:length(cindA1)`.
"""
function trace!{CA}(α, A::StridedArray, ::Type{Val{CA}}, β, C::StridedArray, indCinA, cindA1, cindA2)
    NC = ndims(C)
    NA = ndims(A)

    _size(A, indCinA) == _size(C) || throw(DimensionMismatch(""))
    (NC + 2*length(cindA1) == NA &&
     _size(A, cindA1) == _size(A, cindA2)) || throw(DimensionMismatch(""))

    pA = vcat(indCinA, cindA1, cindA2)
    dims, stridesA, stridesC, minstrides = trace_strides(_permute(size(A),pA), _permute(_strides(A),pA), _strides(C))
    dataA = StridedData(A, stridesA, Val{CA})
    offsetA = 0
    dataC = StridedData(C, stridesC)
    offsetC = 0

    if α == 0
        β == 1 || _scale!(dataC, β, dims)
    elseif α == 1 && β == 0
        trace_rec!(Val{1}, dataA, Val{0}, dataC, dims, offsetA, offsetC, minstrides)
    elseif α == 1 && β == 1
        trace_rec!(Val{1}, dataA, Val{1}, dataC, dims, offsetA, offsetC, minstrides)
    elseif β == 0
        trace_rec!(α, dataA, Val{0}, dataC, dims, offsetA, offsetC, minstrides)
    elseif β == 1
        trace_rec!(α, dataA, Val{1}, dataC, dims, offsetA, offsetC, minstrides)
    else
        trace_rec!(α, dataA, β, dataC, dims, offsetA, offsetC, minstrides)
    end
    return C
end

"""`contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, [method])`

Implements `C = β*C+α*contract(op(A),op(B))` where `A` and `B` are contracted according to `oindA`, `cindA`, `oindB`, `cindB` and `indCinoAB`. The operation `op` acts as `conj` if `conjA` or `conjB` equal `Val{:C}` or as the identity map if `conjA` (`conjB`) equal `Val{:N}`. The dimension `cindA[i]` of `A` is contracted with dimension `cindB[i]` of `B`. The `n`th dimension of C is associated with an uncontracted (open) dimension of `A` or `B` according to `indCinoAB[n] < NoA ? oindA[indCinoAB[n]] : oindB[indCinoAB[n]-NoA]` with `NoA=length(oindA)` the number of open dimensions of `A`.

The optional argument `method` specifies whether the contraction is performed using BLAS matrix multiplication by specifying `Val{:BLAS}` (default), or using a native algorithm by specifying `Val{:native}`. The native algorithm does not copy the data but is typically slower.
"""
function contract!{CA,CB,TC<:Base.LinAlg.BlasFloat}(α, A::StridedArray, ::Type{Val{CA}}, B::StridedArray, ::Type{Val{CB}}, β, C::StridedArray{TC},
                                                    oindA, cindA, oindB, cindB, indCinoAB, ::Type{Val{:BLAS}}=Val{:BLAS})
    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)
    TA = eltype(A)
    TB = eltype(B)

    # dimension checking
    cdims = _size(A, cindA)
    cdimsB = _size(B, cindB)
    odimsA = _size(A, oindA)
    odimsB = _size(B, oindB)
    odimsAB = tuple(odimsA..., odimsB...)

    _iselequal(cdims, cdimsB) || throw(DimensionMismatch())

    dimC = size(C)
    for i in 1:length(indCinoAB)
        dimC[i] == odimsAB[indCinoAB[i]] || throw(DimensionMismatch())
    end

    olengthA = prod(odimsA)
    olengthB = prod(odimsB)
    clength = prod(cdims)

    # permute A
    if CA == :C
        conjA = 'C'
        if isa(A, Array{TC}) && _iselequal(cindA, oindA, 1:NA)
            Amat = reshape(A, (clength, olengthA))
        else
            Apermuted = Array{TC}(tuple(cdims..., odimsA...))
            # tensorcopy!(A, 1:NA, Apermuted, pA)
            add!(1, A, Val{:N}, 0, Apermuted, tuple(cindA..., oindA...))
            Amat = reshape(Apermuted, (clength, olengthA))
        end
    else
        conjA = 'N'
        if isa(A, Array{TC}) && _iselequal(oindA, cindA, 1:NA)
            Amat = reshape(A, (olengthA, clength))
        elseif isa(A, Array{TC}) && _iselequal(cindA, oindA, 1:NA)
            conjA = 'T'
            Amat = reshape(A, (clength, olengthA))
        else
            Apermuted = Array{TC}(tuple(odimsA..., cdims...))
            # tensorcopy!(A, 1:NA, Apermuted, pA)
            add!(1, A, Val{:N}, 0, Apermuted, tuple(oindA..., cindA...))
            Amat = reshape(Apermuted, (olengthA, clength))
        end
    end

    # permute B
    if CB == :C
        conjB = 'C'
        if isa(B, Array{TC}) && _iselequal(oindB, cindB, 1:NB)
            Bmat = reshape(B, (olengthB, clength))
        else
            Bpermuted = Array{TC}(tuple(odimsB..., cdims...))
            # tensorcopy!(B, 1:NB, Bpermuted, pB)
            add!(1, B, Val{:N}, 0, Bpermuted, tuple(oindB..., cindB...))
            Bmat = reshape(Bpermuted, (olengthB, clength))
        end
    else
        conjB = 'N'
        if  isa(B, Array{TC}) && _iselequal(cindB, oindB, 1:NB)
            Bmat = reshape(B, (clength, olengthB))
        elseif isa(B, Array{TC}) && _iselequal(oindB, cindB, 1:NB)
            conjB = 'T'
            Bmat = reshape(B, (olengthB, clength))
        else
            Bpermuted = Array{TC}(tuple(cdims..., odimsB...))
            # tensorcopy!(B, 1:NB, Bpermuted, pB)
            add!(1, B, Val{:N}, 0, Bpermuted, tuple(cindB..., oindB...))
            Bmat = reshape(Bpermuted, (clength, olengthB))
        end
    end

    # calculate C
    if isa(C, Array) && _iselequal(indCinoAB, 1:NC)
        Cmat = reshape(C, (olengthA, olengthB))
        BLAS.gemm!(conjA, conjB, TC(α), Amat, Bmat, TC(β), Cmat)
    else
        Cmat = Array{TC}(olengthA, olengthB)
        BLAS.gemm!(conjA, conjB, TC(1), Amat, Bmat, TC(0), Cmat)
        # tensoradd!(α, reshape(Cmat, tuple(odimsA..., odimsB...)), pC, β, C, 1:NC)
        add!(α, reshape(Cmat, tuple(odimsA..., odimsB...)), Val{:N}, β, C, indCinoAB)
    end
    return C
end

function contract!{CA,CB}(α, A::StridedArray, ::Type{Val{CA}}, B::StridedArray, ::Type{Val{CB}}, β, C::StridedArray,
                          oindA, cindA, oindB, cindB, indCinoAB, ::Type{Val{:native}}=Val{:native})
    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)

    # dimension checking
    cdimsA = _size(A, cindA)
    cdimsB = _size(B, cindB)
    odimsA = _size(A, oindA)
    odimsB = _size(B, oindB)
    odimsAB = tuple(odimsA..., odimsB...)

    # Perform contraction
    pA = vcat(oindA, cindA)
    pB = vcat(oindB, cindB)
    sA = _permute(_strides(A), pA)
    sB = _permute(_strides(B), pB)
    sC = _permute(_strides(C), invperm(indCinoAB))

    dimsA = _permute(size(A), pA)
    dimsB = _permute(size(B), pB)

    dims, stridesA, stridesB, stridesC, minstrides = contract_strides(dimsA, dimsB, sA, sB, sC)
    offsetA = offsetB = offsetC = 0
    dataA = StridedData(A, stridesA, Val{CA})
    dataB = StridedData(B, stridesB, Val{CB})
    dataC = StridedData(C, stridesC)

    # contract via recursive divide and conquer
    if α == 0
        β == 1 || _scale!(dataC, β, dims)
    elseif α == 1 && β == 0
        contract_rec!(Val{1}, dataA, dataB, Val{0}, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif α == 1 && β == 1
        contract_rec!(Val{1}, dataA, dataB, Val{1}, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif β == 0
        contract_rec!(α, dataA, dataB, Val{0}, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif β == 1
        contract_rec!(α, dataA, dataB, Val{1}, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    else
        contract_rec!(α, dataA, dataB, β, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    end
    return C
end
