# auxiliary/stridedarray.jl
#
# Simple auxiliary methods to interface with StridedArray from Julia Base.


"""`numind(A)`

Returns the number of indices of a tensor-like object `A`, i.e. for a multidimensional array (`<:AbstractArray`) we have `numind(A) = ndims(A)`. Also works in type domain.
"""
numind(A::AbstractArray) = ndims(A)
numind{T<:AbstractArray}(::Type{T}) = ndims(T)

"""`similar_from_indices(T, indices, A, conjA=Val{:N})`

Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes corresponding to a selection of those of `op(A)`, where the selection is specified by `indices` (which contains integer between `1` and `numind(A)`) and `op` is `conj` if `conjA=Val{:C}` or does nothing if `conjA=Val{:N}` (default).
"""
function similar_from_indices{T,CA}(::Type{T}, indices, A::StridedArray, ::Type{Val{CA}}=Val{:N})
    dims = size(A)
    return similar(A,T,dims[indices])
end

"""`similar_from_indices(T, indices, A, B, conjA=Val{:N}, conjB={:N})`

Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes corresponding to a selection of those of `op(A)` and `op(B)` concatenated, where the selection is specified by `indices` (which contains integers between `1` and `numind(A)+numind(B)` and `op` is `conj` if `conjA` or `conjB` equal `Val{:C}` or does nothing if `conjA` or `conjB` equal `Val{:N}` (default).
"""
function similar_from_indices{T,CA,CB}(::Type{T}, indices, A::StridedArray, B::StridedArray, ::Type{Val{CA}}=Val{:N}, ::Type{Val{CB}}=Val{:N})
    dims = tuple(size(A)...,size(B)...)
    return similar(A,T,dims[indices])
end

"""`scalar(C)`

Returns the single element of a tensor-like object with zero dimensions, i.e. if `numind(C)==0`.
"""
scalar(C::StridedArray) = numind(C)==0 ? C[1] : throw(DimensionMismatch())

"""`_iselequal(A, B)`

Checks if the collection `A` contains the same elements and in the same order as `B`.
"""
function _iselequal(A, B)
    eltype(A) == eltype(B) || return false
    length(A) == length(B) || return false
    for (a, b) in zip(A, B)
        a == b || return false
    end
    return true
end

"""`_iselequal(A1, A2, B)`

Checks if the combined collection of `A1` and `A2` contains the same elements and in the same order as `B`.
"""
function _iselequal(A1, A2, B)
    eltype(A1) == eltype(A2) == eltype(B) || return false
    (length(A1) + length(A2) == length(B)) || return false
    for (a, b) in zip(chain(A1, A2), B)
        a == b || return false
    end
    return true
end

"""`_size(A, [dims])`

`size()` wrapper that always returns tuple.

Returns empty tuple if `dims` is specified and empty.
"""
_size(A) = size(A)
_size(A, dim::Number) = (size(A, dim), )
_size(A, dims::Tuple{}) = tuple()
_size(A, dims::Tuple{Int}) = (size(A, dims[1]), )
_size{N}(A, dims::NTuple{N,Int}) = size(A, dims...)::NTuple{N,Int}
_size(A, dims) = isempty(dims) ? tuple() :
                 (length(dims) == 1 ? (size(A, dims[1]), ) : size(A, dims[1], dims[2], dims[3:end]...))
