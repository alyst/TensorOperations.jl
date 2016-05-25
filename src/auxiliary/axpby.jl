# auxiliary/axpby.jl
#
# Simple wrapper for the operation of computing α * x + β * y
# to remove any overhead of multiplication
# by one or addition by zero.

axpby(α::Type{Val{1}}, x, β::Type{Val{1}}, y) = x+y
axpby(α::Type{Val{0}}, x, β::Type{Val{1}}, y) = y
axpby(α::Type{Val{1}}, x, β::Type{Val{0}}, y) = x
axpby(α::Type{Val{0}}, x, β::Type{Val{0}}, y) = zero(y)

axpby(α::Type{Val{1}}, x, β,               y) = x+β*y
axpby(α::Type{Val{0}}, x, β,               y) = β*y
axpby(α,               x, β::Type{Val{0}}, y) = α*x
axpby(α,               x, β::Type{Val{1}}, y) = α*x+y

axpby(α,               x, β,               y) = α*x+β*y
