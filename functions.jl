module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra, Distributed, Random

struct Flip{T}
    flip::T
end

function Flip()
    flip = Vector{Diagonal{Float32}}(undef, Const.layer[1])
    for i in 1:Const.layer[1]
        o = ones(Float32, Const.layer[1])
        o[i] *= -1f0
        flip[i] = Diagonal(o)
    end
    Flip(flip)
end

a = Flip()

function update(x::Vector{Float32})
    rng = MersenneTwister(1234)
    randamnum = rand(rng, Float32, length(x))
    for ix in 1:length(x)
        x₁ = x[ix]
        z = ANN.forward(x)
        xflip = a.flip[ix] * x
        zflip = ANN.forward(xflip)
        prob = exp(2.0f0 * real(zflip - z))
        @inbounds x[ix] = ifelse(randamnum[ix] < prob, -x₁, x₁)
    end
end

function hamiltonianS(x::Vector{Float32}, z::Complex{Float32}, ix::Integer)
    out = 0f0im
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    if x[ix] != x[ixnext]
        xflip = a.flip[ix] * a.flip[ixnext] * x
        zflip = ANN.forward(xflip)
        out  += 2f0 * exp(zflip - z) - 1f0
    else
        out += 1f0
    end
    return -Const.J * out / 4f0
end

function energyS(x::Vector{Float32})
    z = ANN.forward(x)
    sum = 0f0im
    for ix in Const.dimB+1:Const.dimB+Const.dimS
        sum += hamiltonianS(x, z, ix)
    end
    return sum
end

function hamiltonianB(x::Vector{Float32}, z::Complex{Float32}, iy::Integer)
    out = 0f0im
    iynext = iy%Const.dimB + 1
    if x[iy] != x[iynext]
        xflip = a.flip[iy] * a.flip[iynext] * x
        zflip = ANN.forward(xflip)
        out  += exp(zflip - z)
    end
    return -Const.t * out
end

function energyB(x::Vector{Float32})
    z = ANN.forward(x)
    sum = 0.0f0im
    for iy in 1:Const.dimB 
        sum += hamiltonianB(x, z, iy)
    end
    return sum
end
end
