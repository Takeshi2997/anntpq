module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra, Random

function makeflip()
    flip = Vector{Vector{Float64}}(undef, Const.layer[1])
    for i in 1:Const.layer[1]
        o = ones(Float64, Const.layer[1])
        o[i] *= -1.0
        flip[i] = o
    end
    return flip
end

const flip = makeflip()

function update(x::Vector{Float64})
    rng = MersenneTwister(1234)
    l = length(x)
    randamnum = rand(rng, Float64, l)
    for ix in 1:l
        x₁ = x[ix]
        z = ANN.forward(x)
        xflip = x .* flip[ix]
        zflip = ANN.forward(xflip)
        prob = exp(2.0 * real(zflip - z))
        @inbounds x[ix] = ifelse(randamnum[ix] < prob, -x₁, x₁)
    end
end

function hamiltonianS(x::Vector{Float64},
                      z::Complex{Float64}, ix::Integer)
    out = 0.0im
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    if x[ix] != x[ixnext]
        xflip = x .* flip[ix] .* flip[ixnext]
        zflip = ANN.forward(xflip)
        out  += 2.0 * exp(zflip - z) - 1.0
    else
        out += 1.0
    end
    return -Const.J * out / 4.0
end

function energyS(x::Vector{Float64})
    z = ANN.forward(x)
    sum = 0.0im
    for ix in Const.dimB+1:Const.dimB+Const.dimS
        sum += hamiltonianS(x, z, ix)
    end
    return sum
end

function hamiltonianB(x::Vector{Float64},
                      z::Complex{Float64}, iy::Integer)
    out = 0.0im
    iynext = iy%Const.dimB + 1
    if x[iy] != x[iynext]
        xflip = x .* flip[iy] .* flip[iynext]
        zflip = ANN.forward(xflip)
        out  += exp(zflip - z)
    end
    return -Const.t * out
end

function energyB(x::Vector{Float64})
    z = ANN.forward(x)
    sum = 0.0im
    for iy in 1:Const.dimB 
        sum += hamiltonianB(x, z, iy)
    end
    return sum
end

end
