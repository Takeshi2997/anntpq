module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra, Distributed, Random

function makeflip()

    flip = Vector{Vector{Float32}}(undef, Const.layer[1])
    for i in 1:Const.layer[1]
        o = ones(Float32, Const.layer[1])
        o[i] *= -1f0
        flip[i] = o
    end
    return flip
end

const flip = makeflip()

function update(x::Vector{Float32}, α::Vector{Float32})

    rng = MersenneTwister(1234)
    l = length(x)
    randamnum = rand(rng, Float32, 2*l)

    for iα in 1:l
        α₁ = α[iα]
        z = ANN.forward(α) + dot(x, ANN.interaction(α))
        αflip = α .* flip[iα]
        zflip = ANN.forward(αflip) + dot(x, ANN.interaction(αflip))
        prob = exp(2f0 * real(zflip - z))
        @inbounds α[iα] = ifelse(randamnum[iα] < prob, -α₁, α₁)
    end

    prob = exp.(-2f0 .* 2f0 .* real.(x .* ANN.interaction(α)))
    for ix in 1:l
        x₁ = x[ix]
        @inbounds x[ix] = ifelse(randamnum[l+ix] < prob[ix], -x₁, x₁)
    end
end

function hamiltonianS(x::Vector{Float32}, 
                      z::Array{Complex{Float32}}, ix::Integer)

    out = 0f0im
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    if x[ix] != x[ixnext]
        xflip = x .* flip[ix] .* flip[ixnext]
        out  += 2f0 * exp(dot(xflip - x, z)) - 1f0
    else
        out += 1f0
    end

    return -Const.J * out / 4f0
end

function energyS(x::Vector{Float32}, α::Vector{Float32})

    z = ANN.interaction(α)
    sum = 0f0im
    @simd for ix in Const.dimB+1:Const.dimB+Const.dimS
        sum += hamiltonianS(x, z, ix)
    end

    return sum
end

function hamiltonianB(x::Vector{Float32}, 
                      z::Array{Complex{Float32}}, iy::Integer)

    out = 0f0im
    iynext = iy%Const.dimB + 1
    if x[iy] != x[iynext]
        xflip = x .* flip[iy] .* flip[iynext]
        out  += exp(dot(xflip - x, z))
    end

    return -Const.t * out
end

function energyB(x::Vector{Float32}, α::Vector{Float32})

    z = ANN.interaction(α)
    sum = 0.0f0im
    @simd for iy in 1:Const.dimB 
        sum += hamiltonianB(x, z, iy)
    end

    return sum
end

end
