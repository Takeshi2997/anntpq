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

function update(s::Vector{Float32}, n::Vector{Float32})

    rng = MersenneTwister(1234)
    ls = length(s)
    ln = length(n)
    randomnum = rand(rng, Float32, ls + ln)

    #Update System
    prob = exp.(-4f0 .* s .* real.(ANN.forward(n)))
    s .*= ifelse.(randomnum[1:ls] .< prob, -1f0, 1f0)

    #Update Bath
    for iy in 1:ln
        n₁ = n[iy]
        z = ANN.forward(n)
        nflip = n .* flip[iy]
        zflip = ANN.forward(nflip)
        prob = exp(2f0 * real(transpose(s) * (zflip - z)))
        @inbounds n[iy] = ifelse(randomnum[ls + iy] < prob, -n₁, n₁)
    end
end

function hamiltonianS(s::Vector{Float32}, z::Vector{Complex{Float32}})

    out = 0f0im
    if s[1] != s[2]
        out  += 2f0 * exp(-2f0 * transpose(s) * z) - 1f0
    else
        out += 1f0
    end

    return -Const.J * out / 4f0
end

function energyS(s::Vector{Float32}, n::Vector{Float32})

    z = ANN.forward(n)
    sum = 0f0im
    @simd for ix in 1:Const.dimS-1
        sum += hamiltonianS(s[ix:ix+1], z[ix:ix+1])
    end
    sum += hamiltonianS(s[end:1-end:1], z[end:1-end:1])
 
    return sum
end

function hamiltonianB(s::Vector{Float32}, n::Vector{Float32},
                      z::Complex{Float32}, iy::Integer)

    out = 0f0im
    iynext = iy%Const.dimB + 1
    if n[iy] != n[iynext]
        nflip = n .* flip[iy] .* flip[iynext]
        zflip = transpose(s) * ANN.forward(nflip)
        out  += exp(zflip - z)
    end

    return -Const.t * out
end

function energyB(s::Vector{Float32}, n::Vector{Float32})

    z = transpose(s) * ANN.forward(n)
    sum = 0.0f0im
    @simd for iy in 1:Const.dimB
        sum += hamiltonianB(s, n, z, iy)
    end

    return sum
end

end
