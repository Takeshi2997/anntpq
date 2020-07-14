module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra, Distributed, Random

function makeflip1()

    flip = Vector{Vector{Float32}}(undef, Const.layer1[1])
    for i in 1:Const.layer1[1]
        o = ones(Float32, Const.layer1[1])
        o[i] *= -1f0
        flip[i] = o
    end
    return flip
end

function makeflip2()

    flip = Vector{Vector{Float32}}(undef, Const.layer2[1])
    for i in 1:Const.layer2[1]
        o = ones(Float32, Const.layer2[1])
        o[i] *= -1f0
        flip[i] = o
    end
    return flip
end

const flip1 = makeflip1()
const flip2 = makeflip2()

function update(s::Vector{Float32}, n::Vector{Float32})

    rng = MersenneTwister(1234)
    ls = length(s)
    ln = length(n)
    randomnum = rand(rng, Float32, ls + ln)

    #Update System
    zB = ANN.forward1(n)
    for ix in 1:ls
        s₁ = s[ix]
        z = ANN.forward2(s)
        sflip = s .* flip2[ix]
        zflip = ANN.forward2(sflip)
        prob = exp(2f0 * real(dot((zflip - z), zB)))
        @inbounds s[ix] = ifelse(randomnum[ix] < prob, -s₁, s₁)
    end

    #Update Bath
    zS = ANN.forward2(s)
    for iy in 1:ln
        n₁ = n[iy]
        z = ANN.forward1(n)
        nflip = n .* flip1[iy]
        zflip = ANN.forward1(nflip)
        prob = exp(2f0 * real(dot(zS, (zflip - z))))
        @inbounds n[iy] = ifelse(randomnum[ls + iy] < prob, -n₁, n₁)
    end
end

function hamiltonianS(s::Vector{Float32}, zS::Vector{Complex{Float32}},
                      zB::Vector{Complex{Float32}}, ix::Integer)

    out = 0f0im
    ixnext = ix%Const.dimS + 1
    if s[ix] != s[ixnext]
        sflip = s .* flip2[ix] .* flip2[ixnext]
        z     = dot(zS, zB)
        zflip = dot(ANN.forward2(sflip), zB)
        out  += 2f0 * exp(zflip - z) - 1f0
    else
        out += 1f0
    end

    return -Const.J * out / 4f0
end

function energyS(s::Vector{Float32}, n::Vector{Float32})

    zB = ANN.forward1(n)
    zS = ANN.forward2(s)
    sum = 0f0im
    @simd for ix in 1:Const.dimS
        sum += hamiltonianS(s, zS, zB, ix)
    end

    return sum
end

function hamiltonianB(n::Vector{Float32}, zS::Vector{Complex{Float32}},
                      zB::Vector{Complex{Float32}}, iy::Integer)

    out = 0f0im
    iynext = iy%Const.dimB + 1
    if n[iy] != n[iynext]
        nflip = n .* flip1[iy] .* flip1[iynext]
        z     = dot(zS, zB)
        zflip = dot(zS, ANN.forward1(nflip))
        out  += exp(zflip - z)
    end

    return -Const.t * out
end

function energyB(s::Vector{Float32}, n::Vector{Float32})

    zB = ANN.forward1(n)
    zS = ANN.forward2(s)
    sum = 0.0f0im
    @simd for iy in 1:Const.dimB 
        sum += hamiltonianB(n, zS, zB, iy)
    end

    return sum
end

end
