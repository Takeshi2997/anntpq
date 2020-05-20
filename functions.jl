module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra, Distributed

function flip(x::Vector{Float32}, iy::Integer)

    xflip = copy(x)
    xflip[iy] *= -1f0
    return xflip
end

function flip2(x::Vector{Float32}, iy::Integer, ix::Integer)

    xflip = copy(x)
    xflip[iy] *= -1f0
    xflip[ix] *= -1f0
    return xflip
end

function updateS(s::Vector{Float32}, n::Vector{Float32})
    
    for ix in 1:length(s)
        s₁ = s[ix]
        z = ANN.forward(s, n)
        sflip = flip(s, ix)
        zflip = ANN.forward(sflip, n)
        prob = exp(2.0f0 * real(zflip - z))
        @inbounds s[ix] = ifelse(rand(Float32) < prob, -s₁, s₁)
    end
end

function updateB(s::Vector{Float32}, n::Vector{Float32})
    
    for iy in 1:length(n)
        n₁ = n[iy]
        z = ANN.forward(s, n)
        nflip = flip(n, iy)
        zflip = ANN.forward(s, nflip)
        prob = exp(2.0f0 * real(zflip - z))
        @inbounds n[iy] = ifelse(rand(Float32) < prob, -n₁, n₁)
    end
end

function hamiltonianS(s::Vector{Float32}, n::Vector{Float32},
                      z::Complex{Float32}, ix::Integer)

    out = 1.0f0 + 0.0f0im
    ixnext = ix % Const.dimS + 1
    if s[ix] != s[ixnext]
        sflip = flip2(s, ix, ixnext)
        zflip = ANN.forward(sflip, n)
        out   = 2.0f0 * exp(zflip - z) - 1.0f0
    end

    return -Const.J * out / 4.0f0
end

function energyS(s::Vector{Float32}, n::Vector{Float32})

    z = ANN.forward(s, n)
    sum = 0.0f0im
    @simd for ix in 1:Const.dimS
        sum += hamiltonianS(s, n, z, ix)
    end

    return sum
end

function hamiltonianB(s::Vector{Float32}, n::Vector{Float32}, 
                      z::Complex{Float32}, iy::Integer)

    out = 0.0f0im
    iynext = iy%Const.dimB + 1
    if n[iy] != n[iynext]
        nflip = flip2(n, iy, iynext)
        zflip = ANN.forward(s, nflip)
        out  += exp(zflip - z)
    end

    return Const.t * out
end

function energyB(s::Vector{Float32}, n::Vector{Float32})

    z = ANN.forward(s, n)
    sum = 0.0f0im
    @simd for iy in 1:Const.dimB 
        sum += hamiltonianB(s, n, z, iy)
    end

    return sum
end

end
