module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra, Random, CuArrays


function makeflip()

    flip = Vector{CuArray{Float32, 1}}(undef, Const.layer[1])
    for i in 1:Const.layer[1]
        o = ones(Float32, Const.layer[1])
        o[i] *= -1f0
        flip[i] = CuArray(o)
    end
    return flip
end

const flip = makeflip()

function update(x::CuArray{Float32, 1})

    rng = MersenneTwister(1234)
    l = length(x)
    randamnum = rand(rng, Float32, l)
    for ix in 1:l
        z = ANN.forward(x)
        xflip = x .* flip[ix]
        zflip = ANN.forward(xflip)
        prob = exp(2.0f0 * real(zflip - z))
        @inbounds x = ifelse(randamnum[ix] < prob, xflip, x)
    end
end

function hamiltonianS(x::CuArray{Float32, 1},
                      z::Complex{Float32}, ix::Integer)

    out = 1.0f0 + 0.0f0im
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    if x[ix] != x[ixnext]
        xflip = x .* flip[ix] .* flip[ixnext]
        zflip = ANN.forward(xflip)
        out   = 2f0 * exp(zflip - z) - 1f0
    end

    return -Const.J * out / 4f0 + 1f0 / 4f0
end

function energyS(x::CuArray{Float32, 1})

    z = ANN.forward(x)
    sum = 0f0im
    @simd for ix in Const.dimB+1:Const.dimB+Const.dimS
        sum += hamiltonianS(x, z, ix)
    end

    return sum
end

function hamiltonianB(x::CuArray{Float32, 1},
                      z::Complex{Float32}, iy::Integer)

    out = 0f0im
    iynext = iy%Const.dimB + 1
    if x[iy] != x[iynext]
        xflip = x .* flip[iy] .* flip[iynext]
        zflip = ANN.forward(xflip)
        out  += exp(zflip - z)
    end

    return -Const.t * out + 1f0
end

function energyB(x::CuArray{Float32, 1})

    z = ANN.forward(x)
    sum = 0.0f0im
    @simd for iy in 1:Const.dimB 
        sum += hamiltonianB(x, z, iy)
    end

    return sum
end

function hamiltonianI(x::CuArray{Float32, 1}, z::Complex{Float32},
                      ix::Integer, iy::Integer)

    out = 0.0f0im
    if x[ix] != x[iy]
        xflip = x .* flip[ix] .* flip[iy]
        zflip = ANN.forward(xflip)
        out  += exp(zflip - z)
    end

    return Const.λ * out
end

function energyI(x::CuArray{Float32, 1})

    z = ANN.forward(x)
    sum = 0.0f0im
    @simd for ixy in CartesianIndices((Const.dimB+1:Const.dimB+Const.dimS, 1:Const.dimB))
        ix, iy = Tuple(ixy)
        sum += hamiltonianI(x, z, ix, iy)
    end

    return sum
end

end
