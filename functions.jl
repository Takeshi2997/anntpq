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

function update(x::Vector{Float32})
    
    for ix in 1:length(x)
        x₁ = x[ix]
        z = ANN.forward(x)
        xflip = flip(x, ix)
        zflip = ANN.forward(xflip)
        prob = exp(2.0f0 * real(zflip - z))
        @inbounds x[ix] = ifelse(rand(Float32) < prob, -x₁, x₁)
    end
end

function hamiltonianS(x::Vector{Float32},
                      z::Complex{Float32}, ix::Integer)

    out = 1.0f0 + 0.0f0im
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    if x[ix] != x[ixnext]
        xflip = flip2(x, ix, ixnext)
        zflip = ANN.forward(xflip)
        out   = 2.0f0 * exp(zflip - z) - 1.0f0
    end

    return -Const.J * out / 4.0f0 + 1.0f0 / 4.0f0
end

function energyS(x::Vector{Float32})

    z = ANN.forward(x)
    sum = 0.0f0im
    @simd for ix in Const.dimB+1:Const.dimB+Const.dimS
        sum += hamiltonianS(x, z, ix)
    end

    return sum
end

function hamiltonianB(x::Vector{Float32}, 
                      z::Complex{Float32}, iy::Integer)

    out = 0.0f0im
    iynext = iy%Const.dimB + 1
    if x[iy] != x[iynext]
        xflip = flip2(x, iy, iynext)
        zflip = ANN.forward(xflip)
        out  += exp(zflip - z)
    end

    return Const.t * out + 1.0f0
end

function energyB(x::Vector{Float32})

    z = ANN.forward(x)
    sum = 0.0f0im
    @simd for iy in 1:Const.dimB 
        sum += hamiltonianB(x, z, iy)
    end

    return sum
end

function hamiltonianI(x::Vector{Float32}, z::Complex{Float32}, 
                      ix::Integer, iy::Integer)

    out = 0.0f0im
    if x[ix] != x[iy]
        xflip = flip2(x, ix, iy)
        zflip = ANN.forward(xflip)
        out  += exp(zflip - z)
    end

    return Const.λ * out
end

function energyI(x::Vector{Float32})

    z = ANN.forward(x)
    sum = 0.0f0im
    @simd for ixy in CartesianIndices((Const.dimB+1:Const.dimB+Const.dimS, 1:Const.dimB))
        ix, iy = Tuple(ixy)
        sum += hamiltonianI(x, z, ix, iy)
    end

    return sum
end

end
