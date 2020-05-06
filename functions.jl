module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra

function flip(x::Vector{Float32}, iy::Integer)

    xflip = copy(x)
    xflip[iy] = -xflip[iy]
    return xflip
end

function flip2(x::Vector{Float32}, iy::Integer, ix::Integer)

    xflip = copy(x)
    xflip[iy] = -xflip[iy]
    xflip[ix] = -xflip[ix]
    return xflip
end

function update(x::Vector{Float32})
    
    for ix in 1:length(x)
        x₁ = x[ix]
        z = ANN.forward(x)
        xflip = flip(x, ix)
        zflip = ANN.forward(xflip)
        prob = exp(2.0f0 * real(zflip .- z))
        @inbounds x[ix] = ifelse(rand(Float32) < prob, -x₁, x₁)
    end
end

function hamiltonianS_shift(x::Vector{Float32},
                            z::Complex{Float32}, ix::Integer)

    out = 0.0f0im
    ixnext = Const.dimB + ix%Const.dimS + 1
    if x[ix] != x[ixnext]
        xflip = flip2(x, ix, ixnext)
        zflip = ANN.forward(xflip)
        out  += 1.0f0 - exp(zflip - z)
    end

    return Const.J * out / 2.0f0
end

function energyS_shift(x::Vector{Float32})

    z = ANN.forward(x)
    sum = 0.0f0im
    for ix in 1:2:Const.dimS-1
        sum += hamiltonianS_shift(x, z, ix)
    end

    return sum
end

function hamiltonianB_shift(x::Vector{Float32}, 
                            z::Complex{Float32}, iy::Integer)

    out = 0.0f0im
    iynext = iy%Const.dimB + 1
    if x[iy] != x[iynext]
        xflip = flip2(x, iy, iynext)
        zflip = ANN.forward(xflip)
        out  += -exp(zflip .- z)
    end

    return Const.t * out + 1.0f0
end

function energyB_shift(x::Vector{Float32})

    z = ANN.forward(x)
    sum = 0.0f0im
    for iy in 1:Const.dimB
        sum += hamiltonianB_shift(x, z, iy)
    end

    return sum
end

end
