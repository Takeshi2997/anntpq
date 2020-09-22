module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra, CuArrays, CUDAnative

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

function update(x::CuArray{Float32, 1}, α::CuArray{Float32, 1}, randvec::CuArray{Float32, 1})

    l = length(x)
    for iα in 1:l
        z = ANN.forward(α) + dot(x, ANN.interaction(α))
        αflip = α .* flip[iα]
        zflip = ANN.forward(αflip) + dot(x, ANN.interaction(αflip))
        prob = exp(2f0 * real(zflip - z))
        @inbounds α[iα] = ifelse(randvec[iα] < prob, αflip, α)
    end

    prob = exp.(-2f0 .* 2f0 .* real.(x .* ANN.interaction(α)))
    x  .*= ifelse.((@view randvec[l+1:end]) .< prob, -1f0, 1f0)
end

function hamiltonianS(x::Vector{Float32}, z::Vector{Complex{Float32}})

    out = 0f0im
    if x[1] != x[2]
        out += 2f0 * exp(-2f0 * transpose(x) * z) - 1f0
    else
        out += 1f0
    end

    return -Const.J * out / 4f0
end

function energyS(x::CuArray{Float32, 1}, α::CuArray{Float32, 1})

    z = ANN.interaction(α)
    sum = 0f0im
    @simd for ix in Const.dimB+1:Const.dimB+Const.dimS
        ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
        xloc = [x[ix], x[ixnext]]
        zloc = [z[ix], z[ixnext]]
        sum += hamiltonianS(xloc, zloc)
    end

    return sum
end

function hamiltonianB(x::Vector{Float32}, z::Vector{Complex{Float32}})

    out = 0f0im
    if x[1] != x[2]
        out  += exp(-2f0 * transpose(x) * z)
    end

    return -Const.t * out
end

function energyB(x::CuArray{Float32, 1}, α::CuArray{Float32, 1})

    z = ANN.interaction(α)
    sum = 0.0f0im
    @simd for iy in 1:Const.dimB 
        iynext = iy%Const.dimB + 1
        xloc = [x[iy], x[iynext]]
        zloc = [z[iy], z[iynext]]
        sum += hamiltonianB(xloc, zloc)
    end

    return sum
end

end
