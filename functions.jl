module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra

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

function update(x::Vector{Float32}, α::Vector{Float32}, α′::Vector{Float32}, randvec)

    l = length(x)

    for iα in 1:l
        α₁ = α[iα]
        z = ANN.forward(α) + dot(x, ANN.interaction(α))
        αflip = α .* flip[iα]
        zflip = ANN.forward(αflip) + dot(x, ANN.interaction(αflip))
        prob = exp(real(zflip - z))
        @inbounds α[iα] = ifelse(randvec[iα] < prob, -α₁, α₁)
    end

    for iα in 1:l
        α′₁ = α′[iα]
        z′ = ANN.forward(α′) + dot(x, ANN.interaction(α′))
        α′flip = α′ .* flip[iα]
        z′flip = ANN.forward(α′flip) + dot(x, ANN.interaction(α′flip))
        prob = exp(real(z′flip - z′))
        @inbounds α′[iα] = ifelse(randvec[l+iα] < prob, -α′₁, α′₁)
    end

    prob = exp.(-2f0 .* real.(x .* (ANN.interaction(α) .+ ANN.interaction(α′))))
    x  .*= ifelse.((@view randvec[2*l+1:end]) .< prob, -1f0, 1f0)
end

function hamiltonianS(x::Vector{Float32}, z::Vector{Complex{Float32}}, z′::Vector{Complex{Float32}})

    out = 0f0im
    if x[1] != x[2]
        out += 2f0 * (exp(-2f0 * transpose(x) * z) + exp(-2f0 * transpose(x) * z′)) / 2f0 - 1f0
    else
        out += 1f0
    end

    return -Const.J * out / 4f0
end

function energyS(x::Vector{Float32}, α::Vector{Float32}, α′::Vector{Float32})

    z  = ANN.interaction(α)
    z′ = ANN.interaction(α′)
    sum = 0f0im
    @simd for ix in Const.dimB+1:Const.dimB+Const.dimS
        ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
        xloc  = [x[ix], x[ixnext]]
        zloc  = [z[ix], z[ixnext]]
        z′loc = conj.([z′[ix], z′[ixnext]])
        sum += hamiltonianS(xloc, zloc, z′loc)
    end

    factor = exp(im * imag(ANN.forward(α)  + dot(x, z) - 
                          (ANN.forward(α′) + dot(x, z′))))
    return sum * factor
end

function hamiltonianB(x::Vector{Float32}, z::Vector{Complex{Float32}}, z′::Vector{Complex{Float32}})

    out = 0f0im
    if x[1] != x[2]
        out  += (exp(-2f0 * transpose(x) * z) + exp(-2f0 * transpose(x) * z′)) / 2f0
    end

    return -Const.t * out
end

function energyB(x::Vector{Float32}, α::Vector{Float32}, α′::Vector{Float32})

    z  = ANN.interaction(α)
    z′ = ANN.interaction(α′)
    sum = 0.0f0im
    @simd for iy in 1:Const.dimB 
        iynext = iy%Const.dimB + 1
        xloc  = [x[iy], x[iynext]]
        zloc  = [z[iy], z[iynext]]
        z′loc = conj.([z′[iy], z′[iynext]])
        sum += hamiltonianB(xloc, zloc, z′loc)
    end

    factor = exp(im * imag(ANN.forward(α)  + dot(x, z) - 
                          (ANN.forward(α′) + dot(x, z′))))
    return sum * factor
end

end
