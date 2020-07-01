module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra, Distributed, Random, CuArrays, CUDAnative

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

    l = length(x)
    rng = MersenneTwister(1234)
    randamnum = rand(rng, Float32, l)
    for ix in 1:l
        z = ANN.forward(x)
        xflip = x .* flip[ix]
        zflip = ANN.forward(xflip)
        prob  = exp(2.0f0 * real(zflip - z))
        x = ifelse(randamnum[ix] < prob, xflip, x)
    end
end

function hamiltonianS(x::CuArray{Float32, 1},
                      z::CuArray{Complex{Float32}, 1}, 
                      out::Vector{Complex{Float32}, 1})

    ixvector     = collect(Const.dimB+1:Const.dimB+Const.dimS)
    ixnextvector = Const.dimB .+ (ixvector .- Const.dimB) .% Const.dimS .+ 1
    out .+= ifelse.(x[ixvector] .!= x[ixnextvector], 
                    2f0 * exp(ANN.forward(x.*flip[ix].*flip[ixnext]) - z) - 1f0, 
                    1f0) .* (-Const.J ./ 4f0)
end

function energyS(x::CuArray{Float32, 1})

    z   = ANN.forward(x)
    out = zeros(Complex{Float32}, Const.dimS)
    hamiltonianS(x, z, out)

    return sum(out)
end

function hamiltonianB(x::CuArray{Float32, 1},
                      z::CuArray{Complex{Float32}, 1}, 
                      out::Vector{Complex{Float32}, 1})

    iyvector     = collect(1:Const.dimB)
    iynextvector = iyvector .% Const.dimB .+ 1
    out .+= ifelse.(x[iyvector] .!= x[iynextvector], 
                    exp(ANN.forward(x.*flip[ix].*flip[ixnext]) - z), 
                    0f0) .* (-Const.t)
end

function energyB(x::CuArray{Float32, 1})

    z = ANN.forward(x)
    out = zeros(Complex{Float32}, Const.dimB)
    hamiltonianB(x, z, out)

    return sum(out)
end

end
