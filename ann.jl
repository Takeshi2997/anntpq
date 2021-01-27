module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, Distributions
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

abstract type Parameters end
mutable struct Params{S<:AbstractArray} <: Parameters
    W::S
end

o   = Vector{Parameters}(undef, Const.layers_num)
oe  = Vector{Parameters}(undef, Const.layers_num)
oo  = Vector{Parameters}(undef, Const.layers_num)

function initO()
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i] + 1)
        S = kron(W, W')
        global o[i]   = Params(W)
        global oe[i]  = Params(W)
        global oo[i]  = Params(S)
    end
end

# Define Network

struct Layer{F,S<:AbstractArray}
    W::S
    σ::F
end
function Layer(in::Integer, out::Integer, σ = identity;
               initW = Flux.glorot_uniform)
    return Layer(initW(out, in+1), σ)
end
@functor Layer
function (m::Layer)(x::AbstractArray)
    W, σ = m.W, m.σ
    z = vcat(x, 1)
    σ.(W*z)
end

struct Output{F,S<:AbstractArray}
    W::S
    σ::F
end
function Output(in::Integer, out::Integer, σ = identity;
               initW = randn)
    return Output(initW(Complex{Float32}, out, in+1), σ)
end
@functor Output
function (m::Output)(x::AbstractArray)
    W, σ = m.W, m.σ
    z = vcat(x, 1)
    σ.(W*z)
end

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

NNlib.logcosh(z::Complex) = log(cosh(z))
function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Layer(Const.layer[i], Const.layer[i+1], tanh)
    end
    layers[end] = Output(Const.layer[end-1], Const.layer[end], logcosh)
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, p)
end

network = Network()

# Network Utility

function save(filename)
    f = getfield(network, :f)
    @save filename f
end

function load(filename)
    @load filename f
    p = Flux.params(f)
    Flux.loadparams!(network.f, p)
end

function init()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]+1) 
        parameters[i] = [W]
    end
    W = randn(Complex{Float32}, Const.layer[end], Const.layer[end-1]+1) 
    parameters[end] = [W]
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float32})
    out = network.f(x)
    return sum(out)
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        o[i].W  += dw
        oe[i].W += dw .* e
        oo[i].W += kron(dw, dw')
    end
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)
    α = 1f0 / Const.iters_num
    for i in 1:Const.layers_num-1
        O  = real.(o[i].W)
        OE = real.(oe[i].W)
        OO = real.(oo[i].W)
        R  = α .*  2f0 .* (energy - ϵ) .* reshape((OE .- energy * O), length(O))
        S  = α .* (OO - kron(O, O'))
        I  = Diagonal(ones(Float32, size(S)))
        ΔW = resgape((S .+ Const.ϵ .* I)\R, size(O))
        update!(opt(lr), network.f[i].W, ΔW)
    end
    O  = o[end].W
    OE = oe[end].W
    OO = oo[end].W
    R  = α .*  2f0 .* (energy - ϵ) .* reshape((OE .- energy * O), length(O))
    S  = α .* (OO - kron(O, O'))
    I  = Diagonal(ones(Float32, size(S)))
    ΔW = resgape((S .+ Const.ϵ .* I)\R, size(O))
    update!(opt(lr), network.f[i].W, ΔW)
end

end
