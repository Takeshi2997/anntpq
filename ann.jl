module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CUDA
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

abstract type Parameters end
mutable struct Params{S<:AbstractArray} <: Parameters
    W::S
end
mutable struct Collect{S<:Complex} <: Parameters
    ϕ::S
end

o   = Vector{Parameters}(undef, Const.layers_num)
oe  = Vector{Parameters}(undef, Const.layers_num)
oo  = Vector{Parameters}(undef, Const.layers_num)
v1  = Vector{Parameters}(undef, Const.layers_num)
v2  = Parameters

const I = [Diagonal(CUDA.ones(Float32, Const.layer[i+1] * (Const.layer[i] + 1))) for i in 1:Const.layers_num]

function initO()
    for i in 1:Const.layers_num
        W  = zeros(Complex{Float32}, Const.layer[i+1] * (Const.layer[i] + 1))
        S  = transpose(W) .* W
        global o[i]  = Params(W)
        global oe[i] = Params(W)
        global oo[i] = Params(S)
        global v1[i] = Params(W)
    end
    global v2 = Collect(0f0im)
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

mutable struct Network
    f::Flux.Chain
    g::Flux.Chain
    p::Zygote.Params
    q::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Layer(Const.layer[i], Const.layer[i+1], tanh)
    end
    layers[end] = Layer(Const.layer[end-1], Const.layer[end])
    ψ = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(ψ)
    Network(ψ, ψ, p, p)
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

function reset()
    g = getfield(network, :g)
    q = Flux.params(g)
    Flux.loadparams!(network.f, q)
end

function init_sub()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]+1) 
        parameters[i] = [W]
    end
    paramset = [param for param in parameters]
    q = Flux.params(paramset...)
    Flux.loadparams!(network.g, q)
end

function init()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]+1) 
        parameters[i] = [W]
    end
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float32})
    out = network.g(x)
    return out[1] + im * out[2]
end

function forward_ϕ(x::Vector{Float32})
    out = network.f(x)
    return out[1] + im * out[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.q)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        dw = reshape(dw, length(dw))
        o[i].W  += dw
        oe[i].W += dw .* e
        oo[i].W += transpose(dw) .* conj.(dw)
        v1[i].W += dw .* forward_ϕ(x) ./ forward(x)
    end
    v2.ϕ += forward_ϕ(x) ./ forward(x)
end

opt(lr::Float32) = Descent(lr)

function update(energy::Float32, ϵ::Float32, lr::Float32)
    for i in 1:Const.layers_num
        o[i].W  /= Const.iters_num
        oe[i].W /= Const.iters_num
        oo[i].W /= Const.iters_num
        v1[i].W /= Const.iters_num
    end
    v2.ϕ /= Const.iters_num
    for i in 1:Const.layers_num-1
        R = CuArray(oe[i].W - ((ϵ - energy) - 1f0) * o[i].W)
        S = CuArray(oo[i].W - transpose(o[i].W) .* conj.(o[i].W))
        x = CuArray(v1[i].W - v2.ϕ .* o[i].W)
        v = (S .+ Const.η .* I[i])\x
        α = dot(R, v) - 1f0
        ΔW = reshape(2f0 .* real.(v./α), (Const.layer[i+1], Const.layer[i]+1)) |> cpu
        update!(opt(lr), network.g[i].W, ΔW)
    end
end

end
