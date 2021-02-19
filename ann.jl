module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize
abstract type Parameters end
mutable struct Params{S<:AbstractArray, T<:AbstractArray} <: Parameters
    W::S
    b::T
end
mutable struct WaveFunction{S<:Complex} <: Parameters
    x::S
    y::S
end
mutable struct ParamSet{T <: Parameters}
    or::Vector{T}
    oi::Vector{T}
    oe::Vector{T}
    ϕr::Vector{T}
    ϕi::Vector{T}
    ϕ::T
end

function ParamSet()
    p = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Int(Const.layer[i+1]/2), Const.layer[i])
        b = zeros(Complex{Float32}, Int(Const.layer[i+1]/2))
        p[i] = Params(W, b)
    end
    ϕ = WaveFunction(0f0im, 0f0im)
    ParamSet(p, p, p, p, p, ϕ)
end

# Define Network

struct Layer{F,S<:AbstractArray,T<:AbstractArray}
    W1::S
    W2::S
    b1::T
    b2::T
    σ::F
end
function Layer(in::Integer, out::Integer, σ = identity;
               initW = Flux.glorot_uniform, initb = Flux.zeros)
    outharf = Int(out/2)
    return Layer(initW(outharf, in), initW(outharf, in), initb(outharf), initb(outharf), σ)
end
@functor Layer
function (m::Layer)(x::AbstractArray)
    W1, W2, b1, b2, σ = m.W1, m.W2, m.b1, m.b2, m.σ
    W = vcat(W1, W2)
    b = vcat(b1, b2)
    σ.(W*x.+b)
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
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, f, p, p)
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
    Flux.loadparams!(network.g, p)
end

function load_f(filename)
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
        W1 = Flux.glorot_uniform(Int(Const.layer[i+1]/2), Const.layer[i])
        W2 = Flux.glorot_uniform(Int(Const.layer[i+1]/2), Const.layer[i])
        b1 = Flux.zeros(Int(Const.layer[i+1]/2)) 
        b2 = Flux.zeros(Int(Const.layer[i+1]/2)) 
        parameters[i] = [W1, W2, b1, b2]
    end
    paramset = [param for param in parameters]
    q = Flux.params(paramset...)
    Flux.loadparams!(network.g, q)
end

function init()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W1 = Flux.glorot_uniform(Int(Const.layer[i+1]/2), Const.layer[i])
        W2 = Flux.glorot_uniform(Int(Const.layer[i+1]/2), Const.layer[i])
        b1 = Flux.zeros(Int(Const.layer[i+1]/2)) 
        b2 = Flux.zeros(Int(Const.layer[i+1]/2)) 
        parameters[i] = [W1, W2, b1, b2]
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

function forward_b(x::Vector{Float32})
    out = network.f(x)
    return out[1] + im * out[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    gs = gradient(() -> loss(x), network.q)
    ϕ = exp(forward_b(x) - forward(x))
    for i in 1:Const.layers_num
        dw1 = gs[network.f[i].W1]
        dw2 = gs[network.f[i].W2]
        db1 = gs[network.f[i].b1]
        db2 = gs[network.f[i].b2]
        paramset.or[i].W += dw1
        paramset.oi[i].W += dw2
        paramset.or[i].b += db1
        paramset.oi[i].b += db2
        paramset.oe[i].W += dw1 .* e
        paramset.oe[i].b += db1 .* e
        paramset.ϕr[i].W += dw1 .* ϕ
        paramset.ϕi[i].W += dw2 .* ϕ
        paramset.ϕr[i].b += db1 .* ϕ
        paramset.ϕi[i].b += db2 .* ϕ
    end
    paramset.ϕ.x += conj(ϕ) * ϕ
    paramset.ϕ.y += ϕ
end

opt(lr::Float32) = AMSGrad(lr, (0.9, 0.999))

function updateparams(e::Float32, lr::Float32, paramset::ParamSet, Δparamset::Vector)
    for i in 1:Const.layers_num
        paramset.or[i].W ./= Const.iters_num
        paramset.oi[i].W ./= Const.iters_num
        paramset.or[i].b ./= Const.iters_num
        paramset.oi[i].b ./= Const.iters_num
        paramset.oe[i].W ./= Const.iters_num
        paramset.oe[i].b ./= Const.iters_num
        paramset.ϕr[i].W ./= Const.iters_num
        paramset.ϕi[i].W ./= Const.iters_num
        paramset.ϕr[i].b ./= Const.iters_num
        paramset.ϕi[i].b ./= Const.iters_num
    end
    paramset.ϕ.x /= Const.iters_num
    paramset.ϕ.y /= Const.iters_num
    X = 1f0 / sqrt(real(paramset.ϕ.x))
    for i in 1:Const.layers_num
        Δparamset[i][1] += 
        real.(paramset.oe[i].W - e * paramset.or[i].W) - 
        X * (real.(paramset.ϕr[i].W) - real.(paramset.or[i].W) .* real.(paramset.ϕ.y))
        Δparamset[i][2] += 
        X * (imag.(paramset.ϕi[i].W) - real.(paramset.oi[i].W) .* imag.(paramset.ϕ.y))
        Δparamset[i][3] += 
        real.(paramset.oe[i].b - e * paramset.or[i].b) - 
        X * (real.(paramset.ϕr[i].b) - real.(paramset.or[i].b) .* real.(paramset.ϕ.y))
        Δparamset[i][4] += 
        X * (imag.(paramset.ϕi[i].b) - real.(paramset.oi[i].b) .* imag.(paramset.ϕ.y))
    end
end

function update(Δparamset::Vector, lr::Float32)
    for i in 1:Const.layers_num
        ΔW1 = Δparamset[i][1]
        ΔW2 = Δparamset[i][2]
        Δb1 = Δparamset[i][3]
        Δb2 = Δparamset[i][4]
        update!(opt(lr), network.g[i].W1, ΔW1)
        update!(opt(lr), network.g[i].W2, ΔW2)
        update!(opt(lr), network.g[i].b1, Δb1)
        update!(opt(lr), network.g[i].b2, Δb2)
    end
end
end
