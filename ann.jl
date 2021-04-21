module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CUDA
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables
mutable struct ParamSet{T <: AbstractArray, S <: AbstractArray}
    o::T
    oe::T
    oo::S
end

function ParamSet()
    W = zeros(Complex{Float32}, Const.networkdim)
    S = transpose(W) .* W
    ParamSet(W, W, S)
end

const I = Diagonal(ones(Float32, Const.networkdim))

# Define Network

struct RBM{F,S<:AbstractArray,T<:AbstractArray}
    W::S
    b::T
    a::T
    σ::F
end
function RBM(in::Integer, out::Integer, σ = identity;
             initW = randn, initb = zeros, inita = randn)
    return RBM(initW(Complex{Float32}, out, in), initb(Complex{Float32}, out), inita(Complex{Float32}, in), σ)
end
@functor RBM
function (m::RBM)(x::AbstractArray)
    W, b, a, σ = m.W, m.b, m.a, m.σ
    sum(σ.(W*x+b))+transpose(a)*x
end

NNlib.logcosh(x::Complex{Float32}) = log(2f0 * cosh(x))

mutable struct Network
    f::RBM
    p::Zygote.Params
end

function Network()
    f = RBM(Const.layer[1], Const.layer[2], logcosh)
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
    W = Flux.kaiming_uniform(Const.layer[2], Const.layer[1])
    b = Flux.zeros(Const.layer[2])
    a = Flux.zeros(Const.layer[1])
    paramset = [W, b, a]
    p = Flux.params(paramset)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float32})
    out = network.f(x)
    return out
end

loss(x::Vector{Float32}) = real(network.f(x))

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    gs = gradient(() -> loss(x), network.p)
    dW = reshape(gs[network.f.W], Const.layer[1]*Const.layer[2])
    db = gs[network.f.b]
    da = gs[network.f.a]
    dθ = vcat(dW, db, da)
    paramset.o  += dθ
    paramset.oe += dθ .* e
end

function updateparams(energy::Float32, paramset::ParamSet, Δparamset::Array)
    o  = paramset.o  / Const.iters_num
    oe = paramset.oe / Const.iters_num
    Δparamset += oe - energy * o
end

opt(lr::Float32) = Descent(lr)

function update(Δparamset::Vector, lr::Float32)
    ΔW = reshape(Δparamset[1:Const.layer[2]*Const.layer[1]], Const.layer[2], Const.layer[1])
    n  = Const.layer[2] * Const.layer[1]
    Δb = Δparamset[n+1:n+Const.layer[2]]
    n += Const.layer[2]
    Δa = Δparamset[n+1:n+Const.layer[1]]
    update!(opt(lr), network.f.W, ΔW)
    update!(opt(lr), network.f.b, Δb)
    update!(opt(lr), network.f.a, Δa)
end

function srbackward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    gs = gradient(() -> loss(x), network.p)
    dW = reshape(gs[network.f.W], Const.layer[1]*Const.layer[2])
    db = gs[network.f.b]
    da = gs[network.f.a]
    dθ = vcat(dW, db, da)
    paramset.o  += dθ
    paramset.oe += dθ .* e
    paramset.oo += transpose(dθ) .* conj.(dθ)
end

function srupdateparams(energy::Float32, paramset::ParamSet, Δparamset::Array)
    o  = CuArray(paramset.o  / Const.iters_num)
    oe = CuArray(paramset.oe / Const.iters_num)
    oo = CuArray(paramset.oo / Const.iters_num)
    R  = oe - energy * o
    S  = oo - transpose(o) .* conj.(o)
    Δparamset += -im .* svd(S) \ R |> cpu
end
end
