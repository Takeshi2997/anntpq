module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CUDA
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

o   = Vector{Array}(undef, Const.layers_num)
oe  = Vector{Array}(undef, Const.layers_num)
oo  = Array{Array}(undef, Const.layers_num, Const.layers_num)
const I   = [Diagonal(CUDA.ones(Float32, Const.layer[i+1] * (Const.layer[i] + 1)))
             for i in 1:Const.layers_num]

function initO()
    for i in 1:Const.layers_num
        j = i%Const.layers_num + 1
        k = j%Const.layers_num + 1
        W  = zeros(Complex{Float32}, Const.layer[j] * (Const.layer[i] + 1))
        W′ = zeros(Complex{Float32}, Const.layer[k] * (Const.layer[j] + 1))
        S  = transpose(W) .* W
        S′ = transpose(W′) .* W
        global o[i]   = W
        global oe[i]  = W
        global oo[i, i] = S
        global oo[j, i] = S′
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

NNlib.logcosh(z::Complex) = log(cosh(z))

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

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
    W = randn(Complex{Float32}, Const.layer[end], Const.layer[end-1]+1) .* sqrt(2f0 / (Const.layer[end-1]+1))
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
    dw = [reshape(gs[network.f[i]], length(gs[network.f[i]])) for i in 1:Const.layers_num]
    for i in 1:Const.layers_num
        j = i%Const.layers_num + 1
        o[i]  += dw[i]
        oe[i] += dw[i] .* e
        oo[i, i] += transpose(dw[i]) .* conj.(dw[i])
        oo[j, i] += transpose(dw[j]) .* conj.(dw[i])
    end
end

opt(lr::Float32) = Descent(lr)

function update(energy::Float32, ϵ::Float32, lr::Float32)
    α = 1f0 / Const.iters_num
    for i in 1:Const.layers_num
        j = i%Const.layers_num + 1
        o[i]  ./= Const.iters_num
        oe[i] ./= Const.iters_num 
        oo[i, i] ./= Const.iters_num 
        oo[j, i] ./= Const.iters_num 
    end
    for i in 1:Const.layers_num
        j = i%Const.layers_num + 1
        R  = CuArray(2f0 .* real.(o[i] .- energy * o[i]))
        S  = CuArray(2f0 .* energy .* real.(oo[i, i] - transpose(o[i]) .* conj.(o[i])))
        R′ = CuArray(2f0 .* real.(o[j+1] .- energy * o[j]))
        S′ = CuArray(2f0 .* energy .* real.(oo[j, i] - transpose(o[j]) .* conj.(o[i])))
        ΔW = reshape((S′ .+ Const.η .* I[j])\R′ - (S .+ Const.η .* I[i])\R, 
                     (Const.layer[i+1], Const.layer[i]+1)) |> cpu
        update!(opt(lr), network.f[i].W, ΔW)
    end
end

end
