module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CUDA, BlockDiagonals
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

o   = Array{Array}(undef, Const.layers_num)
oe  = Array{Array}(undef, Const.layers_num)
oo  = Array{Array}(undef, Const.layers_num)
const I = [Diagonal(CUDA.ones(Float32, Const.layer[i+1] * (Const.layer[i] + 1))) for i in 1:Const.layers_num]

function initO()
    for i in 1:Const.layers_num-1
        W  = zeros(Complex{Float32}, Const.layer[i+1] * (Const.layer[i] + 1))
        S  = transpose(W) .* W
        global o[i]  = W
        global oe[i] = W
        global oo[i] = S
    end
    W  = zeros(Complex{Float32}, Const.layer[end] * Const.layer[end-1])
    S  = transpose(W) .* W
    global o[end]  = W
    global oe[end] = W
    global oo[end] = S
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

struct Output{S<:AbstractArray}
    W::S
end
function Output(in::Integer, out::Integer;
                initW = Flux.glorot_uniform)
    return Output(initW(out, in))
end
@functor Output
function (m::Output)(x::AbstractArray)
    W = m.W
    W*x
end

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Layer(Const.layer[i], Const.layer[i+1], tanh)
    end
    layers[end] = Output(Const.layer[end-1], Const.layer[end])
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
    W = Flux.glorot_uniform(Const.layer[end], Const.layer[end-1]) 
    parameters[end] = [W]
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float32})
    out = network.f(x)
    return out[1] + im * out[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        oo[i] += BlockDiagonal([transpose(dw[:, j]) .* conj.(dw[:, j]) for j in 1:size(dw, 2)])
        dw = reshape(dw, length(dw))
        o[i]  += dw
        oe[i] += dw .* e
    end
end

opt(lr::Float32) = Descent(lr)

function update(energy::Float32, ϵ::Float32, lr::Float32)
    for i in 1:Const.layers_num
        o[i]  ./= Const.iters_num
        oe[i] ./= Const.iters_num 
        oo[i] ./= Const.iters_num 
    end
    for i in 1:Const.layers_num-1
        R   = CuArray(2f0 * real.(o[i] - (ϵ - energy) * o[i]))
        OO  = BlockDiagonal([transpose(o[i][:, j]) .* conj.(o[i][:, j]) for j in 1:size(o[i], 2)])
        S   = CuArray(2f0 * real.(oo[i] - OO))
        ΔW  = reshape((S + Const.η * I[i])\R, (Const.layer[i+1], Const.layer[i]+1)) |> cpu
        update!(opt(lr), network.f[i].W, ΔW)
    end
    R   = CuArray(2f0 * real.(o[end] - (ϵ - energy) * o[end]))
    OO  = BlockDiagonal([transpose(o[end][:, j]) .* conj.(o[end][:, j]) for j in 1:size(o[end], 2)])
    S   = CuArray(2f0 * real.(oo[end] - OO))
    ΔW  = reshape((S + Const.η * I[end])\R, (Const.layer[end], Const.layer[end-1])) |> cpu
    update!(opt(lr), network.f[end].W, ΔW)
end

end
