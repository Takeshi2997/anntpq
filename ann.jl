module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CuArrays, CUDAnative
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

abstract type Parameters end

mutable struct Layer <: Parameters
    W::CuArray{Complex{Float32}, 2}
    b::CuArray{Complex{Float32}, 1}
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()
    for i in 1:Const.layers_num-1
        W = CuArray(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]))
        b = CuArray(zeros(Complex{Float32}, Const.layer[i+1]))
        global o[i]  = Layer(W, b)
        global oe[i] = Layer(W, b)
    end
    W = CuArray(zeros(Complex{Float32}, Const.layer[end], Const.layer[end-1]))
    b = CuArray(zeros(Complex{Float32}, Const.layer[1]))
    global o[end]  = Layer(W, b)
    global oe[end] = Layer(W, b)
end

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

# Define Network

struct Output{S<:AbstractArray,T<:AbstractArray}
  W::S
  b::T
end

function Output(in::Integer, out::Integer, first::Integer;
               initW = Flux.glorot_uniform, initb = zeros)
    return Output(initW(out, in), initb(first))
end

@functor Output

function (a::Output)(x::AbstractArray)
  W, b = a.W, a.b
  W*x, b
end

(a::Output{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Output{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

function Network()
    layer = Vector{Flux.Dense}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], CUDAnative.tanh) |> gpu
    end
    layer[end] = Output(Const.layer[end-1], Const.layer[end], Const.layer[1]) |> gpu
    f = Chain([layer[i] for i in 1:Const.layers_num]...)
    p = params(f)
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
    p = params(f)
    Flux.loadparams!(network.f, p)
end

function init()
    parameters = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        W = CuArray(Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]))
        b = CuArray(zeros(Float32, Const.layer[i+1]))
        parameters[i] = Parameters(W, b)
    end
    W = CuArray(Flux.glorot_uniform(Const.layer[end], Const.layer[end-1]))
    b = CuArray(zeros(Float32, Const.layer[1]))
    parameters[end] = Parameters(W, b)
    p = params([[parameters[i].W, parameters[i].b] for i in 1:Const.layers_num]...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::CuArray{Float32, 1})
    out, b = network.f(x)
    return log(out[1] + im * out[2]) + transpose(b) * x
end

realloss(x::CuArray{Float32, 1}) = network.f(x)[1]
imagloss(x::CuArray{Float32, 1}) = network.f(x)[2]

function backward(x::CuArray{Float32, 1}, e::Complex{Float32})
    realgs = gradient(() -> realloss(x), network.p)
    imaggs = gradient(() -> imagloss(x), network.p)
    for i in 1:Const.layers_num
        dw = realgs[network.f[i].W] .- im * imaggs[network.f[i].W]
        db = realgs[network.f[i].b] .- im * imaggs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)
    α = 2.0f0 * (energy - ϵ) / Const.iters_num
    for i in 1:Const.layers_num
        ΔW = α .* 2f0 .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* 2f0 .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
end

end
