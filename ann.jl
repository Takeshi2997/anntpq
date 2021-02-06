module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

o   = Vector{Vector{Array}}(undef, Const.layers_num)
oe  = Vector{Vector{Array}}(undef, Const.layers_num)
oo  = Vector{Vector{Array}}(undef, Const.layers_num)
const I = [Diagonal(ones(Float32, Const.layer[i+1], Const.layer[i+1])) for i in 1:Const.layers_num] 

function initO()
    for i in 1:Const.layers_num-1
        v = zeros(Complex{Float32}, Const.layer[i+1])
        S = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i+1])
        global o[i]  = [v for j in 1:Const.layer[i] + 1]
        global oe[i] = [v for j in 1:Const.layer[i] + 1]
        global oo[i] = [S for j in 1:Const.layer[i] + 1]
    end
    v = zeros(Complex{Float32}, Const.layer[end])
    S = zeros(Complex{Float32}, Const.layer[end], Const.layer[end])
    global o[end]  = [v for j in 1:Const.layer[end-1]]
    global oe[end] = [v for j in 1:Const.layer[end-1]]
    global oo[end] = [S for j in 1:Const.layer[end-1]]
end

# Define Network

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Dense(Const.layer[i], Const.layer[i+1], tanh)
    end
    layers[end] = Dense(Const.layer[end-1], Const.layer[end])
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
    for i in 1:Const.layers_num
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i])
        b = Flux.zeros(Const.layer[i+1])
        parameters[i] = [W, b]
    end
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
        dwvec = vcat([@views dw[:, j] for j in 1:Const.layer[i]])
        for j in 1:Const.layer[i]
            oo[i][j] += transpose(dwvec[j]) .* conj.(dwvec[j])
            o[i][j]  += dwvec[j]
            oe[i][j] += dwvec[j] .* e
        end
        if i < Const.layers_num
            db = gs[network.f[i].b]
            oo[i][end] += transpose(db) .* conj.(db)
            o[i][end]  += db
            oe[i][end] += db .* e
        end 
    end
end

opt(lr::Float32) = Descent(lr)

function update(energy::Float32, ϵ::Float32, lr::Float32)
    for i in 1:Const.layers_num
        for j in 1:length(o[i])
            o[i][j]  ./= Const.iters_num
            oe[i][j] ./= Const.iters_num 
            oo[i][j] ./= Const.iters_num 
        end
    end
    @simd for i in 1:Const.layers_num
        R  = [2f0 * real.(oe[i][j] - (ϵ - energy) * o[i][j]) for j in 1:Const.layer[i]]
        ΔW = zeros(Float32, Const.layer[i+1], Const.layer[i])
        for j in 1:Const.layer[i]
            OO = transpose(oo[i][j]) .* conj.(o[i][j])
            S  = 2f0 * real.(oo[i][j] - OO)
            ΔW[:, j]  = -(S + Const.η * I[i])\R[j]
        end
        update!(opt(lr), network.f[i].W, ΔW)
        if i < Const.layers_num
            Rb = 2f0 * real.(oe[i][end] - (ϵ - energy) * o[i][end])
            OO = transpose(o[i][end]) .* conj.(o[i][end])
            S  = 2f0 * real.(oo[i][end] - OO)
            Δb  = -(S + Const.η * I[i])\Rb
            update!(opt(lr), network.f[i].b, Δb)
        end
    end
end

end
