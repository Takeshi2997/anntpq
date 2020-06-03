module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CuArrays
using BSON: @save
using BSON: @load

mutable struct CuParams

    W::CuArray{Complex{Float32}, 2}
    b::CuArray{Complex{Float32}, 1}
end

o  = Vector{CuParams}(undef, Const.layers_num)
oe = Vector{CuParams}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num
        W = CuArray(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]))
        b = CuArray(zeros(Complex{Float32}, Const.layer[i+1]))
        global o[i]  = CuParams(W, b)
        global oe[i] = CuParams(W, b)
    end
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

function Network()

    func(x::Float32) = tanh(x)
    output(x::Complex{Float32}) = log(cosh(x))
    layer = Vector{Flux.Dense}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], func) |> gpu
    end
    W = randn(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end])
    layer[end] = Dense(W, b, output)
    f = Chain([layer[i] for i in 1:Const.layers_num]...)
    p = params(f)
    Network(f, p)
end

network = Network()

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
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i])
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = Parameters(W, b)
    end
    W = Flux.glorot_uniform(Const.layer[end], Const.layer[end-1]) .* 
    exp.(π*im* rand(Float32, Const.layer[end], Const.layer[end-1]))
    b  = zeros(Complex{Float32}, Const.layer[end])
    parameters[end] = Parameters(W, b)
    p  = params([[parameters[i].W, parameters[i].b] for i in 1:Const.layers_num]...)
    Flux.loadparams!(network.f, p)
end

function forward(x::Vector{Float32})

    out = network.f(x)
    return sum(out)
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})

    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num-1
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
    u = network.f[1:end-1](x)
    v = tanh.(network.f[end].W * u + network.f[end].b)
    dw = transpose(u) .* v
    db = v
    o[end].W  += dw
    oe[end].W += dw * e
    o[end].b  += db
    oe[end].b += db * e
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energy::Float32, ϵ::Float32, lr::Float32)

    α = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* real.(oe[i].b .- energy * o[i].b)
        vW = abs2.(o[i].W)
        vb = abs2.(o[i].b)
        update!(opt(lr), network.f[i].W, ΔW, vW)
        update!(opt(lr), network.f[i].b, Δb, vb)
    end
    ΔW = α .* (oe[end].W .- energy * o[end].W)
    Δb = α .* (oe[end].b .- energy * o[end].b)
    vW = conj.(o[end].W) .* o[end].W
    vb = conj.(o[end].b) .* o[end].b
    update!(opt(lr), network.f[end].W, ΔW, vW)
    update!(opt(lr), network.f[end].b, Δb, vb)
end

const ϵ = 1e-8

mutable struct QRMSProp
  eta::Float64
  rho::Float64
  acc::IdDict
end

QRMSProp(η = 0.001, ρ = 0.9) = QRMSProp(η, ρ, IdDict())

function apply!(o::QRMSProp, x, g, Δ)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * Δ
  @. g *= η / (√acc + ϵ)
end

function update!(opt, x, x̄, x̂)
  x .-= apply!(opt, x, x̄, x̂)
end

function update!(opt, xs::Params, gs, o)
  for x in xs
    gs[x] == nothing && continue
    update!(opt, x, gs[x], o)
  end
end

end
