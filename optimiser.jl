module Optimise
using Flux
import Zygote: Params

const ϵ = 1f-8

mutable struct ADAM
  eta::Float32
  beta::Tuple{Float32,Float32}
  state::IdDict
end

ADAM(η = 0.001f0, β = (0.9f0, 0.999f0)) = ADAM(η, β, IdDict())

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
  return Δ
end

mutable struct QRMSProp
  eta::Float32
  rho::Float32
  acc::IdDict
end

QRMSProp(η = 0.001f0, ρ = 0.9f0) = QRMSProp(η, ρ, IdDict())

function apply!(o::QRMSProp, x, g, O)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * abs2(O)
  @. g *= η / (√acc + ϵ)
end

function update!(opt, x, x̄)
  x .-= apply!(opt, x, x̄)
end

function update!(opt, xs::Params, gs)
  for x in xs
    gs[x] == nothing && continue
    update!(opt, x, gs[x])
  end
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


