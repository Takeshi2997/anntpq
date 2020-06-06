module Const

# System Size
const dimS = 8
const dimB = 80

# System Param
const t = 1.0f0
const J = 1.0f0
const λ = 0.001f0
const η = 0.5f0

# Repeat Number
const burnintime = 10
const iters_num = 200
const it_num = 1000
const iϵmax = 10
const num = 1000

# Network Params
const layer = [dimB+dimS, 48, 48, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.00001f0

end
