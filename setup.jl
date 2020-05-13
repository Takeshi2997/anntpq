module Const

# System Size
const dimS = 8
const dimB = 64

# System Param
const t = 1.0f0
const J = 1.0f0
const λ = 0.1f0

# Repeat Number
const burnintime = 100
const iters_num = 500
const it_num = 200
const iϵmax = 10
const num = 10000

# Network Params
const layer = [dimB+dimS, 72, 72, 72, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.0001f0

end
