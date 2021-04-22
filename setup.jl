module Const

# System Size
const dimS = 8
const dimB = 56

# System Param
const t = 1f0
const J = 1f0
const λ = 1f0

# Repeat Number
const burnintime = 10
const iters_num = 200
const it_num = 100
const iϵmax = 1

# Network Params
const layer = [dimB+dimS, 32]
const layers_num = length(layer) - 1
const networkdim = layer[2] * (layer[1] + 1) + layer[1]

# Learning Rate
const lr = 1f-2
const dt = 1f-4
const batchsize = 24

end
