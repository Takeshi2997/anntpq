module Const

# System Size
const dimS = 8
const dimB = 64
const dim  = 2 * (dimS + dimB)

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 200
const it_num = 1000
const iϵmax = 20
const num = 10000

# Network Params
const layer = [dim, dim, dim, dim, dim, dim, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.0001f0

end
