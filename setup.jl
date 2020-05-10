module Const

# System Size
const dimS = 8
const dimB = 56

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 100
const iters_num = 500
const it_num = 100
const iϵmax = 10
const num = 10000

# Network Params
const layer = [dimB+dimS, 64, 64, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.0001f0

end
