module Const

# System Size
const dimS = 8
const dimB = 40

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 100
const iters_num = 500
const it_num = 500
const iϵmax = 2
const num = 10000

# Network Params
const layer = [dimB, 40, 40, 2*dimS]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.001f0

end
