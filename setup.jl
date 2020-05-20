module Const

# System Size
const dimS = 8
const dimB = 56

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 100
const iters_num = 200
const it_num = 500
const iϵmax = 1
const num = 1000

# Network Params
const layer = [dimB, 56, 56, 2*dimS]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.000001f0

end
