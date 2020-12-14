module Const

# System Size
const dimS = 8
const dimB = 72

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 500
const it_num = 1000
const iϵmax = 10
const num = 10000

# Network Params
const layer = [dimB+dimS, 80, 80, 80, 80]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.0001f0

end
