module Const

# System Size
const dimS = 4
const dimB = 12

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 100
const iters_num = 200
const it_num = 1000
const iϵmax = 8
const num = 10000

# Network Params
const layer = [dimB+dimS, 8, 8, 8, 8, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.0001f0

end
