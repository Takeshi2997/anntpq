module Const

# System Size
const dimS = 8
const dimB = 80

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 20
const iters_num = 100
const it_num = 500
const iϵmax = 10
const num = 10000

# Network Params
const layer = [dimB+dimS, 88, 88, 88, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.01f0

end
