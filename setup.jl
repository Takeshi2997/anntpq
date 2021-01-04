module Const

# System Size
const dimS = 16
const dimB = 80

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 100
const iters_num = 200
const it_num = 10000
const iϵmax = 8
const num = 10000

# Network Params
const layer = [dimB+dimS, 48, 48, 48, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.0001f0

end
