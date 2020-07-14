module Const

# System Size
const dimS = 8
const dimB = 120

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 1000
const it_num = 500
const iϵmax = 10
const num = 10000

# Network Params
const layer = [dimB, 56, 56, 56, 56, dimS]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 0.0001f0

end
