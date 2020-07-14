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
const layer1 = [dimB, 56, 56, 56, 56]
const layers1_num = length(layer1) - 1
const layer2 = [dimS, 56, 56]
const layers2_num = length(layer2) - 1

# Learning Rate
const lr = 0.0001f0

end
