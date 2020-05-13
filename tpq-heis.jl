using HPhiJulia

L = 4
J = 1
results = HPhiJulia.HPhi("Spin","chain",L,J=J,mpinum=1)
println(results["Energy"])
