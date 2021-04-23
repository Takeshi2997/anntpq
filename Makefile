JL          = ~/julia/julia
OPTS        = "-t 8"
BASE        = functions.jl setup.jl ann.jl
CORE        = ml_core.jl
OBJS        = main.jl
CALC        = calculation.jl

main: $(BASE) $(CORE) $(OBJS) $(CALC)
	$(JL) $(OPTS) $(OBJS)

calc: $(BASE) $(CORE) $(CALC)
	$(JL) $(OPTS) $(CALC)

clean:
	-rm -f *.txt *.png *.dat nohup.out *.mem
	-rm -rf data error
