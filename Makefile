JL          = ~/julia/julia
OPTS        = "-t 8"
OBJS        = main.jl
CALC        = calculation.jl

main: $(OBJS)
	$(JL) $(OPTS) $(OBJS)

calc: $(CORE) $(CALC)
	$(JL) $(CALC)

clean:
	-rm -f *.txt *.png *.dat nohup.out *.mem
	-rm -rf data error
