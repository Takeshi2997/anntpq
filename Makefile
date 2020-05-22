JL          = ~/.julia/julia
BASE        = functions.jl setup.jl ann.jl
CORE        = ml_core.jl
OBJS        = main.jl
CALC        = calculation.jl

main: $(BASE) $(CORE) $(OBJS) $(CALC)
	$(JL) $(OBJS)
	$(JL) $(CALC)
	sudo shutdown -h now

calc: $(BASE) $(CORE) $(CALC)
	$(JL) $(CALC)

clean:
	-rm -f *.txt *.png *.dat nohup.out
	-rm -rf data error datainit errorinit
