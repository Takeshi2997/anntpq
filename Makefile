JL          = ~/.julia/julia
BASE        = functions.jl setup.jl ann.jl
CORE        = ml_core.jl
OBJS        = main.jl
INIT        = init.jl
CALC        = calculation.jl

main: $(BASE) $(CORE) $(OBJS) $(INIT) $(CALC)
	$(JL) $(INIT)
	$(JL) $(OBJS)
	cp ./datainit/params_at_001.bson ./data
	$(JL) $(CALC)
	sudo shutdown -h now

calc: $(BASE) $(CORE) $(CALC)
	$(JL) $(CALC)

init: $(BASE) $(INIT)
	$(JL) $(INIT)

clean:
	-rm -f *.txt *.png *.dat nohup.out
	-rm -rf data error datainit errorinit
