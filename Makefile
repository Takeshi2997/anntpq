JL          = ~/.julia/julia
BASE        = functions.jl setup.jl
CORE        = ml_core.jl
OBJS        = main.jl
INIT        = init.jl
CALC        = calculation.jl

main: $(BASE) $(CORE) $(OBJS) $(CALC)
	$(JL) $(INIT)
	$(JL) $(OBJS)
	cp ./datainit/params_at_001.bson ./data
	$(JL) $(CALC)
	sudo shutdown -h now

init: $(BASE) $(INIT)
	$(JL) $(INIT)
	sudo shutdown -h now

clean:
	-rm -f *.txt *.png *.dat
	-rm -rf data error datainit errorinit
