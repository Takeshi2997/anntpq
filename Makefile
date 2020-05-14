JL          = ~/.julia/julia -p 4
BASE        = functions.jl setup.jl ann.jl
CORE        = ml_core.jl
OBJS        = main.jl
INIT        = init.jl
CALC        = calculation.jl

main: $(BASE) $(CORE) $(OBJS) $(CALC) $(INIT)
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
