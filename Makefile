JL          = ~/.julia/julia
BASE        = functions.jl setup.jl ann.jl
CORE        = ml_core.jl
OBJS        = main.jl
INIT        = init.jl
CALC        = calculation.jl

main: $(BASE) $(CORE) $(OBJS) 
	$(JL) $(OBJS)

calc: $(BASE) $(CORE) $(CALC)
	cp ./datainit/params_at_001.bson ./data
	$(JL) $(CALC)
	sudo shutdown -h now

init: $(BASE) $(INIT)
	$(JL) $(INIT)

clean:
	-rm -f *.txt *.png *.dat nohup.out
	-rm -rf data error datainit errorinit
