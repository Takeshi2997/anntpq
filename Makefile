JL          = ~/.julia/julia
BASE        = functions.jl setup.jl
CORE        = ml_core.jl
OBJS        = main.jl
INIT        = init.jl
CALC        = calculation.jl
MAIL        = ./mail.sh

main: $(BASE) $(CORE) $(OBJS) $(CALC) $(MAIL)
	$(JL) $(OBJS)
	cp ./datainit/params_at_001.bson ./data
	$(JL) $(CALC)
	$(MAIL)

init: $(BASE) $(INIT)
	$(JL) $(INIT)

clean:
	-rm -f *.txt *.png *.dat
	-rm -rf data error datainit errorinit
