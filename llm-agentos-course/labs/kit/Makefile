# Shortcuts for the AgentOS lab kit. Everything also runs as plain `python3 ...` (see README.md).
PY ?= python3
LAB ?= 0
TRIALS ?= 3
NAME ?= latest

.PHONY: grade test-kit demo suite clean

grade:          ## run one lab's autograder: make grade LAB=0
	$(PY) grade.py $(LAB)

test-kit:       ## the kit's own tests
	$(PY) grade.py kit

demo:           ## offline run of the basic tier with each task's scripted solution
	$(PY) run_suite.py --model demo --out runs/demo

suite:          ## run the basic tier on your model endpoint: make suite TRIALS=3 NAME=baseline
	$(PY) run_suite.py --model api --trials $(TRIALS) --out runs/$(NAME)

clean:
	rm -rf runs
