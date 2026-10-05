# Every command this repository needs, in one place.
#
#     make help
#
# The whole experiment is src/Program.cs: the task, the model, the training loop
# and the command line. TorchSharp carries the autograd, so there are no
# derivatives to read here. `make` alone compiles and checks the style.
#
# Needs the .NET SDK 10 or newer. The first build downloads libtorch, which is a
# few hundred megabytes.

DOTNET ?= dotnet
PAIR   ?= 12+35
NOTE   ?= with the saved model
ARGS   ?=

.DEFAULT_GOAL := validate

.PHONY: help build lint format validate run control save infer explain clean publish status

help:
	@echo make build      compile in Release
	@echo make validate   build and check the formatting
	@echo make run        the full run: 12,000 steps, about half a minute
	@echo make control    the run without weight decay: the rule never arrives
	@echo make save       train and keep the model in model.pt
	@echo make infer      ask the saved model, PAIR=12+35 by default
	@echo make explain    draw every answer it considered for one sum
	@echo make clean      remove the build output
	@echo make publish MESSAGE=what changed   validate, commit and push
	@echo Extra flags go through ARGS, as in: make run ARGS=--p 13 --steps 3000

build:
	$(DOTNET) build -c Release

# dotnet format is the linter of a .NET repository: it checks the style of the
# whole project. `make format` writes the files instead of checking them.
lint:
	$(DOTNET) format --verify-no-changes

format:
	$(DOTNET) format

validate: build lint

run: build
	$(DOTNET) run -c Release --no-build -- $(ARGS)

# The same run with the decay set to zero. It memorizes the train set and never
# learns the rule, and five times the steps do not change that. It writes over the
# artifacts of `make run`, so the recipe puts the committed ones back when it
# finishes: the log is the result, and the table of the two runs is in the README.
control: build
	$(DOTNET) run -c Release --no-build -- --wd 0 $(ARGS)
	git checkout -- runs figures data

save: build
	$(DOTNET) run -c Release --no-build -- --save

# A framework keeps the whole state dictionary, so saving and loading are one line
# each and there is nothing to keep in step by hand.
infer:
	$(DOTNET) run -c Release -- --infer $(PAIR)

# A model does not answer, it spreads a chance over the options. This writes the
# 53 probabilities for one sum to runs/probabilities.csv and draws them.
explain:
	$(DOTNET) run -c Release -- --explain $(PAIR) --note "$(NOTE)"

clean:
	$(DOTNET) clean

publish: validate
	git add -A
	git commit -m "$(MESSAGE)"
	git push

status:
	git status --short
	git branch --show-current
