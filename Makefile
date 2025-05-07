.PHONY: build train game

build:
	python cpp_setup.py build_ext --inplace

train:
	python train.py

game:
	python game.py