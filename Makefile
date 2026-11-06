# Shortcuts for the common commands. Variables: TYPE (robot type, from
# `make list`), ARCHIVE (archive.pkl; default: the one train.py writes),
# WORKERS, BATCHES.
PY ?= python
ARCHIVE ?=
TYPE ?=
WORKERS ?= 3
BATCHES ?= 30
A := $(if $(ARCHIVE),--archive $(ARCHIVE))

.PHONY: help install test train list replay validate export docs docker

help:
	@echo "make install | test | train | list | replay TYPE=.. | validate TYPE=.. | export TYPE=.. | docs | docker"

install:
	$(PY) -m pip install -e ".[video,cad]" pytest

test:
	$(PY) -m pytest tests -q

train:
	$(PY) experiments/train.py --terrains terrain --batches $(BATCHES) --workers $(WORKERS)

list:
	$(PY) experiments/replay.py $(A) --list

replay:
	$(PY) experiments/replay.py $(A) --type $(TYPE)

validate:
	$(PY) experiments/validate.py $(A) --type $(TYPE)

export:
	$(PY) experiments/export.py $(A) --type $(TYPE)

docs:
	mkdocs serve

docker:
	docker build -t darwinbot .
