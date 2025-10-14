PROJ_VENV=$(CURDIR)/.venv

MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

.PHONY: help
help:
	@echo "Available targets:"
	@$(MAKE) -pRrq -f $(MAKEFILE_LIST) : 2>/dev/null |\
	  awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}'|\
	  egrep -v -e '^[^[:alnum:]]' -e '^$@' |\
	  sort |\
	  awk '{print "  " $$0}'

$(PROJ_VENV):
	uv venv "$@"

.PHONY: init clean
init: $(PROJ_VENV)
	uv sync

clean:
	rm -rf "$(PROJ_VENV)"

.PHONY: check check-py
check-py:
	uv run ruff check
	uv run ruff format --check
	uv run mypy .
check: check-py

.PHONY: format
format:
	uv run ruff format

.PHONY: fix
fix: format
	uv run ruff check --fix
