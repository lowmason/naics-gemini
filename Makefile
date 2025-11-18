.PHONY: setup lambda-setup install help check-uv

help:
	@echo "Available targets:"
	@echo "  make setup        - Configure git, zsh, oh-my-zsh, and development environment"
	@echo "  make lambda-setup - Same as setup, but optimized for Lambda Labs instances"
	@echo "  make install      - Install project dependencies with uv"
	@echo "  make check-uv     - Check if uv is installed, install if missing"

check-uv:
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.cargo/bin:$$PATH"; \
		echo "uv installed"; \
	else \
		echo "uv is already installed: $$(uv --version)"; \
	fi

setup: check-uv
	@echo "Running setup script..."
	@export PATH="$$HOME/.cargo/bin:$$PATH" && bash scripts/setup.sh

lambda-setup: check-uv
	@echo "Running Lambda Labs setup script..."
	@export PATH="$$HOME/.cargo/bin:$$PATH" && bash scripts/lambda-setup.sh

install: check-uv
	@echo "Installing dependencies..."
	@export PATH="$$HOME/.cargo/bin:$$PATH" && uv sync

