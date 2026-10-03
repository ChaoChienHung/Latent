.PHONY: help install run dev

help:
	@echo "Backend commands:"
	@echo "  make install   Install backend dependencies via uv or pip"
	@echo "  make run       Start the Flask API server on port 5000"
	@echo "  make dev       Start the Flask API server in debug mode"

install:
	uv sync || pip install -r requirements.txt

run:
	uv run python app.py

dev:
	FLASK_DEBUG=1 uv run python app.py
