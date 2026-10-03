.PHONY: help install install-backend install-frontend backend frontend dev build lint

help:
	@echo "Latent development commands:"
	@echo "  make install           Install all dependencies (backend + frontend)"
	@echo "  make install-backend   Install backend Python dependencies"
	@echo "  make install-frontend  Install frontend npm dependencies"
	@echo "  make backend           Start Flask API backend on port 5000"
	@echo "  make frontend          Start React/Vite frontend on port 5173"
	@echo "  make dev               Start backend and frontend concurrently"
	@echo "  make build             Build frontend production bundle"
	@echo "  make lint              Run frontend linter"

install: install-backend install-frontend

install-backend:
	cd backend && (uv sync || pip install -r requirements.txt)

install-frontend:
	cd frontend && npm install

backend:
	cd backend && (uv run python app.py || python app.py)

frontend:
	cd frontend && npm run dev

dev:
	@trap 'kill 0' INT TERM EXIT; \
	$(MAKE) backend & \
	$(MAKE) frontend & \
	wait

build:
	cd frontend && npm run build

lint:
	cd frontend && npm run lint
