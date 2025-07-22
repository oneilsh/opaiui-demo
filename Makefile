.PHONY: install demo

install:
	@echo "Installing dependencies..."
	poetry install --no-root

demo:
	@echo "Running demo..."
	poetry run streamlit run demo_app.py