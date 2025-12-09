.PHONY: run build up down clean

run:
	python -m streamlit run src/app.py

build:
	docker build -t pdf-classifier:latest .

up:
	docker-compose up --build

down:
	docker-compose down

clean:
	docker rmi pdf-classifier:latest || true
