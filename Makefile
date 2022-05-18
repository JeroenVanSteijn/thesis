install:
	pip3 install -r requirements.txt

data:
	python3 instances/generate_instances.py

run:
	python3 src/main.py

plot:
	python3 src/plot.py

