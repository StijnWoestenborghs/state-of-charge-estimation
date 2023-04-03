setup: venv
	. .venv/Scripts/activate && python -m pip install --upgrade pip
	. .venv/Scripts/activate && pip install -r requirements.txt

venv:
	test -d .venv || python -m venv .venv

clean:
	rm -rf .venv

clean-pyc:
	find . -name "*.pyc" -exec rm -f {} + 
	find . -name "*.pyo" -exec rm -f {} +
	find . -name "*~" -exec rm -f {} +
	find . -name "__pycache__" -exec rm -fr {} +

train:
	. .venv/Scripts/activate && python -m src.main

test:
	. .venv/Scripts/activate && python -m src.test