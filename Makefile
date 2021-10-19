lint:
		pylint easyfsl

test:
		pytest easyfsl

dev-install:
		pip install -r dev_requirements.txt

soft-exp-clean:
		dvc exp gc -w
		dvc gc -w
