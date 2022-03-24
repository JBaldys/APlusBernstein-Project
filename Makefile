
## output requirements.txt
deps:
	pip freeze > requirements.txt

## split data by subcategory
split-subcategory:
	python src/data/split_category.py
split-category:
	python src/data/split_category.py Category


pythonpath:
	export PYTHONPATH="${PYTHONPATH}:${pwd}"