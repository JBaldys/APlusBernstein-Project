
## output requirements.txt
deps:
	pip freeze > requirements.txt

## split data by subcategory
split-subcategory:
	python src/data/split_category.py
split-category:
	python src/data/split_category.py Category

summary:
	python src/data/summary_category.py

remove_missing:
	python src/data/remove_missing.py

pythonpath:
	export PYTHONPATH="${PYTHONPATH}:${pwd}"