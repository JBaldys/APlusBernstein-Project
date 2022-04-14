
## output requirements.txt
deps:
	pip freeze > requirements.txt

## split data by subcategory
split_subcategory:
	python src/data/split_category.py
split_category:
	python src/data/split_category.py Category

summary:
	python src/data/summary_category.py

remove_missing:
	python src/data/remove_missing.py

pythonpath:
	export PYTHONPATH="${PYTHONPATH}:${pwd}"

regression_data:
	python src/data/make_model_data.py

classification_data:
	python src/data/make_model_data.py classification