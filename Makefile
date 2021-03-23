.PHONY: clean build gen-docs gen-docs-full docs prepare-release update-from-cookiecutter help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean:  ## clean all build, python, and testing files
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +
	find . -name '*.pyc' -exec rm -fr {} +
	find . -name '*.pyo' -exec rm -fr {} +
	find . -name '*~' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .tox/
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache

build: ## run tox / run tests and lint
	tox

gen-docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/aicsimageio*.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ aicsimageio **/tests/
	sphinx-autogen docs/index.rst
	python docs/_fix_internal_links.py --current-suffix .md --target-suffix .html
	$(MAKE) -C docs html
	python docs/_fix_internal_links.py --current-suffix .html --target-suffix .md

gen-docs-full: ## generate Sphinx docs + benchmark docs
	make gen-docs
	asv publish
	cp -r .asv/html/ docs/_build/html/_benchmarks

docs: ## generate Sphinx HTML documentation, including API docs, and serve to browser
	make gen-docs
	$(BROWSER) docs/_build/html/index.html

prepare-release: ## Checkout main, generate new section of changelog
	bumpversion dev_release
	gitchangelog
	git add docs/CHANGELOG.rst
	git commit -m "Update changelog"
	git reset --soft HEAD~1
	git commit --amend  # This and the line above squash the "Update changelog" and bumpversion commits together

update-from-cookiecutter: ## update this repo using latest cookiecutter-pypackage
	cookiecutter gh:AllenCellModeling/cookiecutter-pypackage --config-file cookiecutter.yaml --no-input --overwrite-if-exists --output-dir ..