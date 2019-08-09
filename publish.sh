#!/usr/bin/env bash

# fail on any error
set -e
echo "Doing $1 update"

# use python 3.7 virtualenv
source ~/venv/py3.7/bin/activate

# check formatting
# flake8 depgrep/* tests/* setup.py
# black depgrep/* tests/* setup.py --check
# isort -m 3 -tc -c depgrep/* tests/* setup.py

# run tests
# python -m unittest

# remove old releases
rm -r -f build dist

# bump the version
bump2version $1

# make new releases
python setup.py bdist_egg sdist

# upload
twine upload dist/*

# push to github
git push origin master --follow-tags
