#!/bin/bash

pip uninstall mmethane
rm -rf *.egg-info
rm -rf dist/
python3 -m build
python3 -m pip install --upgrade twine
twine upload dist/*