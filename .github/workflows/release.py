name: publish new version to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    strategy:
      max-parallel: 1
      matrix:
        package: ["test_validation"]

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install setuptools setuptools_scm wheel

      - name: Build packages
        run: python setup.py sdist bdist_wheel

      - name: Publish package
        uses: pypa/gh-action-pypi-publish@master
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
          packages_dir: dist/
