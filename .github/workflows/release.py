name: release
on:
  release:
    types: [published]
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      # For pypi trusted publishing
      id-token: write
    strategy:
      max-parallel: 1
      matrix:
        package: ["pymatgen-io-validation"]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install setuptools setuptools_scm wheel
      - name: Build packages
        run: python setup.py sdist bdist_wheel
      - name: Publish package
        uses: pypa/gh-action-pypi-publish@master
        with:
          # user: __token__
          # password: ${{ secrets.PYPI_API_TOKEN }}
          skip-existing: true
          verbose: true
          packages_dir: dist/





# name: publish new version to PyPI

# on: release
#   # release:
#     # types: [published]

# jobs:
#   deploy:
#     runs-on: ubuntu-latest
#     permissions:
#       # For pypi trusted publishing
#       id-token: write
#     strategy:
#       max-parallel: 1
#       matrix:
#         package: ["test_validation"]

#     steps:
#       - uses: actions/checkout@v4

#       - uses: actions/setup-python@v5
#         with:
#           python-version: 3.11

#       - name: Install dependencies
#         run: |
#           python -m pip install --upgrade pip
#           pip install setuptools setuptools_scm wheel

#       - name: Build packages
#         run: python setup.py sdist bdist_wheel

#       - name: Publish package
#         uses: pypa/gh-action-pypi-publish@master
#         with:
#           # user: __token__
#           # password: ${{ secrets.PYPI_API_TOKEN }}
#           skip-existing: true
#           verbose: true
#           packages_dir: dist/
