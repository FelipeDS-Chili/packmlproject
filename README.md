# packmlproject - Python package builder

This package is a meta-package that provide python libs for projects
and mainly `packmlproject` script.

`packmlproject` create a Python package template.

## Install `packmlproject`
```bash
pip install git+https://github.com/krokrob/packmlproject.git
```

## Create a `newpkgname` package

Use `packmlproject` to create a new python package:

```bash
packmlproject newpkgname
```

Check that the package has been created:

```bash
cd newpkgname
tree
.
├── MANIFEST.in
├── Makefile
├── README.md
├── newpkgname
│   ├── __init__.py
│   └── data
├── notebooks
├── raw_data
├── requirements.txt
├── scripts
│   └── newpkgname-run
├── setup.py
└── tests
    └── __init__.py

6 directories, 8 files
```
