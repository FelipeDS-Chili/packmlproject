# packmlproject - Python package builder

Copyright (C)  Le Wagon Staff

This is a full machine learning project generator, based on packgenlite from Le Wagon. It contains some modifications in MLOps code and aditional files ready to use in your next DS project. It provides python libs for projects and mainly `packmlproject` script.



`packmlproject` create a Python package template.

## Install `packmlproject`
```bash
pip install git+https://github.com/FelipeDS-Chili/packmlproject.git
```

## Create a `new_project` package

Use `packmlproject` to create a new python package:

```bash
packmlproject new_project
```

The tree of the new package created is:

```bash
cd new_project
tree
.
├── MANIFEST.in
├── Dockerfile
├── Procfile
├── Makefile
├── README.md
├── new_project
│   ├── __init__.py
│   ├── data
│   ├── data.py
│   ├── encoders.py
│   ├── gcp.py
│   ├── trainer.py
│   └── utils.py
├── api
│   ├── __init__.py
│   ├── app.py
│   ├── app_streamlit.py
│   └── fast.py
├── notebooks
├── raw_data
├── requirements.txt
├── scripts
│   └── new_project-run
├── setup.py
├── setup.sh
└── tests
    ├── __init__.py
    └── data_test.py

6 directories, 21 files
```
