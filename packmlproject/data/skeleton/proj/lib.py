# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for proj Project
"""

from os.path import split
import pandas as pd
import datetime

pd.set_option('display.width', 200)


def clean_data(data):
    """ clean data
    """
    pass


if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    import proj
    folder_source, _ = split(proj.__file__)
    df = pd.read_csv('{}/data/data.csv.gz'.format(folder_source))
    clean_data = clean_data(df)
    print(' dataframe cleaned')
