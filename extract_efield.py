import numpy as np
import pandas as pd
import math
import vtk
import os


from filter_and_resample import (
    remove_coordinate_outliers,
    resample_coordinates_simnibs_resolution
)
from efield_helper_functions import streamline_extraction, efield_extraction



