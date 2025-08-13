import os
import subprocess

import numpy as np
import pandas as pd


def extract_efield(base_path, fiber_tract, streamline_number, stim_type, stim_location):
    """
    Extract E-Field

    Extracting the e-field based on stimulation type.

    Parameters:
        stim_type (str): Type of stimulation (TMS, MST, ECT).
        dicf (str): Path to the interpolated coordinate file.
        stim_location (str): Stimulation location for MST/ECT.

    Returns:
        None
    """

    coordinate_file = f"{base_path}/coordinate_path/{fiber_tract}_{streamline_number}.csv"
    efield_file = coordinate_file.replace('.csv', '_E.csv')

    if stim_type == "TMS":
        command = f"get_fields_at_coordinates -s '{coordinate_file}' -m '{base_path}/ernie_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh'"

    elif stim_type == "MST":
        mapping = {
            "Fz": f"{base_path}/mst_ernie_position_fz.msh",
            "Cz": f"{base_path}/mst_ernie_position_cz.msh"
        }
        command = f"get_fields_at_coordinates -s '{coordinate_file}' -m '{mapping[stim_location]}'"

    elif stim_type == "ECT":
        mapping = {
            "Bilateral": f"{base_path}/bl_ernie_TDCS_1_vn.msh",
            "Bifrontal": f"{base_path}/bf_ernie_TDCS_1_scalar.msh",
            "Right Unilateral": f"{base_path}/ru_ernie_TDCS_1_scalar.msh",
            "Testing": f"{base_path}/test_ernie_TDCS_1_scalar.msh"
        }
        command = f"get_fields_at_coordinates -s '{coordinate_file}' -m '{mapping[stim_location]}'"

    elif stim_type == "Uniform":
        coordinates = pd.read_csv(coordinate_file, header=None).to_numpy()
        efield = np.tile([1.0, 0.0, 0.0], (coordinates.shape[0], 1))
        np.savetxt(efield_file, efield, delimiter=",", fmt="%.6f")
        command = None

    if command:
        subprocess.run(command, shell=True, check=True)

    stim_desc = stim_type if stim_type != "Uniform" else "UniformField"
    if stim_type in ["MST", "ECT"]:
        stim_desc += f"_{stim_location.replace(' ', '')}"

    new_efield_file = coordinate_file.replace(
        '.csv',
        f"_{stim_desc}_E.csv"
    )

    if os.path.exists(efield_file) and efield_file != new_efield_file:
        os.rename(efield_file, new_efield_file)
