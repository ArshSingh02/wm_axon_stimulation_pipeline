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


def calculate_projected_efield(coordinates, efields):
    """Calcuate scalar projection of e-field

    Calculates the scalar projection of the e-field
    vectors onto the tangent direction of the path
    of the streamline.

    Parameters
    ----------
    coordinates : (N,3) array_like
        Input polyline points in mm.
    efields : (N,3) array_like
        Electric field vectors at each point.

    Returns
    -------
    scalar_proj_efield : (N,1) ndarray
        Scalar projection of e-field at point N.
    """
    directions = np.diff(coordinates, axis=0)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    scalar_proj_efield = np.sum(efields[:-1] * directions, axis=1)
    scalar_proj_efield = np.append(scalar_proj_efield, scalar_proj_efield[-1])
    return scalar_proj_efield


def calculate_quasipotentials(coords, scalar_proj_efield):
    """
    Calculate the quasi-potentials

    Calculates the quasi-potentials along a streamline
    given the scalar projection of the e-field
    at each point.

    Parameters
    ----------
    coords : (N,3) array_like
        Input polyline points in mm.
    scalar_proj_efield : (N,1) array_like
        Scalar projection of e-field at each point.

    Returns
    -------
    ec : (N,1) ndarray
        Quasi-potentials at each point along the streamline.
    """
    ec = np.zeros(len(scalar_proj_efield))
    for i in range(1, len(scalar_proj_efield)):
        step_len = np.linalg.norm(coords[i] - coords[i-1]) / 1000.0
        ec[i] = ec[i-1] - scalar_proj_efield[i-1] * step_len


def interpolate_proj_efield(interp_arc, arc_uniform, scalar_proj_efield):
    """
    Interpolate projected e-field
    
    Interpolate the scalar projected e-field onto
    MRG resolution

    Parameters
    ----------
    interp_arc : (M,) array_like
        Arc lengths at which to interpolate the scalar projected e-field.
    arc_uniform : (N,) array_like
        Original arc lengths corresponding to the scalar projected e-field.
    scalar_proj_efield : (N,) array_like
        Original scalar projected e-field values.

    Returns
    -------
    scalar_proj_efield_interp : (M,) ndarray
        Interpolated scalar projected e-field values at interp_arc positions.
    """
    scalar_proj_efield_interp = np.interp(interp_arc, arc_uniform, scalar_proj_efield)
    return scalar_proj_efield_interp


def interpolate_quasipotentials(interp_arc, arc_uniform, ec_potentials_uniform):
    """
    Interpolate quasi-potentials

    Interpolate the quasi-potentials onto
    MRG resolution

    Parameters
    ----------
    interp_arc : (M,) array_like
        Arc lengths at which to interpolate the quasi-potentials.
    arc_uniform : (N,) array_like
        Original arc lengths corresponding to the quasi-potentials.
    ec_potentials_uniform : (N,) array_like
        Original quasi-potential values.

    Returns
    -------
    ec_potentials_interp : (M,) ndarray
        Interpolated quasi-potential values at interp_arc positions.
    """
    ec_potentials_interp = np.interp(interp_arc, arc_uniform, ec_potentials_uniform)
    return ec_potentials_interp


def calculate_activating_function(fiber):
    """
    Calculate activating function

    Calculate the activating function along the fiber
    based on the quasi-potentials at each Node of Ranvier.

    Parameters
    ----------
    fiber : Fiber object
        Fiber object containing sections and potentials.

    Returns
    -------
    activating_function : (N,) ndarray
        Activating function values at each section.
    """
    activating_function = np.zeros(len(fiber.sections), dtype=float)

    for n in range(len(fiber.sections)):

        if n % 11 != 0:
            continue

        if n - 11 < 0 or n + 11 >= len(fiber.sections):
            continue

        Ve_node = fiber.potentials[n]
        Ve_prevnode = fiber.potentials[n - 11]
        Ve_nextnode = fiber.potentials[n + 11]

        Ri_prev = (
            0.5 * fiber.sections[n - 11].L * fiber.sections[n - 11].Ra
            + fiber.sections[n - 10].L * fiber.sections[n - 10].Ra
            + fiber.sections[n - 9].L * fiber.sections[n - 9].Ra
            + sum(sec.L * sec.Ra for sec in fiber.sections[n - 8:n - 2])
            + fiber.sections[n - 2].L * fiber.sections[n - 2].Ra
            + fiber.sections[n - 1].L * fiber.sections[n - 1].Ra
            + 0.5 * fiber.sections[n].L * fiber.sections[n].Ra

        )

        Ri_next = (
            0.5 * fiber.sections[n].L * fiber.sections[n].Ra
            + fiber.sections[n + 1].L * fiber.sections[n + 1].Ra
            + fiber.sections[n + 2].L * fiber.sections[n + 2].Ra
            + sum(sec.L * sec.Ra for sec in fiber.sections[n + 3:n + 9])
            + fiber.sections[n + 9].L * fiber.sections[n + 9].Ra
            + fiber.sections[n + 10].L * fiber.sections[n + 10].Ra
            + 0.5 * fiber.sections[n + 11].L * fiber.sections[n + 11].Ra

        )

        I_prev = (Ve_prevnode - Ve_node) / (Ri_prev)
        I_next = (Ve_node - Ve_nextnode) / (Ri_next)

        I_m = I_prev - I_next

        activating_function[n] = I_m
