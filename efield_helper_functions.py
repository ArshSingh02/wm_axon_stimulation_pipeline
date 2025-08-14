import numpy as np
import pandas as pd
import math
import vtk
import os
import subprocess


from filter_and_resample import (
    remove_coordinate_outliers,
    resample_coordinates_simnibs_resolution
)


def extract_efield_simnibs(base_path, coordinate_file, stim_type,
                           stim_location):
    """
    Extract E-Field

    Extracting the e-field based on stimulation type.

    Parameters
    ------
    base_path : str
        user's base directory
    coordinate_file : str
        path to the coordinate file.
    stim_type : str
        type of stimulation
    stim_location : str
        stimulation location

    Returns
    ------
    None
    """

    efield_file = coordinate_file.replace('.csv', '_E.csv')

    if stim_type == "TMS":
        command = (
            f"get_fields_at_coordinates -s '{coordinate_file}' "
            (
                f"-m '{base_path}/"
                "ernie_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh'"
            )
        )

    elif stim_type == "MST":
        mapping = {
            "Fz": f"{base_path}/mst_ernie_position_fz.msh",
            "Cz": f"{base_path}/mst_ernie_position_cz.msh"
        }
        command = (
            f"get_fields_at_coordinates -s '{coordinate_file}' "
            f"-m '{mapping[stim_location]}'"
        )

    elif stim_type == "ECT":
        mapping = {
            "Bilateral": f"{base_path}/bl_ernie_TDCS_1_vn.msh",
            "Bifrontal": f"{base_path}/bf_ernie_TDCS_1_scalar.msh",
            "Right Unilateral": f"{base_path}/ru_ernie_TDCS_1_scalar.msh",
            "Testing": f"{base_path}/test_ernie_TDCS_1_scalar.msh"
        }
        try:
            mesh_file = mapping[stim_location]
        except KeyError:
            raise ValueError(
                f"Invalid stim_location '{stim_location}' for ECT. "
                f"Valid options are: {list(mapping.keys())}"
            )
        command = (
            f"get_fields_at_coordinates -s '{coordinate_file}' "
            f"-m '{mesh_file}'"
        )

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
    ------
    coordinates : (N,3) array_like
        input polyline points (resolution on mm scale)
    efields : (N,3) array_like
        electric field vectors at each point (in V/m)

    Returns
    ------
    scalar_proj_efield : (N,1) ndarray
        scalar projection of e-field at point N (in V/m)
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
    ------
    coords : (N,3) array_like
        input polyline points (resolution on mm scale)
    scalar_proj_efield : (N,1) array_like
        scalar projection of e-field at each point (in V/m)

    Returns
    ------
    ec : (N,1) ndarray
        quasi-potentials at each point along the streamline (in mV)
    """
    ec = np.zeros(len(scalar_proj_efield))
    for i in range(1, len(scalar_proj_efield)):
        step_len = np.linalg.norm(coords[i] - coords[i-1])
        ec[i] = ec[i-1] - scalar_proj_efield[i-1] * step_len
    return ec


def interpolate_proj_efield(interp_arc, arc_uniform, scalar_proj_efield):
    """
    Interpolate projected e-field

    Interpolate the scalar projected e-field onto
    MRG resolution

    Parameters
    ----------
    interp_arc : (M,) array_like
        arc lengths at which to interpolate the scalar projected e-field (in µm)
    arc_uniform : (N,) array_like
        original arc lengths corresponding to the scalar projected e-field (in µm)
    scalar_proj_efield : (N,) array_like
        original scalar projected e-field values (in V/m)

    Returns
    -------
    scalar_proj_efield_interp : (M,) ndarray
        Interpolated scalar projected e-field values at interp_arc positions (in V/m)
    """
    scalar_proj_efield_interp = np.interp(interp_arc, arc_uniform,
                                          scalar_proj_efield)
    return scalar_proj_efield_interp


def interpolate_quasipotentials(interp_arc, arc_uniform,
                                ec_potentials_uniform):
    """
    Interpolate quasi-potentials

    Interpolate the quasi-potentials onto
    MRG resolution

    Parameters
    ----------
    interp_arc : (M,) array_like
        arc lengths at which to interpolate the quasi-potentials (in µm)
    arc_uniform : (N,) array_like
        original arc lengths corresponding to the quasi-potentials (in µm)
    ec_potentials_uniform : (N,) array_like
        original quasi-potential values (in mV)

    Returns
    -------
    ec_potentials_interp : (M,) ndarray
        interpolated quasi-potential values at interp_arc positions (in mV)
    """
    ec_potentials_interp = np.interp(interp_arc, arc_uniform,
                                     ec_potentials_uniform)
    return ec_potentials_interp


def calculate_activating_function(fiber):
    """
    Calculate activating function

    Calculate the activating function along the fiber
    based on the quasi-potentials at each Node of Ranvier.

    Parameters
    ----------
    fiber : PyFibers Fiber object
        fiber object containing sections and potentials.

    Returns
    -------
    activating_function : (N,) ndarray
        activating function values at each section.
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

    return activating_function


def streamline_extraction(base_path, head_model, fiber_tract, num_streamlines):
    """
    Extract Streamlines

    Create streamline coordinate .csv files

    Parameters
    ----------
    base_path : str
        user's base directory
    head_model : str
        patient ID
    fiber_tract : str
        white matter fiber tract
    num_streamlines : str
        total number of streamlines to extract

    Returns
    -------
    None
    """
    fiber_tract_file = (
        base_path + f'WM Fiber Tracts/{head_model}/{fiber_tract}.vtk'
    )

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fiber_tract_file)
    reader.Update()

    wm_fiber_tract_vtk = reader.GetOutput()

    for cell_id in range(num_streamlines):
        streamline = wm_fiber_tract_vtk.GetCell(cell_id)

        points_ids = streamline.GetPointIds()
        points = wm_fiber_tract_vtk.GetPoints()

        point_data = [points.GetPoint(points_ids.GetId(i)) for i
                      in range(points_ids.GetNumberOfIds())]

        end_terminal = math.sqrt(point_data[-1][0]**2 +
                                 point_data[-1][1]**2 + point_data[-1][2]**2)

        start_terminal = math.sqrt(point_data[0][0]**2 +
                                   point_data[0][1]**2 + point_data[0][2]**2)

        if end_terminal > start_terminal:
            point_data.reverse()

        streamline_number = cell_id + 1

        coordinate_directory = (
            base_path
            + f'/WM Fiber Tracts/{head_model}/{fiber_tract}/Coordinates/'
        )
        os.makedirs(coordinate_directory, exist_ok=True)
        coordinate_file = (
            coordinate_directory +
            f'{fiber_tract}_{streamline_number}_coordinates.csv'
        )

        pd.DataFrame(point_data).to_csv(
            coordinate_file, index=False, header=False
        )


def efield_extraction(base_path, head_model, fiber_tract, streamline_number,
                      stim_type, stim_location):
    """
    Extract e-fields

    Extract e-field for specified streamline

    Parameters
    ----------
    base_path : str
        user's base directory
    head_model : str
        patient ID
    fiber_tract : str
        white matter fiber tract
    streamline_number : str
        streamline ID in fiber tract
    stim_type : str
        type of stimulation
    stim_location : str
        placement of electrode/coil

    Returns
    -------
    None
    """
    coordinate_directory = (
        base_path + f'/WM Fiber Tracts/{head_model}/{fiber_tract}/Coordinates/'
    )
    os.makedirs(coordinate_directory, exist_ok=True)
    coordinate_file = (
        coordinate_directory +
        f'{fiber_tract}_{streamline_number}_coordinates.csv'
    )

    original_coordinates = pd.read_csv(
        coordinate_file, header=None
    ).to_numpy()

    filtered_coordinates = remove_coordinate_outliers(
        original_coordinates=original_coordinates,
        min_len=0.02,
        mad_k=3.5
    )
    resampled_coordinates = resample_coordinates_simnibs_resolution(
        filtered_coordinates=filtered_coordinates,
        spacing=2,
        include_end=False,
        atol=1e-12
    )

    coordinate_file = (
        f"{base_path}/coordinate_path/"
        f"{fiber_tract}_{streamline_number}.csv"
    )
    np.savetxt(
        coordinate_file,
        resampled_coordinates,
        delimiter=",",
        fmt="%.6f"
    )

    extract_efield_simnibs(
        base_path,
        coordinate_file,
        stim_type,
        stim_location
    )
