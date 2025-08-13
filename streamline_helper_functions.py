import os
import pandas as pd

from filter_and_resample import (
    compute_delta_z,
    compute_effective_streamline_length,
    compute_cumulative_distances,
    compute_n_sections,
    interpolate_fiber_variable_centers
)

from efield_helper_functions import (
    calculate_projected_efield,
    calculate_quasipotentials,
    interpolate_proj_efield,
    interpolate_quasipotentials,
    calculate_activating_function
)

from pyfibers import build_fiber, FiberModel, ScaledStim
from waveforms import select_waveform


def calculate_fiber_parameters(base_path, head_model, fiber_tract,
                               streamline_number, diameter,
                               stim_type, stim_location):

    file_directory = (
        base_path + f'/WM Fiber Tracts/{head_model}/{fiber_tract}/Coordinates/'
    )
    os.makedirs(file_directory, exist_ok=True)
    coordinate_file = (
        file_directory +
        f'{fiber_tract}_{streamline_number}_coordinates.csv'
    )

    stim_desc = stim_type if stim_type != "Uniform" else "UniformField"
    if stim_type in ["MST", "ECT"]:
        stim_desc += f"_{stim_location.replace(' ', '')}"

    efield_file = coordinate_file.replace(
        '.csv',
        f"_{stim_desc}_E.csv"
    )

    coordinates_uniform_resolution = pd.read_csv(
        coordinate_file, header=None
    ).to_numpy()

    efield_uniform_resolution = pd.read_csv(
        efield_file, header=None
    ).to_numpy()

    # Calculate Effective Streamline Length and Number of Sections
    cumulative_distances = compute_cumulative_distances(
        coordinates_uniform_resolution)
    streamline_length = cumulative_distances[-1] * 1000  # Convert to microns
    delta_z = compute_delta_z(diameter)
    n_sections = compute_n_sections(streamline_length, delta_z)

    scalar_proj_efield = calculate_projected_efield(
        coordinates_uniform_resolution,
        efield_uniform_resolution)

    quasipotentials = calculate_quasipotentials(
        coordinates_uniform_resolution,
        scalar_proj_efield
    )

    coords_mrg_resolution, interp_arc = interpolate_fiber_variable_centers(
        coordinates_uniform_resolution, diameter, n_sections=n_sections)
    arc_uniform = compute_cumulative_distances(coordinates_uniform_resolution)

    scalar_proj_efield_interp = interpolate_proj_efield(
        interp_arc, arc_uniform, scalar_proj_efield)
    quasipotentials_interp = interpolate_quasipotentials(
        interp_arc, arc_uniform, quasipotentials)

    return (
        n_sections,
        quasipotentials_interp,
        scalar_proj_efield_interp,
        coords_mrg_resolution
    )


def create_fiber(diameter, n_sections, quasipotentials_interp,
                 scalar_proj_efield_interp, coords_mrg_resolution):

    fiber = build_fiber(FiberModel.MRG_INTERPOLATION,
                        diameter=diameter, n_sections=n_sections)
    fiber.potentials = quasipotentials_interp[0:len(fiber.sections)] * 1000

    activating_function = calculate_activating_function(fiber)

    return fiber
