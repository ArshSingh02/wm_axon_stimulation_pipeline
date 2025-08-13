import os
import numpy as np
import pandas as pd
import csv

from filter_and_resample import (
    compute_delta_z,
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

from visualizations_helper_functions import (
    create_vtk_points,
    create_vtk_scalars,
    save_vtk_file,
    process_fiber_ap
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
                 scalar_proj_efield_interp, coords_mrg_resolution,
                 base_path, streamline_number,
                 stim_type, stim_location, head_model, fiber_tract):

    fiber = build_fiber(FiberModel.MRG_INTERPOLATION,
                        diameter=diameter, n_sections=n_sections)
    fiber.potentials = quasipotentials_interp[0:len(fiber.sections)] * 1000

    activating_function = calculate_activating_function(fiber)

    file_directory = os.path.join(
        base_path,
        f'{stim_type} Results/{stim_location}/{head_model}/Visualizations/'
        f'{fiber_tract}'
    )

    os.makedirs(file_directory, exist_ok=True)

    diam = diameter
    output_vtk_file = os.path.join(
        file_directory,
        f"{diam}_{fiber_tract}_{streamline_number}_{stim_type}_{stim_location}"
    )

    num_points = len(fiber.sections)

    vtk_coords = create_vtk_points(
        coords_mrg_resolution[:num_points]
    )

    vtk_proj_efield = create_vtk_scalars(
        scalar_proj_efield_interp[:num_points],
        name="Projected E-Field"
    )
    vtk_potentials = create_vtk_scalars(
        quasipotentials_interp[:num_points],
        name="EC Potentials"
    )
    vtk_af = create_vtk_scalars(
        activating_function[:num_points],
        name="Activating Function"
    )
    save_vtk_file(
        output_vtk_file,
        vtk_coords,
        vtk_proj_efield,
        vtk_potentials,
        vtk_af
    )
    save_vtk_file(output_vtk_file, vtk_coords, vtk_proj_efield,
                  vtk_potentials, vtk_af)

    return fiber


def find_streamline_threshold(
            fiber, diameter, base_path, streamline_number, pulse_width,
            stim_type, stim_location, head_model, fiber_tract,
            stimamp_bottom, stimamp_top,
            coords_mrg_resolution):

    results_directory = os.path.join(
        base_path,
        f'{stim_type} Results/{stim_location}/{head_model}/{str(pulse_width)}'
    )

    thresholds_file = os.path.join(
        results_directory,
        f"{fiber_tract}_{diameter}microns_"
        f"{stim_type}_{stim_location}_{str(pulse_width)}ms.csv"
    )

    waveform_func, simulation_time_step, simulation_duration = select_waveform(
        stim_type, pulse_width
    )
    stimulation = ScaledStim(
        waveform=waveform_func,
        dt=simulation_time_step,
        tstop=simulation_duration
    )
    stimamp, _ = stimulation.find_threshold(
        fiber,
        stimamp_bottom=stimamp_bottom,
        stimamp_top=stimamp_top,
        condition="activation"
    )

    with open(thresholds_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([streamline_number, stimamp])

    amp, _ = stimulation.run_sim(stimamp, fiber)

    activation_map_directory = os.path.join(
        base_path,
        (
            f'{stim_type} Results/{stim_location}/{head_model}/'
            f'{str(pulse_width)}/Activation Mapping/Threshold'
        )
    )
    os.makedirs(activation_map_directory, exist_ok=True)

    streamline_activation_file = os.path.join(
        activation_map_directory,
        f"{fiber_tract}_{diameter}microns_{streamline_number}_"
        f"{stim_type}_{stim_location}_{str(pulse_width)}ms.vtk"
    )

    process_fiber_ap(fiber, coords_mrg_resolution,
                     streamline_activation_file)


def stimulate_streamline(
            fiber, diameter, base_path, streamline_number, pulse_width,
            stim_type, stim_location, head_model, fiber_tract,
            stimamp, coords_mrg_resolution):

    percent_mso_amp = str(int(np.round(stimamp * 100)))

    activation_map_directory = os.path.join(
        base_path,
        (
            f'{stim_type} Results/{stim_location}/{head_model}/'
            f'{str(pulse_width)}/Activation Mapping/'
            f'{percent_mso_amp} % MSO'
        )
    )
    os.makedirs(activation_map_directory, exist_ok=True)

    waveform_func, simulation_time_step, simulation_duration = select_waveform(
        stim_type, pulse_width
    )
    stimulation = ScaledStim(
        waveform=waveform_func,
        dt=simulation_time_step,
        tstop=simulation_duration
    )

    fiber.record_vm()
    fiber.record_gating()
    fiber.record_im()
    amp, _ = stimulation.run_sim(stimamp, fiber)

    streamline_activation_file = os.path.join(
        activation_map_directory,
        f"{fiber_tract}_{diameter}microns_{streamline_number}_"
        f"{stim_type}_{stim_location}_{str(pulse_width)}ms.vtk"
    )

    process_fiber_ap(fiber, coords_mrg_resolution,
                     streamline_activation_file)
