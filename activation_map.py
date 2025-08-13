from multiprocessing import Pool
import argparse
import os
import numpy as np

from streamline_helper_functions import stimulate_streamline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run simulations for white matter fiber tracts."
    )
    parser.add_argument(
        "--base_path", required=True, type=str,
        help="User Base Directory"
    )
    parser.add_argument(
        "--head_model", required=True, type=str,
        help="Patient ID"
    )
    parser.add_argument(
        "--stim_type", required=True, type=str,
        help="Type of Stimulation"
    )
    parser.add_argument(
        "--stim_location", required=True, type=str,
        help="Stimulation Location"
    )
    parser.add_argument(
        "--diameter", required=True, type=float,
        help="Streamline Diameter"
    )
    parser.add_argument(
        "--pulse_width", required=True, type=float,
        help="Pulse Width"
    )
    parser.add_argument(
        "--stimamp", required=True, type=float,
        help="Stimulus Amplitude (in terms of % MSO)"
    )

    args = parser.parse_args()

    fiber_tract_directory = os.path.join(
        args.base_path,
        f'WM Fiber Tracts/{args.head_model}'
    )

    fiber_tracts = [
        f.replace('.vtk', '')
        for f in os.listdir(fiber_tract_directory)
        if f.endswith('.vtk')
    ]

    num_CPUs = 50

    for fiber_tract in fiber_tracts:

        results_directory = os.path.join(
            args.base_path,
            f"{args.stim_type} Results",
            args.stim_location,
            args.head_model,
            str(args.pulse_width)
        )

        percent_mso_amp = str(int(np.round(args.stimamp * 100)))

        activation_map_directory = os.path.join(
            args.base_path,
            (
                f'{args.stim_type} Results/{args.stim_location}/'
                f'{args.head_model}/'
                f'{str(args.pulse_width)}/Activation Mapping/'
                f'{percent_mso_amp} % MSO'
            )
        )
        os.makedirs(activation_map_directory, exist_ok=True)

        streamline_stimulation_arguments = [
            (
                args.base_path,
                args.head_model,
                args.diameter,
                fiber_tract,  # Changes after all jobs for one tract finish
                streamline_number,  # Changes for each job
                args.stim_type,
                args.stim_location,
                args.pulse_width,
                args.stimamp
            )
            for streamline_number in range(num_CPUs)
        ]

        # Parallel execution
        try:
            with Pool(num_CPUs) as pool:
                pool.starmap(stimulate_streamline,
                             streamline_stimulation_arguments)
        except Exception as e:
            print(f"Error stimulating {fiber_tract}: {e}", flush=True)
            exit(1)
