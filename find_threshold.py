from multiprocessing import Pool
import argparse
import os
import csv

from streamline_helper_functions import find_streamline_threshold


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

        os.makedirs(results_directory, exist_ok=True)
        thresholds_file = os.path.join(
            results_directory,
            f"{fiber_tract}_{args.diameter}microns_"
            f"{args.stim_type}_{args.stim_location}_{args.pulse_width}ms.csv"
        )

        with open(thresholds_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Fiber Number', 'Activation Threshold'])

        streamline_stimulation_arguments = [
            (
                args.base_path,
                args.head_model,
                args.diameter,
                fiber_tract,  # Changes after all jobs for one tract finish
                streamline_number,  # Changes for each job
                args.stim_type,
                args.stim_location,
                args.pulse_width
            )
            for streamline_number in range(num_CPUs)
        ]

        # Parallel execution
        try:
            with Pool(num_CPUs) as pool:
                pool.starmap(find_streamline_threshold,
                             streamline_stimulation_arguments)
        except Exception as e:
            print(f"Error stimulating {fiber_tract}: {e}", flush=True)
            exit(1)
