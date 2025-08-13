from multiprocess import Pool
import argparse
import os

from efield_helper_functions import streamline_extraction, efield_extraction


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

    args = parser.parse_args()

    fiber_tract_directory = (
        args.base_path
        + f'WM Fiber Tracts/{args.head_model}/{args.fiber_tract}.vtk'
    )

    fiber_tracts = [
        f.replace('.vtk', '')
        for f in os.listdir(fiber_tract_directory)
        if f.endswith('.vtk')
    ]

    num_CPUs = 50

    for fiber_tract in fiber_tracts:

        streamline_extraction(
            base_path=args.base_path,
            head_model=args.head_model,
            fiber_tract=fiber_tract,
            num_streamlines=num_CPUs
        )

        efield_extraction_arguments = [
            (
                args.base_path,
                args.head_model,
                fiber_tract,  # Changes after all jobs for one tract finish
                streamline_number,  # Changes for each job
                args.stim_type,
                args.stim_location
            )
            for streamline_number in range(num_CPUs)
        ]

        # Parallel execution
        with Pool(num_CPUs) as pool:
            pool.starmap(efield_extraction, efield_extraction_arguments)
