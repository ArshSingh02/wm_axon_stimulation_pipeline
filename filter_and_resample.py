import pandas as pd
import numpy as np
import math


def remove_coordinate_outliers(original_coordinates, min_len=None, max_len=None, mad_k=3.5, max_passes=5, atol=1e-12):
    """
    Remove vertices that create outlier segment lengths along a 3D polyline.
    Outliers are detected on the distribution of consecutive step lengths using MAD.

    Parameters
    ----------
    coordinates : (N,3) array_like
    min_len : float or None 
        absolute lower bound (mm) to keep; e.g., 0.02 to kill near-duplicates
    max_len : float or None
        absolute upper bound (mm) to keep
    mad_k : float
        robustness threshold; larger = less aggressive
    max_passes : int
        iterate removal/recompute until no outliers or passes exhausted
    atol : float
        Tolerance used internally for floating-point guard rails.

    Returns
    -------
    (M,3) ndarray
        Filtered out points with outlier resolutions
    """
    pts = np.asarray(original_coordinates, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("coordinates must be an Nx3 array")

    out = pts.copy()
    for _ in range(max_passes):
        if len(out) < 3:
            break
        seg = np.linalg.norm(np.diff(out, axis=0), axis=1)

        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        sigma = 1.4826 * mad if mad > 0 else 0.0

        lo = med - mad_k * sigma if sigma > 0 else -np.inf
        hi = med + mad_k * sigma if sigma > 0 else  np.inf
        if min_len is not None: lo = max(lo, float(min_len))
        if max_len is not None: hi = min(hi, float(max_len))

        bad = np.where((seg < lo - atol) | (seg > hi + atol))[0]
        if bad.size == 0:
            break

        drop = np.unique(bad + 1)
        drop = drop[drop < len(out) - 1]
        if drop.size == 0:
            break

        mask = np.ones(len(out), dtype=bool)
        mask[drop] = False
        filtered_coordinates = out[mask]

    return filtered_coordinates

def resample_coordinates_simnibs_resolution(filtered_coordinates, spacing=0.1, include_end=False, atol=1e-12):
    """
    Resample a 3D polyline so every consecutive pair of points is EXACTLY `spacing` mm apart.
    Walking is done along the original polyline geometry (no axis-wise interpolation).

    Parameters
    ----------
    coordinates : (N,3) array_like
        Input polyline points in mm.
    spacing : float
        Desired fixed spacing in mm (e.g., 0.1).
    include_end : bool
        If True, append the original last point even if the final segment < spacing.
        If False, stop at the last exact multiple of spacing.
    atol : float
        Tolerance used internally for floating-point guard rails.

    Returns
    -------
    (M,3) ndarray
        Resampled points at exact spacing.
    """
    coords = np.asarray(filtered_coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must be an Nx3 array")

    # Remove zero-length steps
    diffs = np.diff(coords, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = seglens > atol
    coords = coords[keep]
    if len(coords) < 2:
        return coords.copy()

    out = [coords[0].copy()]
    carry = 0.0

    for i in range(1, len(coords)):
        p0 = coords[i - 1].copy()
        p1 = coords[i].copy()
        seg_vec = p1 - p0
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len <= atol:
            continue
        seg_dir = seg_vec / seg_len

        remaining = seg_len
        while carry + remaining >= spacing - atol:
            step = spacing - carry
            new_pt = p0 + seg_dir * step

            last = out[-1]
            vec = new_pt - last
            d = float(np.linalg.norm(vec))
            if d == 0.0:
                new_pt = new_pt + seg_dir * spacing
                vec = new_pt - last
                d = float(np.linalg.norm(vec))
            new_pt = last + vec * (spacing / d)

            out.append(new_pt)

            p0 = new_pt
            remaining = float(np.linalg.norm(p1 - p0))
            carry = 0.0

        carry += remaining

    if include_end:

        if np.linalg.norm(out[-1] - coords[-1]) > atol:
            out.append(coords[-1].copy())
        
    resampled_coordinates = np.vstack(out)

    return resampled_coordinates

def compute_cumulative_distances(points):
    """
    Compute cumulative distances along the streamline.

    Parameters:
        points (np.ndarray): Array of points (Nx3) representing coordinates.

    Returns:
        np.ndarray: Cumulative distances along the streamline.
    """
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    return np.insert(np.cumsum(distances), 0, 0)


def compute_delta_z(diameter):
    """
    Compute the delta_z value based on the diameter.

    Parameters:
        diameter (float): Diameter of the neuron.

    Returns:
        float: Calculated delta_z value.
    """
    if diameter >= 5.643:
        return (-8.215 * (diameter**2)) + (272.4 * diameter) - 780.2
    return (81.08 * diameter) + 37.84


def compute_n_sections(streamline_length, delta_z):
    return (math.floor(streamline_length / delta_z) * 11) + 1


def compute_streamline_length(n_sections, delta_z):
    return ((n_sections - 1) / 11) * delta_z