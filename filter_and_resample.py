import numpy as np
import math


def remove_coordinate_outliers(original_coordinates, min_len=None, max_len=None, mad_k=3.5, max_passes=5, atol=1e-12):
    """
    Filter streamline coordinates

    Remove vertices that create outlier segment lengths
    along a 3D polyline. Outliers are detected on the
    distribution of consecutive step lengths using MAD.

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
    Resample coordinates to a fixed spacing

    Resample a 3D polyline so every consecutive pair of
    points has a fixed spacing within a tolerance.

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


def compute_delta_z(diameter):
    """
    Compute delta_z

    Compute the node to node distance (delta_z value)
    based on the diameter.

    Parameters:
        diameter (float): Diameter of the neuron.

    Returns:
        float: Calculated delta_z value.
    """
    if diameter >= 5.643:
        return (-8.215 * (diameter**2)) + (272.4 * diameter) - 780.2
    return (81.08 * diameter) + 37.84


def compute_n_sections(streamline_length, delta_z):
    """
    Compute n_sections

    Compute the number of sections based on the
    streamline length and delta_z

    Parameters:
        streamline_length (float): Length of the streamline in mm.
        delta_z (float): Node to node distance in mm.

    Returns:
        int: Calculated number of sections.
    """
    n_sections = (math.floor(streamline_length / delta_z) * 11) + 1
    return n_sections


def compute_effective_streamline_length(n_sections, delta_z):
    """
    Compute effective streamline length

    Compute the streamline length for NEURON
    based on the number of sections and delta_z

    Parameters:
        n_sections (int): Number of sections.
        delta_z (float): Node to node distance in mm.

    Returns:
        float: Calculated streamline length in mm.
    """
    effective_streamline_length = ((n_sections - 1) / 11) * delta_z
    return effective_streamline_length


def compute_cumulative_distances(points):
    """
    Compute cumulative distance

    Compute cumulative distances along the streamline.

    Parameters:
        points (np.ndarray): Array of points (Nx3) representing coordinates.

    Returns:
        np.ndarray: Cumulative distances along the streamline.
    """
    d = np.diff(points, axis=0)
    s = np.linalg.norm(d, axis=1)
    return np.concatenate(([0.0], np.cumsum(s)))


def mrg_section_lengths_mm(d):
    """
    Compute section lengths for MRG

    Compute the section lengths based on the diameter.

    Parameters:
        d (float): Diameter of the neuron.

    Returns:
        np.ndarray: Array of section lengths in mm.
    """
    if d >= 5.643:
        delta_z = (-8.215 * d**2) + (272.4 * d) - 780.2
    else:
        delta_z = (81.08 * d) + 37.08
    nor = 1.0
    mysa = 3.0
    flut = (-0.1652 * d**2) + (6.354 * d) - 0.2862
    stin = (delta_z - nor - 2*mysa - 2*flut) / 6.0
    return np.array([nor, mysa, flut, stin, stin, stin, stin, stin, stin, flut, mysa], dtype=float) / 1000.0


def make_mrg_centers(total_len_mm, d, n_sections, atol=1e-12):
    """
    Make MRG centers

    Generate the center positions for MRG sections
    along the streamline based on the total length,
    diameter, and number of sections.

    Parameters:
        total_len_mm (float): Total length of the streamline in mm.
        d (float): Diameter of the neuron.
        n_sections (int): Number of sections.
        atol (float): Tolerance for floating-point comparisons.

    Returns:
        np.ndarray: Array of center positions for each section in mm.
    """
    L = mrg_section_lengths_mm(d)
    per_rep = L.sum()
    centers = []
    offset = 0.0
    while len(centers) < n_sections and offset + 0.5*L[0] <= total_len_mm + atol:
        starts = offset + np.concatenate(([0.0], np.cumsum(L[:-1])))
        ends = starts + L
        c = 0.5*(starts + ends)
        for ci in c:
            if ci <= total_len_mm + atol:
                centers.append(ci)
                if len(centers) == n_sections:
                    break
        offset += per_rep
        if offset > total_len_mm + atol:
            break
    return np.array(centers[:n_sections], dtype=float)


def interpolate_fiber_variable_centers(coordinates, d, n_sections):
    """
    Interpolate fiber variable centers

    Interpolate the fiber coordinates to match
    the MRG section centers based on diameter
    and number of sections.

    Parameters:
        coordinates (np.ndarray): Original fiber coordinates (Nx3).
        d (float): Diameter of the neuron.
        n_sections (int): Number of sections.

    Returns:
        tuple: (interp_coords, interp_arc)
            interp_coords (np.ndarray): Interpolated coordinates (n_sections x 3).
            interp_arc (np.ndarray): Cumulative distances for interpolated coordinates.
    """
    arc = compute_cumulative_distances(coordinates)
    total_len = arc[-1]
    center_arc = make_mrg_centers(total_len, d, n_sections)
    interp_coords = np.column_stack([np.interp(center_arc, arc, coordinates[:, k]) for k in range(3)])
    return interp_coords, center_arc
