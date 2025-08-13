import numpy as np
import vtk


def create_vtk_points(points):
    """
    Create a VTK Points object from a list of 3D points.

    Parameters:
        points (list of tuple): List of (x, y, z) coordinates.

    Returns:
        vtk.vtkPoints: VTK Points object containing the coordinates.
    """
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)
    return vtk_points


def create_vtk_vectors(vectors, name):
    """
    Create a VTK DoubleArray object for vector data.

    Parameters:
        vectors (list of tuple): List of (x, y, z) vectors.
        name (str): Name for the VTK data array.

    Returns:
        vtk.vtkDoubleArray: VTK DoubleArray object containing the vector data.
    """
    vtk_vectors = vtk.vtkDoubleArray()
    vtk_vectors.SetNumberOfComponents(3)
    vtk_vectors.SetName(name)
    for vector in vectors:
        vtk_vectors.InsertNextTuple3(*vector)
    return vtk_vectors


def create_vtk_scalars(scalars, name):
    """
    Create a VTK DoubleArray object for scalar data.

    Parameters:
        scalars (list of float): List of scalar values.
        name (str): Name for the scalar data array.

    Returns:
        vtk.vtkDoubleArray: VTK DoubleArray object containing the scalar data.
    """
    vtk_scalars = vtk.vtkDoubleArray()
    vtk_scalars.SetName(name)
    for value in scalars:
        vtk_scalars.InsertNextValue(value)
    return vtk_scalars


def save_vtk_file(output_path, vtk_coords, vtk_proj_efield,
                  vtk_potentials, vtk_af):
    """
    Save VTK data (points and vectors) to a VTK file.

    Parameters:
        output_path (str): Path to the output VTK file.
        vtk_points (vtk.vtkPoints): VTK Points object.
        vtk_vectors (vtk.vtkDoubleArray): VTK DoubleArray object for vectors.
    """
    # Create a VTK PolyData object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_coords)
    # polydata.GetPointData().AddArray(vtk_efield)
    polydata.GetPointData().AddArray(vtk_proj_efield)
    polydata.GetPointData().AddArray(vtk_potentials)
    polydata.GetPointData().AddArray(vtk_af)

    # Create a VTK VertexGlyphFilter for visualization
    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    # Write the VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(vertex_filter.GetOutput())
    writer.Write()


def detect_ap_nodes(vm_list, threshold=-30.0):
    """
    Detects APs based on spatial peak detection at each timepoint:
    A node is marked as having an AP at time i if:
        - Its voltage >= threshold
        - Its neighboring nodes (j-1 and j+1) are below threshold
    Returns a set of node indices with detected APs across all timepoints.
    """

    timeIdxs = []

    for i in range(0, len(vm_list)):
        timeIdxs.append(np.where(np.array(vm_list[i]) >= threshold))

    firstAP_time = float('inf')
    nodeIdx = 0
    for i, node in enumerate(timeIdxs):

        if len(node[0]) != 0:
            if node[0][0] < firstAP_time:
                nodeIdx = i
                firstAP_time = node[0][0]

    ap_node_set = set([(nodeIdx - 1) * 11])
    return ap_node_set


def create_voltage_labeled_vtk(vtk_coords, ap_nodes,
                               output_vtk_file, num_points):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_coords)

    ap_node_list = list(ap_nodes)

    voltage_array = vtk.vtkFloatArray()
    voltage_array.SetName("Voltage")

    for node_index in range(num_points):

        if (ap_node_list[0] - 15) <= node_index <= (ap_node_list[0] + 15):
            voltage = 80.0
        else:
            voltage = 0.0

        voltage_array.InsertNextValue(voltage)

    # Fill in any extra points with 0.0
    while voltage_array.GetNumberOfTuples() < num_points:
        voltage_array.InsertNextValue(0.0)

    polydata.GetPointData().SetScalars(voltage_array)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_file)
    writer.SetInputData(polydata)
    writer.Write()


def process_fiber_ap(fiber, coords_mrg_resolution,
                     streamline_activation_file):
    """
    Detect APs from fiber.vm and generate VTK with voltage labeling.
    """

    num_points = len(fiber.sections)
    ap_nodes = detect_ap_nodes(fiber.vm)

    vtk_coords = create_vtk_points(
        coords_mrg_resolution[:num_points]
    )

    create_voltage_labeled_vtk(vtk_coords, ap_nodes,
                               streamline_activation_file,
                               num_points)
