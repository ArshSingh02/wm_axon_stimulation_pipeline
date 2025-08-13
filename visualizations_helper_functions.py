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


def save_vtk_file(output_path, vtk_points, vtk_proj_efield,
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
    polydata.SetPoints(vtk_points)
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
