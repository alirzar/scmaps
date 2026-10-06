"""Off-screen drawing with VTK: triangles with per-vertex colours, unlit,
orthographic, supersampled and averaged down for antialiased edges."""

import numpy as np

SS = 2                      # supersampling per axis


def downsample(img, ss):
    """ss x ss box average in premultiplied colour (antialiased edges), in row
    blocks so a large supersampled canvas stays small in memory."""
    h, w = img.shape[0] // ss, img.shape[1] // ss
    out = np.empty((h, w, 4), np.uint8)
    step = max(1, (1 << 22) // max(1, w * ss * ss))
    for r0 in range(0, h, step):
        r1 = min(h, r0 + step)
        f = img[r0 * ss:r1 * ss, :w * ss].astype(np.float32) / 255.0
        f[..., :3] *= f[..., 3:]
        f = f.reshape(r1 - r0, ss, w, ss, 4).mean(axis=(1, 3))
        a = f[..., 3:]
        f[..., :3] = np.where(a > 0, f[..., :3] / np.maximum(a, 1e-9), 0.0)
        out[r0:r1] = np.round(np.clip(f, 0.0, 1.0) * 255).astype(np.uint8)
    return out


def crop(img, pad=4):
    """`img` cropped to its non-transparent pixels plus `pad` on each side."""
    a = img[..., 3] > 0
    if not a.any():
        return img
    rows, cols = np.where(a.any(1))[0], np.where(a.any(0))[0]
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, img.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def draw_unlit(pts, faces, rgb, width, height, focal, position, view_up, half_height):
    """The triangles with per-vertex colours interpolated across each face, no
    lighting, orthographic (`half_height` world units from the centre to the
    top edge), on a transparent background: an RGBA image of width x height,
    drawn SS x larger and averaged down."""
    import vtk
    from vtk.util import numpy_support as ns
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(ns.numpy_to_vtk(np.ascontiguousarray(pts, float), deep=True))
    poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    conn = np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel()
    cells.ImportLegacyFormat(ns.numpy_to_vtkIdTypeArray(conn, deep=True))
    poly.SetPolys(cells)
    poly.GetPointData().SetScalars(ns.numpy_to_vtk(
        np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarModeToUsePointData()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(1, 1, 1)
    ren.SetBackgroundAlpha(0.0)
    W, H = int(width) * SS, int(height) * SS
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetAlphaBitPlanes(1)
    win.SetMultiSamples(0)
    win.AddRenderer(ren)
    win.SetSize(W, H)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetFocalPoint(*focal)
    cam.SetPosition(*position)
    cam.SetViewUp(*view_up)
    cam.SetParallelScale(half_height)
    ren.ResetCameraClippingRange()
    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.SetInputBufferTypeToRGBA()
    grab.ReadFrontBufferOff()
    grab.Update()
    img = ns.vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(H, W, 4)[::-1]
    win.Finalize()
    return downsample(img, SS)
