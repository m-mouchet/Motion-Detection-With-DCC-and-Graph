import numpy as np
import itk
from itk import RTK as rtk


def NPtoRTK(geometryArray, i):
    sid, sdd, ga, dx, dy, oa, ia, sx, sy = geometryArray[0, i], geometryArray[1, i], geometryArray[2, i], geometryArray[3, i], geometryArray[4, i], geometryArray[5, i], geometryArray[6, i], geometryArray[7, i], geometryArray[8, i]
    g = rtk.ThreeDCircularProjectionGeometry.New()
    g.SetRadiusCylindricalDetector(geometryArray[9, i])
    g.AddProjectionInRadians(sid, sdd, ga, dx, dy, oa, ia, sx, sy)
    return g


def ARRAYtoRTK(projArray, projSpacing, projOrigin, projSize, projDirection, i):
    slicea = projArray[i:i+1, :, :]
    projSlice = itk.GetImageFromArray(np.float32(slicea))
    projSlice.SetOrigin(projOrigin)
    projSlice.SetSpacing(projSpacing)
    projDirection = itk.Matrix[itk.D, 3, 3](itk.GetVnlMatrixFromArray(projDirection))
    projSlice.SetDirection(projDirection)
    projSlice.Update()
    return projSlice
