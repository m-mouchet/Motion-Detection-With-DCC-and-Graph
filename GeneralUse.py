import numpy as np
import itk
from itk import RTK as rtk
from RTKToArrayConversion import *

""" Use RTK otherwise this wont work """


def ExtractGeometricParameters(geometryArray, projSize, projSpacing):
    # Extract different geometric parameters of the trajectory
    gantry = geometryArray[2, :]  # Radians angles
    nproj = len(gantry)
    SY = geometryArray[8, :]
    SX = geometryArray[7, :]
    SDD = geometryArray[1, 0]
    SID = geometryArray[0, 0]

    center = [SX[nproj//2], SY[nproj//2], 0]
    a0 = gantry[0]
    nb_rotations = len(np.where(gantry == a0)[0])
    if nb_rotations == 1:
        nb_proj_per_rotations = nproj
    else:
        nb_proj_per_rotations = np.where(gantry == a0)[0][1] - np.where(gantry == a0)[0][0]

    d = np.abs(SY[nb_proj_per_rotations-1]-SY[0])

    pitch = d/(projSize[1]*projSpacing[1])

    gamma = projSize[0]*360*projSpacing[0]/(4*np.pi*SDD)
    gammar = gamma * np.pi/180
    if gammar > 0 and gammar < np.pi/2:
        r_fov = SID*np.sin(gammar)  # ANGLE EN RADIAN
    else:
        print('Error, theta='+str(gamma))
    # print(d*SDD/(projSize[1]*SID*projSpacing[1]))
    return center, nb_rotations, nb_proj_per_rotations, d, pitch, r_fov


def SubsampleAcquisition(geometryArray, projArray, projSpacing, projOrigin, projSize, projDirection, nprojSub=20):
    center, nb_rotations, nb_proj_per_rotations, d, pitch, r_fov = ExtractGeometricParameters(geometryArray, projSize, projSpacing)
    # On réduit le nombre de projection par tour (20 suffisent pour les dcc)

    sampleIdxTurn = np.round(np.linspace(0, nb_proj_per_rotations-1, nprojSub))

    sampleIdx = []
    for i in range(geometryArray[0, :].shape[0]):
        if i % nb_proj_per_rotations in sampleIdxTurn:
            sampleIdx.append(i)
    # print(len(sampleIdx))
    # print(sampleIdx)

    geometry_sub = rtk.ThreeDCircularProjectionGeometry.New()
    geometry_sub.SetRadiusCylindricalDetector(geometryArray[9, 0])

    projArray_sub = np.zeros((len(sampleIdx), projSize[1], projSize[0]))

    for j in range(len(sampleIdx)):
        geometry_sub.AddProjectionInRadians(geometryArray[0, sampleIdx[j]], np.round(geometryArray[1, sampleIdx[j]], 1), geometryArray[2, sampleIdx[j]], geometryArray[3, sampleIdx[j]], geometryArray[4, sampleIdx[j]], geometryArray[5, sampleIdx[j]], geometryArray[6, sampleIdx[j]], geometryArray[7, sampleIdx[j]], geometryArray[8, sampleIdx[j]])
        #     print(ga*180/np.pi)
    #     print(j,sampleIdx[j])
        slicea = projArray[int(sampleIdx[j]):int(sampleIdx[j])+1, :, :]
    #     print(slicea.shape,sub_projArray[i:i+1,:,:].shape)
        projArray_sub[j:j+1, :, :] += slicea

    projSub = itk.GetImageFromArray(projArray_sub)
    projSub.SetOrigin(projOrigin)
    projSub.SetSpacing(projSpacing)
    projSub.SetOrigin(projOrigin)
    projDirection = itk.Matrix[itk.D, 3, 3](itk.GetVnlMatrixFromArray(projDirection))
    projSub.SetDirection(projDirection)
    projSub.Update()
    # itk.imwrite(projSub,filesDir+'proj_sub100.mha')

    geometry_subArray = RTKtoNP(geometry_sub)
    return projSub, geometry_sub, projArray_sub, geometry_subArray


def RecupParam(geo, idx):
    # Function that extract the geometric parameters of the projection index idx with geometry geo
    sid = geo.GetSourceToIsocenterDistances()[idx]
    sdd = geo.GetSourceToDetectorDistances()[idx]
    ga = geo.GetGantryAngles()[idx]
    dx = geo.GetProjectionOffsetsX()[idx]
    dy = geo.GetProjectionOffsetsY()[idx]
    oa = geo.GetOutOfPlaneAngles()[idx]
    ia = geo.GetInPlaneAngles()[idx]
    sx = geo.GetSourceOffsetsX()[idx]
    sy = geo.GetSourceOffsetsY()[idx]
    R = geo.GetRadiusCylindricalDetector()
    return sid, sdd, ga, dx, dy, oa, ia, sx, sy, R


def ExtractSlice(stack, num):
    # Function that extract the projection num in the projections stack stack
    ar = itk.GetArrayFromImage(stack)
    projslicea = ar[num:num+1, :, :]
    projslice = itk.GetImageFromArray(projslicea)
    projslice.CopyInformation(stack)
    return projslice


def ExtractSourcePosition(geometry0, geometry1):
    sourcePos0 = geometry0.GetSourcePosition(0)
    sourcePos1 = geometry1.GetSourcePosition(0)
    sourcePos0 = itk.GetArrayFromVnlVector(sourcePos0.GetVnlVector())[0:3]
    sourcePos1 = itk.GetArrayFromVnlVector(sourcePos1.GetVnlVector())[0:3]
    return sourcePos0, sourcePos1
