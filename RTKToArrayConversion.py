import numpy as np
import itk
from itk import RTK as rtk


def RTKtoNP(geometryRTK):
    # Extrait les informations de rtk pour en faire des vecteurs
    nproj = len(geometryRTK.GetGantryAngles())
    R_det = geometryRTK.GetRadiusCylindricalDetector()
    geometryArray = np.zeros((10, nproj))
    geometryArray[0, :] = geometryRTK.GetSourceToIsocenterDistances()
    geometryArray[1, :] = geometryRTK.GetSourceToDetectorDistances()
    geometryArray[2, :] = geometryRTK.GetGantryAngles()
    geometryArray[3, :] = geometryRTK.GetProjectionOffsetsX()
    geometryArray[4, :] = geometryRTK.GetProjectionOffsetsY()
    geometryArray[5, :] = geometryRTK.GetOutOfPlaneAngles()
    geometryArray[6, :] = geometryRTK.GetInPlaneAngles()
    geometryArray[7, :] = geometryRTK.GetSourceOffsetsX()
    geometryArray[8, :] = geometryRTK.GetSourceOffsetsY()
    geometryArray[9, :] = np.ones(nproj)*R_det
    return geometryArray


def GetProjectionInformations(projection):
    # Extrait les information des projections pour les convertir en vecteurs
    projSpacing = np.copy(projection.GetSpacing())
    projOrigin = np.copy(projection.GetOrigin())
    projSize = np.copy(projection.GetLargestPossibleRegion().GetSize())
    projDirection = np.copy(itk.GetArrayFromMatrix(projection.GetDirection()))
    return projSpacing, projOrigin, projSize, projDirection


def GetSourcePositions(geometryRTK):
    # Extrait les positions des sources d'une geometrie RTK
    nproj = len(geometryRTK.GetGantryAngles())
    sourcePositions = np.zeros((nproj, 3))
    for i in range(nproj):
        pos = geometryRTK.GetSourcePosition(i)
        pos = itk.GetArrayFromVnlVector(pos.GetVnlVector())
        sourcePositions[i, :] = pos[0:3]
    return sourcePositions


def GetRotationMatrices(geometryRTK):
    # Extrait les matrices de rotation d'une geometie RTK
    matrices = []
    nproj = len(geometryRTK.GetGantryAngles())
    for i in range(nproj):
        MatRot = geometryRTK.GetRotationMatrix(i)
        MatRot = itk.GetArrayFromVnlMatrix(MatRot.GetVnlMatrix().as_matrix())
        matrices.append(MatRot)
    return matrices


def GetFixedSystemMatrices(geometryRTK):
    # Extrait les matrices de chaque positions de source d'une geometrie RTK
    matrices = []
    nproj = len(geometryRTK.GetGantryAngles())
    for i in range(nproj):
        fsMat = geometryRTK.GetProjectionCoordinatesToFixedSystemMatrix(i)
        fsMat = itk.GetArrayFromVnlMatrix(fsMat.GetVnlMatrix().as_matrix())
        matrices.append(fsMat)
    return matrices
