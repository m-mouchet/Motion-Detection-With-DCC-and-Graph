import numpy as np
import itk
from itk import RTK as rtk

""" Pour utiliser les fonctions ci-dessous, il faut que
la geometrie RTK ait été convertie en vecteurs. """


def ComputeVirtualFrame(sourcePos, matrices, i0, i1):
    # Calcul le nouveau repère pour les projections i0 et i1
    sourceDir0 = sourcePos[i0, :]-sourcePos[i1, :]
    sourceDir0 /= np.linalg.norm(sourceDir0)
    if (np.dot(sourceDir0, np.array([1., 0., 0.])) < 0):
        sourceDir0 *= -1.
    n0 = matrices[i0][2, 0:3]
    n1 = matrices[i1][2, 0:3]
    sourceDir2 = 0.5*(n0+n1)
    sourceDir2 /= np.linalg.norm(sourceDir2)
    # y'direction
    sourceDir1 = np.cross(sourceDir2, sourceDir0)
    sourceDir1 /= np.linalg.norm(sourceDir1)
    # backprojection direction matrix
    volDirection = np.vstack((sourceDir0, sourceDir1, sourceDir2))
    return volDirection


def CheckFOVcondition(geometryArray, projSize, projSpacing, i0, i1):
    SID = geometryArray[0, 0]
    SDD = geometryArray[1, 0]
    gamma = (projSize[0]-1)*360*projSpacing[0]/(4*np.pi*SDD)
    gammar = gamma * np.pi/180
    r_fov = SID*np.sin(gammar)
    a0, a1 = geometryArray[2, i0], geometryArray[2, i1]
    cosDelta = np.cos(a0-a1)
    if cosDelta > 2*((r_fov)/SID)**2-1:
        return True
    else:
        return False


def ComputeRetroProjLimits(R_det, sourcePos, matrices, fsMatrices, directionProj, projOrigin, projSize, projSpacing, i0, i1):
    volDirection = ComputeVirtualFrame(sourcePos, matrices, i0, i1)
    sourceDir2 = volDirection[2, :]
    sourcePos0 = sourcePos[i0, :]
    sourcePos1 = sourcePos[i1, :]
    projIdxToCoord0 = fsMatrices[i0]
    projIdxToCoord1 = fsMatrices[i1]

    # Change s'il faut la direction de la projection
    matId = np.identity(3)
    matProd = directionProj * matId != directionProj
    if (np.sum(matProd) != 0):
        print("la matrice a %f element(s) non diagonal(aux)" % (np.sum(matProd)))
    else:
        size = []
        for i in range(len(projOrigin)):
            size.append(projSpacing[i]*(projSize[i]-1)*directionProj[i, i])

    # Retroprojection des coin sur le plan de retroprojection
    rp0 = []
    rp1 = []
    invMag_List = []
    for j in projOrigin[1], projOrigin[1]+size[1]:
        for i in projOrigin[0], projOrigin[0]+size[0]:
            if R_det == 0:  # flat detector
                u = i
                v = j
                w = 0
            else:  # cylindrical detector
                theta = i/R_det
                u = R_det*np.sin(theta)
                v = j
                w = R_det*(1-np.cos(theta))
            idx = np.array((u, v, w, 1))
            coord0 = projIdxToCoord0.dot(idx)
            coord1 = projIdxToCoord1.dot(idx)
            # Project on the plane direction, compute inverse mag to go to isocenter and compute the source to pixel / plane intersection
            coord0Source = sourcePos0-coord0[0:3]
            invMag = np.dot(sourcePos0, sourceDir2)/np.dot(coord0Source, sourceDir2)
            invMag_List.append(invMag)
            rp0.append(np.dot(volDirection, sourcePos0-invMag*coord0Source))
            coord1Source = sourcePos1-coord1[0:3]
            invMag = np.dot(sourcePos1[0:3], sourceDir2)/np.dot(coord1Source, sourceDir2)
            invMag_List.append(invMag)
            rp1.append(np.dot(volDirection, sourcePos1-invMag*coord1Source))

    invMagSpacing = np.mean(invMag_List)
    volSpacing = np.array([projSpacing[0]*invMagSpacing, projSpacing[1]*invMagSpacing, 1])
    return rp0, rp1, volSpacing


def CheckOverlapCondition(geometryArray, R_det, sourcePos, matrices, fsMatrices, directionProj, projOrigin, projSize, projSpacing, i0, i1):
    rp0, rp1, volSpacing = ComputeRetroProjLimits(R_det, sourcePos, matrices, fsMatrices, directionProj, projOrigin, projSize, projSpacing, i0, i1)
    volDirection = ComputeVirtualFrame(sourcePos, matrices, i0, i1)
    sourceDir1 = volDirection[1, :]
    ey = np.array([0, 1, 0])
    if np.dot(ey, sourceDir1) < 0:
        for i in range(4):
            rp0[i][1] *= -1
            rp1[i][1] *= -1

    if directionProj[1][1] < 0:
        if max(rp0[2][1], rp0[3][1], rp1[2][1], rp1[3][1]) + volSpacing[1] < min(rp0[0][1], rp0[1][1], rp1[0][1], rp1[1][1]):
            return True
        else:
            return False
    elif directionProj[1][1] > 0:
        if max(rp0[0][1], rp0[1][1], rp1[0][1], rp1[1][1]) + volSpacing[1] < min(rp0[2][1], rp0[3][1], rp1[2][1], rp1[3][1]):
            return True
        else:
            return False
    else:
        return False


def CheckPairGeometry(geometryArray, R_det, sourcePos, matrices, fsMatrices, directionProj, projOrigin, projSize, projSpacing, i0, i1):
    # FOVcondition = CheckFOVcondition(geometryArray, projSize, projSpacing, i0,i1)
    overlapCondition = CheckOverlapCondition(geometryArray, R_det, sourcePos, matrices, fsMatrices, directionProj, projOrigin, projSize, projSpacing, i0, i1)
    if overlapCondition:
        # if FOVcondition == True and overlapCondition == True :
        return i1-i0
    else:
        return 0
