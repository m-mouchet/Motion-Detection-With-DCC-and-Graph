import numpy as np
import itk
from itk import RTK as rtk
from ArrayToRTKConversion import NPtoRTK, ARRAYtoRTK


def computeDCCidx(projArray, projSpacing, projOrigin, projSize, projDirection, rangeDCC, geometryArray, i0):
    # Calcul les dcc de toutes les pairs possible avec la projection i0
    nproj = projSize[2]
    listOfPairs = []
    momentsError = []
    for i1 in rangeDCC:
        if i1+i0 < 0:
            pass
        elif i1+i0 > nproj-1:
            break
        else:
            m0, m1 = ComputePairDCC(projArray, projSpacing, projOrigin, projSize, projDirection, geometryArray, i0, i0+i1)
            if np.sum(m0) == 0 or np.sum(m1) == 0:
                print(i0, i1+i0)
            else:
                listOfPairs.append([i0, i0+i1])
#               # momentsError.append(np.sqrt(np.sum((m0-m1)**2))/(len(m0)*np.mean([np.sum(m0),np.sum(m1)])))
                momentsError.append(np.sqrt(np.sum((m0-m1)**2)/len(m0)))
    return (listOfPairs, momentsError)


def ComputePairDCC(projArray, projSpacing, projOrigin, projSize, projDirection, geometryArray, i0, i1):
    g0 = NPtoRTK(geometryArray, i0)
    p0 = ARRAYtoRTK(projArray, projSpacing, projOrigin, projSize, projDirection, i0)
    g1 = NPtoRTK(geometryArray, i1)
    p1 = ARRAYtoRTK(projArray, projSpacing, projOrigin, projSize, projDirection, i1)
    m0, origin0, oCorner0, volDirection0 = LinesMomentCorner(g0, g1, p0)
    m1, origin1, oCorner1, volDirection1 = LinesMomentCorner(g1, g0, p1)
    if m0.shape == () and m1.shape == ():
        return np.array([m0]), np.array([m1])
    else:
        return m0, m1


def LinesMomentCorner(geometry0, geometry1, projection):
    # start_time_tot = time.time()
    # start_time = time.time()
    ImageType = itk.Image[itk.F, 3]

    R = geometry0.GetRadiusCylindricalDetector()
    # SDD = geometry0.GetSourceToDetectorDistances()[0]
    # print("R=%f" %(R))

    # Compute backprojection plane direction
    # x'direction
    sourcePos0 = geometry0.GetSourcePosition(0)
    sourcePos1 = geometry1.GetSourcePosition(0)
    sourcePos0 = itk.GetArrayFromVnlVector(sourcePos0.GetVnlVector())
    sourcePos1 = itk.GetArrayFromVnlVector(sourcePos1.GetVnlVector())
    sourceDir0 = sourcePos0[0:3] - sourcePos1[0:3]
    sourceDir0 /= np.linalg.norm(sourceDir0)
    if (np.dot(sourceDir0, np.array([1., 0., 0.])) < 0):
        sourceDir0 *= -1.
    # z' direction
    matRot0 = geometry0.GetRotationMatrix(0)
    matRot1 = geometry1.GetRotationMatrix(0)
    matRot0 = itk.GetArrayFromVnlMatrix(matRot0.GetVnlMatrix().as_matrix())
    matRot1 = itk.GetArrayFromVnlMatrix(matRot1.GetVnlMatrix().as_matrix())
    n0 = matRot0[2, 0:3]
    n1 = matRot1[2, 0:3]
    sourceDir2 = 0.5*(n0+n1)
    sourceDir2 /= np.linalg.norm(sourceDir2)
    # y'direction
    sourceDir1 = np.cross(sourceDir2, sourceDir0)
    sourceDir1 /= np.linalg.norm(sourceDir1)
    # backprojection direction matrix
    volDirection = np.vstack((sourceDir0, sourceDir1, sourceDir2))
    # elapsed_time = time.time() - start_time
    # print("VF computation time = %f" %(elapsed_time))

    # start_time = time.time()
    # Compute BP plane corners
    corners = None
    projIdxToCoord0 = geometry0.GetProjectionCoordinatesToFixedSystemMatrix(0)
    projIdxToCoord1 = geometry1.GetProjectionCoordinatesToFixedSystemMatrix(0)
    projIdxToCoord0 = itk.GetArrayFromVnlMatrix(projIdxToCoord0.GetVnlMatrix().as_matrix())
    projIdxToCoord1 = itk.GetArrayFromVnlMatrix(projIdxToCoord1.GetVnlMatrix().as_matrix())

    directionProj = itk.GetArrayFromMatrix(projection.GetDirection())
    matId = np.identity(3)
    matProd = directionProj * matId != directionProj
    if (np.sum(matProd) != 0):
        print("la matrice a %f element(s) non diagonal(aux)" % (np.sum(matProd)))
    else:
        size = []
        for i in range(len(projection.GetOrigin())):
            size.append(projection.GetSpacing()[i]*(projection.GetLargestPossibleRegion().GetSize()[i]-1)*directionProj[i, i])

    invMag_List = []

    for j in projection.GetOrigin()[1], projection.GetOrigin()[1]+size[1]:
        for i in projection.GetOrigin()[0], projection.GetOrigin()[0]+size[0]:
            if R == 0:  # flat detector
                u = i
                v = j
                w = 0
            else:  # cylindrical detector
                theta = i/R
                u = R*np.sin(theta)
                v = j
                w = R*(1-np.cos(theta))
            idx = np.array((u, v, w, 1))
            coord0 = projIdxToCoord0.dot(idx)
            coord1 = projIdxToCoord1.dot(idx)
            # Project on the plane direction, compute inverse mag to go to isocenter and compute the source to pixel / plane intersection
            coord0Source = sourcePos0-coord0
            invMag = np.dot(sourcePos0[0:3], sourceDir2)/np.dot(coord0Source[0:3], sourceDir2)
            invMag_List.append(invMag)
            planePos0 = np.dot(volDirection, sourcePos0[0:3]-invMag*coord0Source[0:3])
            coord1Source = sourcePos1-coord1
            invMag = np.dot(sourcePos1[0:3], sourceDir2)/np.dot(coord1Source[0:3], sourceDir2)
            invMag_List.append(invMag)
            planePos1 = np.dot(volDirection, sourcePos1[0:3]-invMag*coord1Source[0:3])

            if corners is None:
                corners = planePos0[0:2]
            else:
                corners = np.vstack((corners, planePos0[0:2]))
            corners = np.vstack((corners, planePos1[0:2]))

    invMagSpacing = np.mean(invMag_List)

    # Check order of corners after backprojection for y
    for i in range(4):
        if(corners[4+i, 1] < corners[i, 1]):
            corners[4+i, 1], corners[i, 1] = corners[i, 1], corners[4+i, 1]
    # print(corners)

    # Find origin and opposite corner
    origin = np.array([np.min(corners[:, 0]), np.max(corners[np.arange(4), 1]), 0.])
    otherCorner = np.array([np.max(corners[:, 0]), np.min(corners[4 + np.arange(4), 1]), 0.])
    # print(origin,otherCorner)

    # Create empty bp plane
    volDirection = volDirection.T.copy()
#     volSpacing = np.array([(otherCorner[0]-origin[0])/(projection.GetLargestPossibleRegion().GetSize()[0]-1),
#                            (otherCorner[1]-origin[1])/(projection.GetLargestPossibleRegion().GetSize()[1]-1),
#                            1])
    volSpacing = np.array([projection.GetSpacing()[0]*invMagSpacing, projection.GetSpacing()[1]*invMagSpacing, 1])
    for i in range(len(volSpacing)):
        if volSpacing[i] < 0:
            volDirection[i, :] *= -1.0
            volSpacing[i] *= -1
#     volSize = [projection.GetLargestPossibleRegion().GetSize()[0],projection.GetLargestPossibleRegion().GetSize()[1],1]
    volSize = [int((otherCorner[0]-origin[0])//volSpacing[0]+1), int((otherCorner[1]-origin[1])//volSpacing[1]+1), 1]
    volOrigin = np.dot(volDirection, origin)
    volOtherCorner = np.dot(volDirection, otherCorner)
#     elapsed_time = time.time() - start_time
#     print("BP plane computation time = %f" %(elapsed_time))
#     start_time = time.time()
    constantVolFilter = rtk.ConstantImageSource[ImageType].New()
    constantVolFilter.SetOrigin(volOrigin)
    constantVolFilter.SetSize(volSize)
    constantVolFilter.SetSpacing(volSpacing)
    volDirITK = itk.Matrix[itk.D, 3, 3](itk.GetVnlMatrixFromArray(volDirection))
    constantVolFilter.SetDirection(volDirITK)
    constantVolFilter.SetConstant(0.)
    constantVolFilter.Update()
    vol = constantVolFilter.GetOutput()
#     elapsed_time = time.time() - start_time
#     print("Empty volume computation time = %f" %(elapsed_time))
    # Backproject
#     start_time = time.time()
    bp = rtk.BackProjectionImageFilter[ImageType, ImageType].New()
    bp.SetGeometry(geometry0)
    vol = bp.SetInput(0, vol)
    vol = bp.SetInput(1, projection)
    bp.Update()
    vol = bp.GetOutput()
    vol.DisconnectPipeline()
    bp.Update()
    vol2 = bp.GetOutput()
    # itk.imwrite(vol, "vol_%d.mha" %(time.time()*1000))
#     elapsed_time = time.time() - start_time
#     print("BP computation time = %f" %(elapsed_time))

    # Reset plane direction and weight
#     start_time = time.time()
    vol.SetOrigin(origin)
    identity = itk.Matrix[itk.D, 3, 3]()
    identity.SetIdentity()
    vol.SetDirection(identity)
    geoWeight = rtk.ThreeDCircularProjectionGeometry.New()
    sourcePosWeight = np.dot(volDirection.T, sourcePos0[0:3])
    geoWeight.AddProjection(sourcePosWeight[2], sourcePosWeight[2], 0, -sourcePosWeight[0], -sourcePosWeight[1], 0, 0, 0, 0)
    weightFilter = rtk.FDKWeightProjectionFilter[ImageType].New()
    weightFilter.SetGeometry(geoWeight)
    weightFilter.SetInput(vol)
    weightFilter.Update()
    weight = weightFilter.GetOutput()
    # itk.imwrite(weightFilter.GetOutput(), "weights_%d.mha" %(time.time()*100))
    weightarray = itk.GetArrayFromImage(weight)  # Make sure that the weighting is done
#     elapsed_time = time.time() - start_time
#     print("Weighting computation time = %f" %(elapsed_time))

#     #test ponderation
#     emptyVol = rtk.ConstantImageSource[ImageType].New()
#     emptyVol.SetOrigin(origin)
#     emptyVol.SetDirection(identity)
#     emptyVol.SetSpacing(volSpacing)
#     emptyVol.SetConstant(1.0)
#     weightFilter2 = rtk.FDKWeightProjectionFilter[ImageType].New()
#     weightFilter2.SetGeometry(geoWeight)
#     weightFilter2.SetInput(emptyVol)
#     weightFilter2.Update()
#     weight2 = weightFilter2.GetOutput()
#     itk.imwrite(weightFilter2.GetOutput(), "weights_%d.mha" %(time.time()*100))
#     elapsed_time_tot = time.time() - start_time_tot
#     print("tot_time = %f" %(elapsed_time_tot))

    return np.squeeze(vol.GetSpacing()[0]*np.sum(weightarray, axis=2)), volOrigin, volOtherCorner, volDirection
