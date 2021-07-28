from GeneralUse import *
import numpy as np
from math import floor
from scipy import interpolate


def ComputeDetectorsPositionFast(geometryArray, fsMatrices, directionProj, projSize, projSpacing, projOrigin, idx0, idx1):  # Compute the coordinates in the canonical frame of the detectors
    D = geometryArray[1, idx0]
    Rd = geometryArray[9, idx1]
    R_traj = geometryArray[0, idx0]
    a0, a1 = geometryArray[2, idx0], geometryArray[2, idx1]
    projIdxToCoord0 = fsMatrices[idx0]
    projIdxToCoord1 = fsMatrices[idx1]

    # Check for non negative spacing
    matId = np.identity(3)
    matProd = directionProj * matId != directionProj
    if (np.sum(matProd) != 0):
        print("la matrice a %f element(s) non diagonal(aux)" % (np.sum(matProd)))
    else:
        size = []
        for k in range(len(projOrigin)):
            size.append(projSpacing[k]*(projSize[k]-1)*directionProj[k, k])

    Det0 = np.zeros((projSize[0], projSize[1], 3))
    Det1 = np.zeros((projSize[0], projSize[1], 3))
    thetaTot = np.zeros((projSize[0], projSize[1]))
    Nx, Ny = Det0.shape[0:2]

    if Rd == 0:  # flat detector
        for j in range(Ny):
            for i in range(Nx):
                u = projOrigin[0] + i*projSpacing[0]*directionProj[0, 0]
                v = projOrigin[1] + j*projSpacing[1]*directionProj[1, 1]
                w = 0
                idx = np.array((u, v, w, 1))
                coord0 = projIdxToCoord0.dot(idx)
                coord1 = projIdxToCoord1.dot(idx)
                thetaTot[i, j] += u
                Det0[i, j, :] += coord0[0:3]
                Det1[i, j, :] += coord1[0:3]
        theta = thetaTot[:, 0]
    else:  # cylindrical detector
        theta = (projOrigin[0] + np.arange(Nx)*projSpacing[0]*directionProj[0, 0]+geometryArray[3, idx0])/D
        for j in range(Ny):
            Det0[:, j, 0] += (R_traj-Rd*np.cos(theta))*np.sin(a0+geometryArray[7, idx0]/R_traj) + Rd*np.sin(theta)*np.cos(a0 + geometryArray[7, idx0]/R_traj)
            Det0[:, j, 1] += np.ones(Nx)*(projOrigin[1] + j*projSpacing[1]*directionProj[1, 1] + geometryArray[8, idx0])
            Det0[:, j, 2] += (R_traj-Rd*np.cos(theta))*np.cos(a0 + geometryArray[7, idx0]/R_traj) - Rd*np.sin(theta)*np.sin(a0 + geometryArray[7, idx0]/R_traj)
            Det1[:, j, 0] = (R_traj-Rd*np.cos(theta))*np.sin(a1 + geometryArray[7, idx1]/R_traj) + Rd*np.sin(theta)*np.cos(a1 + geometryArray[7, idx1]/R_traj)
            Det1[:, j, 1] = np.ones(Nx)*(projOrigin[1] + j*projSpacing[1]*directionProj[1, 1] + geometryArray[8, idx1])
            Det1[:, j, 2] = (R_traj-Rd*np.cos(theta))*np.cos(a1 + geometryArray[7, idx1]/R_traj) - Rd*np.sin(theta)*np.sin(a1 + geometryArray[7, idx1]/R_traj)
    return Det0, Det1, theta


def ComputePlaneEquation(A, B, C):  # compute the cartesian equation of the plane formed by the three point ABC
    AB = B-A
    AC = C-A
    normal = np.cross(AB, AC)
    normal /= np.linalg.norm(normal)
    d = -1*np.dot(normal, C)
    return [normal, d]


def ComputeCylindersIntersection(geometryArray, sourcePos, idx0, idx1):  # Compute the intersection of two cylindrical detectors
    D = geometryArray[1, 0]
    s0, s1 = sourcePos[idx0, :], sourcePos[idx1, :]
    # on définit les deux vecteurs directeurs
    u_dir = (np.array([s1[2], s1[0]])-np.array([s0[2], s0[0]]))/np.linalg.norm((np.array([s1[2], s1[0]])-np.array([s0[2], s0[0]])))
    v_dir = np.array([-u_dir[1], u_dir[0]])  # perpendiculaire
    mid_point = (s0+s1)/2
    c = (s0[2]-s1[2])**2/4 + (s0[0]-s1[0])**2/4 - D**2
    roots = np.array([-np.sqrt(np.abs(c)),np.sqrt(np.abs(c))])
    t1 = np.array([roots[0]*v_dir[1]+mid_point[0], 0, roots[0]*v_dir[0]+mid_point[2]])
    t2 = np.array([roots[1]*v_dir[1]+mid_point[0], 0, roots[1]*v_dir[0]+mid_point[2]])
    return t1, t2


def ComputeNewFrame(geometryArray, sourcePos, idx0, idx1):  # Compute the frame centered on each source position
    sourcePos0, sourcePos1 = sourcePos[idx0, :], sourcePos[idx1, :]
    a0, a1 = geometryArray[2, idx0], geometryArray[2, idx1]
    sourceDir0 = sourcePos0[0:3] - sourcePos1[0:3]  # First direction is the line s0/s1
#     if np.abs(a1-a0) > np.pi and a0 < a1:  # Always keep angle between 0-pi
#         sourceDir0 *= -1
    if a0 < np.pi and a1 < np.pi:
        if np.dot(sourceDir0, np.array([0, 0, 1])) < 0 and np.abs(a1-a0) <= np.pi:
            sourceDir0 *= -1
        elif np.abs(a1-a0) > np.pi and np.dot(sourceDir0, np.array([0, 0, 1])) > 0:
            sourceDir0 *= -1
    elif a0 > np.pi and a1 > np.pi:
        if np.dot(sourceDir0, np.array([0, 0, 1])) > 0 and np.abs(a1-a0) <= np.pi:
            sourceDir0 *= -1
    elif np.dot(sourceDir0, np.array([0, 0, 1])) == 0 and a0 < np.pi and a1 > np.pi and np.abs(a1-a0) > np.pi:
        sourceDir0 *= -1
    elif np.dot(sourceDir0, np.array([0, 0, 1])) == 0 and a0 > np.pi and a1 < np.pi and np.abs(a1-a0) < np.pi:
        sourceDir0 *= -1
    sourceDir0[1] = 0
    sourceDir1 = np.array([0, 1, 0])  # Axial direction is kept the same
    sourceDir2 = np.cross(sourceDir0, sourceDir1)  # The third direction is obtained to have a right-handed coordinate system and points towards the detector
    sourceDir0 /= np.linalg.norm(sourceDir0)
    sourceDir2 /= np.linalg.norm(sourceDir2)
    volDir = np.vstack((sourceDir0, sourceDir1, sourceDir2))
    return volDir


def ComputeNewFrameAndMPoints(geometryArray, sourcePos, D0, D1, projSize, projSpacing, idx0, idx1):  # Compute the new frame and the M-points that are used to form planes with the source positions for the cylindrical detectors
    Nx = projSize[0]
    DeltaY = projSpacing[1]
    # Axial coordinates
    y_min = np.max([D0[Nx//2, 0, 1], D1[Nx//2, 0, 1]])
    y_max = np.min([D0[Nx//2, -1, 1], D1[Nx//2, -1, 1]])
    y_dcc = np.linspace(y_min + DeltaY/2, y_max - DeltaY/2, floor(np.abs(y_max-y_min)/DeltaY))
    # Radial coordinates
    ta, tb = ComputeCylindersIntersection(geometryArray, sourcePos, idx0, idx1)
    dist0a = np.linalg.norm(np.array([ta[0], ta[2]])-np.array([D0[Nx//2, 0, 0], D0[Nx//2, 0, 2]]))
    dist0b = np.linalg.norm(np.array([tb[0], tb[2]])-np.array([D0[Nx//2, 0, 0], D0[Nx//2, 0, 2]]))
    dist1a = np.linalg.norm(np.array([ta[0], ta[2]])-np.array([D1[Nx//2, 0, 0], D1[Nx//2, 0, 2]]))
    dist1b = np.linalg.norm(np.array([tb[0], tb[2]])-np.array([D1[Nx//2, 0, 0], D1[Nx//2, 0, 2]]))
    if dist0a <= dist0b and dist1a <= dist1b:
        intersect = ta
#         print("ta")
    else:
        intersect = tb
#         print("tb")
    PMs = np.array([intersect[0]*np.ones(len(y_dcc)), y_dcc, intersect[2]*np.ones(len(y_dcc))])
    return PMs


def CanWeApplyDirectlyTheFormula(angle, xs):  # Check if there is a singularity present in the samples
    a = np.min(angle)
    b = np.max(angle)
    if np.sign(a*b) >0 and np.abs(xs) > b:
        return True
    else:
        if xs < 0 and xs < a:
            return True
        elif xs > 0 and xs > b:
            return True
        else:
            return False


def ComputeAllInOneFunction(geometryArray, sourcePos, M_points, gamma, v_det, idx0, idx1):
    sourcePos0, sourcePos1 = sourcePos[idx0, :], sourcePos[idx1, :]
    b = sourcePos0 - sourcePos1
    b /= np.linalg.norm(b)
    if (np.dot(b, np.array([1., 0., 0.])) < 0):
        b *= -1

    n0 = []
    n1 = []
    M_ACC = []

    D = geometryArray[1, 0]
    a0, a1 = geometryArray[2, idx0], geometryArray[2, idx1]

    volDir0 = np.vstack((np.array([np.cos(a0), 0., -np.sin(a0)]), np.array([0., 1., 0.]), np.array([np.sin(a0), 0., np.cos(a0)])))
    b0 = np.dot(volDir0, b)
    volDir1 = np.vstack((np.array([np.cos(a1), 0., -np.sin(a1)]), np.array([0., 1., 0.]), np.array([np.sin(a1), 0., np.cos(a1)])))
    b1 = np.dot(volDir1, b)

    for j in range(len(M_points[1])):
        n, d = ComputePlaneEquation(sourcePos0, sourcePos1, np.array([M_points[0][j], M_points[1][j], M_points[2][j]]))
        n_new0 = np.dot(volDir0, n)
        n_new1 = np.dot(volDir1, n)
        gamma_e0 = np.arctan(-n_new0[0]/n_new0[2])
        gamma_e1 = np.arctan(-n_new1[0]/n_new1[2])
        if CanWeApplyDirectlyTheFormula(gamma, gamma_e0):
            x0 = np.array([gamma[0], gamma[-1]])
        else:
            x0 = np.array([gamma[0], gamma_e0, gamma[-1]])
        if CanWeApplyDirectlyTheFormula(gamma, gamma_e1):
            x1 = np.array([gamma[0], gamma[-1]])
        else:
            x1 = np.array([gamma[0], gamma_e1, gamma[-1]])

        v0 = D*(-np.sin(x0)*n_new0[0]+np.cos(x0)*n_new0[2])/n_new0[1]
        v1 = D*(-np.sin(x1)*n_new1[0]+np.cos(x1)*n_new1[2])/n_new1[1]
        c0_min = (np.min(v_det) <= np.min(v0) and np.min(v0) <= np.max(v_det))
        c0_max = (np.min(v_det) <= np.max(v0) and np.max(v0) <= np.max(v_det))
        c1_min = (np.min(v_det) <= np.min(v1) and np.min(v1) <= np.max(v_det))
        c1_max = (np.min(v_det) <= np.max(v1) and np.max(v1) <= np.max(v_det))
        if (c0_min and c0_max) and (c1_min and c1_max):
            M_ACC.append(np.array([M_points[0][j], M_points[1][j], M_points[2][j]]))
            n0.append(n_new0)
            n1.append(n_new1)
    return np.array(M_ACC), np.array(n0), b0, np.array(n1), b1


def ComputeMomentsOnCylindricalDetectorsWithSingularity(geometryArray, fsMatrices, directionProj, projArray, projSize, projSpacing, projOrigin, sourcePos, idx0, idx1):
    sy0, sy1 = geometryArray[8, idx0], geometryArray[8, idx1]
    dy0, dy1 = geometryArray[4, idx0], geometryArray[5, idx1]
    projar0 = projArray[idx0, :, :]
    projar1 = projArray[idx1, :, :]
    D = geometryArray[1, 0]

    # Compute the two detectors coordinates RTK angles
    Det0, Det1, gamma_RTK = ComputeDetectorsPositionFast(geometryArray, fsMatrices, directionProj, projSize, projSpacing, projOrigin, idx0, idx1)

    # Compute intersection between the two detectors and all possible M points
    PMs = ComputeNewFrameAndMPoints(geometryArray, sourcePos, Det0, Det1, projSize, projSpacing, idx0, idx1)

    v_det = projOrigin[1] + dy0-sy0 + (np.arange(projSize[1])*projSpacing[1])*directionProj[1, 1]
    M_ACC, n0, b0, n1, b1 = ComputeAllInOneFunction(geometryArray, sourcePos, PMs, gamma_RTK, v_det, idx0, idx1)


    # Axial interpolation of the projection at the different intersection points of the curve
    proj_interp0 = np.zeros((Det0.shape[0], n0.shape[0]))
    proj_interp1 = np.zeros((Det1.shape[0], n1.shape[0]))
    v0 = np.zeros((Det0.shape[0], n0.shape[0]))
    v1 = np.zeros((Det1.shape[0], n1.shape[0]))
    for i in range(Det0.shape[0]):
        v0[i, :] = (-n0[:, 0]*D*np.sin(gamma_RTK[i])+n0[:, 2]*D*np.cos(gamma_RTK[i]))/n0[:, 1]
        v1[i, :] = (-n1[:, 0]*D*np.sin(gamma_RTK[i])+n1[:, 2]*D*np.cos(gamma_RTK[i]))/n1[:, 1]
        proj_interp0[i, :] += np.interp(v0[i, :], v_det, projar0[:, i])
        proj_interp1[i, :] += np.interp(v1[i, :], v_det, projar1[:, i])

    m0_trpz, m1_trpz = np.zeros(v0.shape[1]), np.zeros(v1.shape[1])
    norm0, norm1 = np.zeros(v0.shape[1]), np.zeros(v1.shape[1])
    grad0, grad1 = np.zeros(v0.shape[1]), np.zeros(v1.shape[1])

    for j in range(v1.shape[1]):
        xs0 = np.arctan(-b0[0]/b0[2])
        if CanWeApplyDirectlyTheFormula(gamma_RTK, xs0):
            new_cos0 = np.sqrt(D**2+v0[:, j]**2)*(np.cos(gamma_RTK)*b0[0]+np.sin(gamma_RTK)*b0[2])/D
            m0_trpz[j], norm0[j], grad0[j] = np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp0[:, j]/new_cos0), np.abs(np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp0[:, j]/new_cos0)),0
        else:
            m0_trpz[j], norm0[j], grad0[j] = TrapIntegration(xs0, gamma_RTK, proj_interp0[:, j], v0[:, j], b0, D)
        xs1 = np.arctan(-b1[0]/b1[2])
        if CanWeApplyDirectlyTheFormula(gamma_RTK, xs1):
            new_cos1 = np.sqrt(D**2+v1[:, j]**2)*(np.cos(gamma_RTK)*b1[0]+np.sin(gamma_RTK)*b1[2])/D
            m1_trpz[j], norm1[j], grad1[j] = np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp1[:, j]/new_cos1), np.abs(np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp1[:, j]/new_cos1)),0
        else:
            m1_trpz[j], norm1[j], grad1[j] = TrapIntegration(xs1, gamma_RTK, proj_interp1[:, j], v1[:, j], b1, D)
    return m0_trpz, m1_trpz, norm0, norm1, grad0, grad1


def TestForSingularity(angle, xs):  #Find the indices of the angles surrounding the singularity
    a = 0
    b = 0
    if len(np.where(angle >= xs)[0]) != len(angle) and len(np.where(angle <= xs)[0]) != len(angle):
        if angle[1]-angle[0] > 0:
            a = np.where(angle <= xs)[-1][-1]
            b = np.where(angle >= xs)[-1][0]
        else:
            a = np.where(angle <= xs)[-1][0]
            b = np.where(angle >= xs)[-1][-1]
    return a, b


def TrapIntegration(xs, gamma, ar, v, xe, D):  #Perform numerical integration using trapezoidale rule
    h = np.abs(gamma[1]-gamma[0])
    g = D*ar/(np.sqrt(D**2+v**2)*(np.cos(gamma)*xe[0]+np.sin(gamma)*xe[2]))
    ii, iii = TestForSingularity(gamma, xs)
    f_lin = D * ar / np.sqrt(D**2+v**2)
    f_int = interpolate.interp1d(gamma,f_lin)
    grad = (f_int(xs+h/2)-f_int(xs-h/2))/h

    if np.abs(gamma[ii]-xs) < np.abs(gamma[iii]-xs):
        #summ1 = h*((g[0]+g[ii-1])/2 + np.sum(g[1:ii-1]))
        #summ2 = h*((g[iii]+g[-1])/2 + np.sum(g[iii+1:-1]))
        #summ = summ1+summ2
        summ = np.trapz(g[0:ii],gamma[0:ii],gamma[1]-gamma[0])+np.trapz(g[iii:],gamma[iii:],gamma[1]-gamma[0])
        a = (f_lin[iii]-f_lin[ii-1])/(gamma[iii]-gamma[ii-1])
        b = f_lin[iii]-a*gamma[iii]
        c = ((np.cos(gamma[iii])*xe[0]+np.sin(gamma[iii])*xe[2])-(np.cos(gamma[ii-1])*xe[0]+np.sin(gamma[ii-1])*xe[2]))/(gamma[iii]-gamma[ii-1])
        d = (np.cos(gamma[iii])*xe[0]+np.sin(gamma[iii])*xe[2]) - c*gamma[iii]
        rest_ii = (a*(c*gamma[ii-1]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[ii-1]+d)))
        rest_iii = (a*(c*gamma[iii]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[iii]+d)))
        rest = (rest_iii-rest_ii)/c**2
        norm = np.abs(np.trapz(g[0:ii],gamma[0:ii],gamma[1]-gamma[0]))+np.abs(np.trapz(g[iii:],gamma[iii:],gamma[1]-gamma[0])) + np.abs(rest)
    else:
        #summ1 = h*((g[0]+g[ii])/2 + np.sum(g[1:ii]))
        #summ2 = h*((g[iii+1]+g[-1])/2 + np.sum(g[iii+2:-1]))
        #summ = summ1+summ2
        summ = np.trapz(g[0:ii+1],gamma[0:ii+1],gamma[1]-gamma[0])+np.trapz(g[iii+1:],gamma[iii+1:],gamma[1]-gamma[0])
        a = (f_lin[iii+1]-f_lin[ii])/(gamma[iii+1]-gamma[ii])
        b = f_lin[ii]-a*gamma[ii]
        c = ((np.cos(gamma[iii+1])*xe[0]+np.sin(gamma[iii+1])*xe[2])-(np.cos(gamma[ii])*xe[0]+np.sin(gamma[ii])*xe[2]))/(gamma[iii+1]-gamma[ii])
        d = (np.cos(gamma[ii])*xe[0]+np.sin(gamma[ii])*xe[2]) - c*gamma[ii]
        rest_ii = (a*(c*gamma[ii]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[ii]+d)))
        rest_iii = (a*(c*gamma[iii+1]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[iii+1]+d)))
        rest = (rest_iii-rest_ii)/c**2
        norm = np.abs(np.trapz(g[0:ii+1],gamma[0:ii+1],gamma[1]-gamma[0]))+np.abs(np.trapz(g[iii+1:],gamma[iii+1:],gamma[1]-gamma[0])) + np.abs(rest)

    return summ+rest, norm, grad


def CheckPairOverlapCondition(geometryArray, fsMatrices, directionProj, projSize, projSpacing, projOrigin, sourcePos, idx0, idx1):
    # Compute the two detectors coordinates RTK angles
    Det0, Det1, gamma_RTK = ComputeDetectorsPositionFast(geometryArray, fsMatrices, directionProj, projSize, projSpacing, projOrigin, idx0, idx1)

    # Compute intersection between the two detectors and all possible M points
    PMs = ComputeNewFrameAndMPoints(geometryArray, sourcePos, Det0, Det1, projSize, projSpacing, idx0, idx1)

    v_det = projOrigin[1] + geometryArray[4, idx0]-geometryArray[8, idx0] + (np.arange(projSize[1])*projSpacing[1])*directionProj[1, 1]
    M_ACC, n0, b0, n1, b1 = ComputeAllInOneFunction(geometryArray, sourcePos, PMs, gamma_RTK, v_det, idx0, idx1)
    if len(n0) == 0:
        return 0
    else:
        return idx1-idx0
