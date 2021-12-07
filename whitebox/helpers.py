import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

def getKt(stateMatrix, accz, mass):
    #Calculate kt using a linear regression
    
    #accz = a*kt + b
    #regression will be using a, accz-b
    a, b = getABlooperKt(stateMatrix, mass)
    otherSide = accz.reshape((-1,1)) - b #column vecs
    
    kt, _, _, _ = np.linalg.lstsq(a, otherSide, rcond=-1)

    #z = np.polyfit(a.flatten(), otherSide.flatten(), 1)
    
    return kt[0,0] #the coefficient
    
def getABlooperKt(stateMatrix, mass):
    #Run through all sets and find the a and b values

    numRows = stateMatrix.shape[0]

    #stash values for each row in the state
    aVec = np.zeros((numRows, 1))
    bVec = np.zeros((numRows, 1))
    
    for ii in range(0, numRows):
        thisState = stateMatrix[ii, :]
        a,b = getABkt(thisState, mass)
        aVec[ii] = a
        bVec[ii] = b

    #and return
    return aVec, bVec

def getABkt(stateVec, mass):
    #Do the actual calculations
    
    #Extract the inputs
    pos   = stateVec[0:3]
    vel   = stateVec[3:6]
    quat  = stateVec[6:10]
    omega = stateVec[10:13]
    motor = stateVec[13:17]

    #Prep a
    #a = sum(motors)/m
    a = sum(motor) / mass

    #Prep the b terms
    #b = Q^T*[0;0;-g][2] - (w x v)[2]    
    Q = getRotMat(quat)
    gVec = np.zeros((3,1))
    gVec[2] = -9.81
    
    crossTerm = np.cross(omega, vel)
    bVec = Q.transpose() @ gVec - crossTerm.reshape((3,1))
    b = bVec[2] #only want the z component
    
    return a,b

def getJxx(stateMatrix, angAcc, kt, sLen):
    #Performs a regression to get Jxx value for inertia matrix
    # assumes Jxx = Jyy = 1/2 Jzz
    
    a, b = getABlooperJxx(stateMatrix, angAcc, kt, sLen)

    #plt.scatter(a,b)
    #plt.show()
    #solving for a*J_11 = b
    Jxx, _, _, _ = np.linalg.lstsq(a, b, rcond=-1)
    
    return Jxx[0,0]

def getABlooperJxx(stateMatrix, angAcc, kt, sLen):
    #Run through all sets and find the a and b values

    numRows = stateMatrix.shape[0]

    #stash values for each row in the state
    aVec = np.zeros((numRows*2, 1))
    bVec = np.zeros((numRows*2, 1))
    
    for ii in range(0, numRows):
        thisState = stateMatrix[ii, :]
        angAccVec = angAcc[ii, :]
        
        a1,b1, a2,b2 = getABJxx(thisState, angAccVec, kt, sLen)
        aVec[ii*2] = a1
        bVec[ii*2] = b1
        aVec[ii*2+1] = a2
        bVec[ii*2+1] = b2

    #and return
    return aVec, bVec

def getABJxx(stateVec, angAccVec, kt, sLen):
    #Do the actual calculations
    
    #Extract the inputs
    pos   = stateVec[0:3]
    vel   = stateVec[3:6]
    quat  = stateVec[6:10]
    omega = stateVec[10:13]
    motor = stateVec[13:17]
    
    theta = math.pi/4
    rotMat = np.zeros((3,3)) #for rotating from Bgiven to Bprincipal
    rotMat[0,0] = math.cos(theta)
    rotMat[0,1] = math.sin(theta)
    rotMat[1,0] = -math.sin(theta)
    rotMat[1,1] = math.cos(theta)
    rotMat[2,2] = 1

    angAccVecPrincipal = rotMat @ angAccVec.reshape((3,1))
    
    angAccX = angAccVecPrincipal[0]
    angAccY = angAccVecPrincipal[1]

    omega = rotMat @ omega.reshape((3,1)) #need in principal axis
    
    #Prep a
    #Comes from lines
    #w\dot_1 = 1/J_11*s*kt*(u4-u1) - w_2w_3
    #(w\dot_1 + w_2w_3)J_11 = s*kt*(u4-u1)
    # ------  a  -----        ---  b ----
    #w\dot_2 = 1/J_22*s*kt*(u3-u2) + w_1w_3
    #
    #(w\dot_2 - w_1w_3)J_22 = s*kt*(u3-u2)
    # ------  a  -----        ---  b ----
    #
    #Note: We're multiplying the J term across to make the regression easier
    # more reliable than regressing for the inverse of our variable... ew
    a1 = angAccX + omega[1]*omega[2]
    a2 = angAccY - omega[0]*omega[2]

    #Prep the b terms
    #b = s*kt*(u2-u4)
    #b = s*kt*(u3-u1)
    
    b1 = sLen * kt * (motor[3] - motor[0])
    b2 = sLen * kt * (motor[2] - motor[1])
    
    return a1,b1, a2,b2

def getKm(stateMatrix, angAcc, Jzz):
    #Performs a regression to get Jxx value for inertia matrix
    # assumes Jxx = Jyy = 1/2 Jzz
    
    a, b = getABlooperKm(stateMatrix, angAcc, Jzz)

    #solving for a*J_11 = b
    km, _, _, _ = np.linalg.lstsq(a, b, rcond=-1)
    
    return km[0,0]

def getABlooperKm(stateMatrix, angAcc, Jzz):
    #Run through all sets and find the a and b values

    numRows = stateMatrix.shape[0]

    #stash values for each row in the state
    aVec = np.zeros((numRows, 1))
    bVec = np.zeros((numRows, 1))
    
    for ii in range(0, numRows):
        thisState = stateMatrix[ii, :]
        angAccVec = angAcc[ii, :]
        
        a,b = getABkm(thisState, angAccVec, Jzz)
        aVec[ii] = a
        bVec[ii] = b

    #and return
    return aVec, bVec

def getABkm(stateVec, angAccVec, Jzz):
    #Do the actual calculations
    
    #Extract the inputs
    pos   = stateVec[0:3]
    vel   = stateVec[3:6]
    quat  = stateVec[6:10]
    omega = stateVec[10:13]
    motor = stateVec[13:17]

    angAccZ = angAccVec[2]
    #based on equation
    # w\dot_3 = 1/J_33*(u1-u2+u3-u4)*km
    # -- b --   -------   a -------
    
    #Prep a
    a = 1/Jzz * (motor[0] - motor[1] - motor[2] + motor[3])

    #Prep the b terms
    #b = angAccZ
    b = angAccZ
    
    return a,b


def NewtEulPredict(state, mass, sLen, kt, km, J):
    #Do the prediction for all input states
    numStates  = state.shape[0]
    accPred    = np.zeros((numStates, 3))
    angAccPred = np.zeros((numStates, 3))

    for ii in range(0, numStates):
        newAcc, newAngAcc = doNewtEul(state[ii, :], mass, sLen, kt, km, J)

        #reshape to avoid issues and plug 'em in!
        newAcc    = newAcc.reshape((1,3))
        newAngAcc = newAngAcc.reshape((1,3))

        accPred[ii,:]    = newAcc
        angAccPred[ii,:] = newAngAcc
    
    return accPred, angAccPred

def doNewtEul(stateVec, mass, sLen, kt, km, J):
    #Do the actual Newton Euler prediction for a particular state
    # v\dot = 1/m*F-wxv
    # w\dot = inv(J)*(tau - cross(w, Jw))
    
    #Extract the inputs
    pos   = stateVec[0:3]
    vel   = stateVec[3:6]
    quat  = stateVec[6:10]
    omega = stateVec[10:13]
    motor = stateVec[13:17]

    #First do acc
    F = np.zeros((3,1))
    F[2] = kt*sum(motor)
    #don't forget gravity
    g = 9.81;
    gVec = np.zeros((3,1))
    gVec[2] = -g*mass
    
    Q = getRotMat(quat)
    
    Fterm = F + Q.transpose() @ gVec
    crossTerm = np.cross(omega.reshape((1,3)), vel.reshape((1,3)))

    acc = 1/mass * Fterm - crossTerm.reshape((3,1))
    
    #And angular acc
    theta = math.pi/4
    rotMat = np.zeros((3,3))
    rotMat[0,0] = math.cos(theta)
    rotMat[0,1] = -math.sin(theta)
    rotMat[1,0] = math.sin(theta)
    rotMat[1,1] = math.cos(theta)
    rotMat[2,2] = 1
    
    tauMat = np.zeros((3,4))
    tauMat[0,1] = sLen * kt
    tauMat[0,3] = -sLen * kt
    tauMat[1,0] = -sLen * kt
    tauMat[1,2] = sLen * kt
    tauMat[2,0] = km
    tauMat[2,1] = -km
    tauMat[2,2] = km
    tauMat[2,3] = -km
    tauTerm = tauMat @ motor.reshape((4,1))

    omegaPrinc = rotMat @ omega.reshape((3,1))
    Jw = J @ omegaPrinc
    crossTerm = np.cross(omegaPrinc.reshape((1,3)), Jw.reshape((1,3)))
    angAccPrinc = np.linalg.inv(J) @ (tauTerm - crossTerm.reshape((3,1)))

    angAcc = rotMat.transpose() @ angAccPrinc
    
    return acc, angAcc

def hat(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def Lmat(q):
    q_vec_hat = hat(q[1:])
    
    L = np.zeros((4,4))
    L[0,0] = q[0]
    L[0,1:] = -q[1:]
    L[1:,0] = q[1:]
    L[1:,1:] = q[0]*np.eye(3) + q_vec_hat
    return L

def Rmat(q):
    q_vec_hat = hat(q[1:])
    
    R = np.zeros((4,4))
    R[0,0] = q[0]
    R[0,1:] = -q[1:]
    R[1:,0] = q[1:]
    R[1:,1:] = q[0]*np.eye(3) - q_vec_hat
    return R

def Hmat():
    #not sure this needs to be a function, but it's handy
    H = np.zeros((4,3))
    H[1,0] = 1
    H[2,1] = 1
    H[3,2] = 1
    return H

def quat_dot(q, w):
    q_vec_hat = hat(q[1:])

    L = Lmat(q)

    # np.vstack term is H from the notes [0; I(3)]
    return 0.5 * L @ np.vstack([np.zeros(3), np.eye(3)]) @ w

def getRotMat(quat):
    L = Lmat(quat)
    R = Rmat(quat)
    H = Hmat()
    
    Q = H.transpose() @ L @ R.transpose() @ H
    return Q
