from sklearn.linear_model import LinearRegression
import numpy as np
import sys
#Set up environment
sys.path.append('.')
sys.path.append('..')
sys.path.append('../Visualizers')
sys.path.append('../whitebox')
from visTool import *
from helpers import *

#Set up variables
sys.path.append("../dataLoader")
from dataLoader.DataLoader import DataLoader
DL = DataLoader('../processed_data')

print('Loading data')
DL.load_selected_data('2021-02-05-14-00-56')#"2021-02-03-16-10-37")
timeVec = DL.get_time_values()
state   = DL.get_state_data()
accx = DL.data['acc x'].values
accy = DL.data['acc y'].values
accz = DL.data['acc z'].values
acc = np.concatenate((accx.reshape((-1,1)),
                      accy.reshape((-1,1)),
                      accz.reshape((-1,1))), axis=1)

mass = 0.752 #kg (from paper)
sLen = 0.126 #m half motor-motor len (https://armattanquads.com/chameleon-ti-6-inch/)

print('Calculating kt (prop thrust coefficient)')
kt = getKt(state, accz, mass)
print('Using kt value ' + str(round(kt, 5)))

#Calculate the J, km values
angaccx = DL.data['ang acc x'].values
angaccy = DL.data['ang acc y'].values
angaccz = DL.data['ang acc z'].values
angAcc = np.concatenate((angaccx.reshape((-1,1)),
                         angaccy.reshape((-1,1)),
                         angaccz.reshape((-1,1))), axis=1)
print('Calculating inertia matrix')
Jxx = getJxx(state, angAcc, kt, sLen)
print('Using Jxx = ' + str(Jxx))
Jyy = Jxx
Jzz = 2*Jxx
J = np.diag([Jxx, Jyy, Jzz])

#Now we need the km value
print('Calculating prop moment coefficient')
km = getKm(state, angAcc, Jzz)
print('Using km = ' + str(km))

#Now we can repredict the data and see what our error looks like
accPred, angAccPred = NewtEulPredict(state, mass, sLen, kt, km, J)

#Last, to plot (we probably also want some sort of a metric
totalAcc = np.concatenate((acc, angAcc), axis=1)
totalNE = np.concatenate((accPred, angAccPred), axis=1)

accCompare(timeVec, totalAcc, totalNE)


#Recalculate with new values - should come out the same
#print('Calculating kt (prop thrust coefficient)')
#kt2 = getKt(state, accPred[:, 2], mass)
#print('Using kt value ' + str(round(kt2, 5)))

#print('Calculating inertia matrix')
#Jxx2 = getJxx(state, angAccPred, kt2, sLen)
#print('Using Jxx = ' + str(Jxx2))
#Jzz2 = 2*Jxx2

#print('Calculating prop moment coefficient')
#km2 = getKm(state, angAccPred, Jzz2)
#print('Using km = ' + str(km2))

