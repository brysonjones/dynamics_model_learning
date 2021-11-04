
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import glob
import math
import numpy as np
import os
import time

def plotState(timeVec, stateMatrix, name='test'):
    # Plots in 3 Dimensions
    x = stateMatrix[0]
    y = stateMatrix[1]
    z = stateMatrix[2]

    #Make our plot with subplots below
    fig = plt.figure(1, figsize=(5,8))
    spec = gridspec.GridSpec(ncols=1, nrows=4, height_ratios=[3, 1, 1, 1])
    ax0 = fig.add_subplot(spec[0], projection='3d')
    ax0.plot3D(x, y, z, 'g')
    ax0.plot3D(x[-1], y[-1], z[-1], 'or')
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    ax0.set_zlabel('Alt (m)')
    ax0.set_title('Quadcopter Trajectory')

    ax1 = fig.add_subplot(spec[1])
    ax1.plot(timeVec, x)
    ax1.set_xlabel('t (sec)')
    ax1.set_ylabel('x (m)')

    ax2 = fig.add_subplot(spec[2])
    ax2.plot(timeVec, y)
    ax2.set_xlabel('t (sec)')
    ax2.set_ylabel('y (m)')

    ax3 = fig.add_subplot(spec[3])
    ax3.plot(timeVec, z)
    ax3.set_xlabel('t (sec)')
    ax3.set_ylabel('Alt (m)')

    plt.draw()
    plt.pause(0.001)
    
    fileName = name+'_states.png'
    plt.savefig(fileName)
    
def animateState(timeVec, stateMatrix, name='test'):
    # Plots in 3 Dimensions
    x = stateMatrix[0]
    y = stateMatrix[1]
    z = stateMatrix[2]

    dataLen = len(x)

    maxFrames = 100;
    numFrames = min(maxFrames, dataLen)
    frameSep = math.floor(dataLen/numFrames)
    
    #Set up the plotting axis
    fig = plt.figure()
    plt.ion()
    plt.show()
    ax = plt.axes(projection='3d')
    
    for ii in range(numFrames):
        #Find the new end index
        maxInd = (ii + 1) * frameSep - 1
        
        #Do new plotting
        plt.cla()
        ax.plot3D(x[:maxInd+1], y[:maxInd+1], z[:maxInd+1], 'g')
        ax.plot3D(x[maxInd], y[maxInd], z[maxInd], 'or')

        #Need to do this every time we clear
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Alt (m)')
        #ax.set_xlim([min(x), max(x)]) #This makes the axes static. 
        #ax.set_ylim([min(y), max(y)]) #I like the motion, so commented
        #ax.set_zlim([min(z), max(z)])
        ax.set_title('Quadcopter Trajectory')

        #And the plotting and saving stuff
        plt.draw()
        plt.pause(0.001)
        fileName = name+'_aStmp_%03d.png' % ii
        plt.savefig(fileName)

    #Now we need to assemble them into a gif
    searchName = name+'_aStmp_*.png'
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(searchName))]
    img.save(fp=name+'.gif', format='GIF', append_images=imgs,
             save_all=True, duration=50, loop=0)

    #And delete all the images
    for f in glob.glob(searchName):
        os.remove(f)
        

