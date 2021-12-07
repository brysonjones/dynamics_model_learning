
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import glob
import math
import numpy as np
import os
import time

def plotState(timeVec, stateMatrix, name='test', stateMatrix2='none', labels='none'):
    # Plots in 3 Dimensions
    x = stateMatrix[:,0]
    y = stateMatrix[:,1]
    z = stateMatrix[:,2]

    has2states = not(isinstance(stateMatrix2, str))
    if has2states:
        x2 = stateMatrix2[:,0]
        y2 = stateMatrix2[:,1]
        z2 = stateMatrix2[:,2]
    
    #Make our plot with subplots below
    fig = plt.figure(1, figsize=(5,8))
    spec = gridspec.GridSpec(ncols=1, nrows=4, height_ratios=[3, 1, 1, 1])
    ax0 = fig.add_subplot(spec[0], projection='3d')
    if has2states:
        ax0.plot3D(x, y, z, 'k')
        ax0.plot3D(x2, y2, z2, 'g')
        ax0.plot3D(x[-1], y[-1], z[-1], 'ok')
        ax0.plot3D(x2[-1], y2[-1], z2[-1], 'og')
    else:
        ax0.plot3D(x, y, z, 'g')
        ax0.plot3D(x[-1], y[-1], z[-1], 'og')
        
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    ax0.set_zlabel('Alt (m)')
    ax0.set_title('Quadcopter Trajectory')

    ax1 = fig.add_subplot(spec[1])
    ax1.plot(timeVec, x, 'k')
    if has2states:
        ax1.plot(timeVec, x, 'k')
        ax1.plot(timeVec, x2, 'g')
        ax1.plot(timeVec[-1], x[-1], 'ok')
        ax1.plot(timeVec[-1], x2[-1], 'og')
    else:
        ax1.plot(timeVec, x, 'g')
        ax1.plot(timeVec[-1], x[-1], 'og')
    ax1.set_xlabel('t (sec)')
    ax1.set_ylabel('x (m)')

    ax2 = fig.add_subplot(spec[2])
    if has2states:
        ax2.plot(timeVec, y, 'k')
        ax2.plot(timeVec, y2, 'g')
        ax2.plot(timeVec[-1], y[-1], 'ok')
        ax2.plot(timeVec[-1], y2[-1], 'og')
    else:
        ax2.plot(timeVec, y, 'g')
        ax2.plot(timeVec[-1], y[-1], 'og')
    ax2.set_xlabel('t (sec)')
    ax2.set_ylabel('y (m)')

    ax3 = fig.add_subplot(spec[3])
    if has2states:
        ax3.plot(timeVec, z, 'k')
        ax3.plot(timeVec, z2, 'g')
        ax3.plot(timeVec[-1], z[-1], 'ok')
        ax3.plot(timeVec[-1], z2[-1], 'og')
        if not(isinstance(labels, str)): #only make legend if we're given array
            ax3.legend(labels)
    else:
        ax3.plot(timeVec, z, 'g')
        ax3.plot(timeVec[-1], z[-1], 'og')
    ax3.set_xlabel('t (sec)')
    ax3.set_ylabel('Alt (m)')

    plt.draw()
    plt.pause(0.001)
    
    fileName = name+'_states.png'
    plt.savefig(fileName)
    
def animateState(timeVec, stateMatrix, name='test', stateMatrix2='None', labels='none'):
    # Plots in 3 Dimensions
    x = stateMatrix[:,0]
    y = stateMatrix[:,1]
    z = stateMatrix[:,2]

    has2states = not(isinstance(stateMatrix2, str))
    if has2states:
        x2 = stateMatrix2[:,0]
        y2 = stateMatrix2[:,1]
        z2 = stateMatrix2[:,2]
        
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
        if has2states:
            ax.plot3D(x[:maxInd+1], y[:maxInd+1], z[:maxInd+1], 'k')
            ax.plot3D(x2[:maxInd+1], y2[:maxInd+1], z2[:maxInd+1], 'g')
            ax.plot3D(x[maxInd], y[maxInd], z[maxInd], 'ok')
            ax.plot3D(x2[maxInd], y2[maxInd], z2[maxInd], 'og')
            #only make legend if we're given array
            if not(isinstance(labels, str)): 
                ax.legend(labels)
        else:
            ax.plot3D(x[:maxInd+1], y[:maxInd+1], z[:maxInd+1], 'g')
            ax.plot3D(x[maxInd], y[maxInd], z[maxInd], 'og')

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
        
def visualize(timeVec, stateMatrix, name='test', stateMatrix2='none', labels='none'):
    # Plots in 3 Dimensions
    x = stateMatrix[:,0]
    y = stateMatrix[:,1]
    z = stateMatrix[:,2]
    
    has2states = not(isinstance(stateMatrix2, str))
    if has2states:
        x2 = stateMatrix2[:,0]
        y2 = stateMatrix2[:,1]
        z2 = stateMatrix2[:,2]

        #Get min and maxes for plotting
        minX = min(np.concatenate((x, x2)))
        maxX = max(np.concatenate((x, x2)))
        minY = min(np.concatenate((y, y2)))
        maxY = max(np.concatenate((y, y2)))
        minZ = min(np.concatenate((z, z2)))
        maxZ = max(np.concatenate((z, z2)))
    else:
        #Get min and maxes for plotting  
        minX = min(x)
        maxX = max(x)
        minY = min(y)
        maxY = max(y)
        minZ = min(z)
        maxZ = max(z)

    dataLen = len(x)

    maxFrames = 100;
    numFrames = min(maxFrames, dataLen)
    frameSep = math.floor(dataLen/numFrames)

    #Set up the plotting axis
    plt.ion()
    plt.show()
    fig = plt.figure(1, figsize=(5,8))
    spec = gridspec.GridSpec(ncols=1, nrows=4, height_ratios=[3, 1, 1, 1])
    ax0 = fig.add_subplot(spec[0], projection='3d')
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax3 = fig.add_subplot(spec[3])
    
    for ii in range(numFrames):
        #Find the new end index
        maxInd = (ii + 1) * frameSep - 1
        
        #Do new plotting
        ax0.clear()
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        #Make our plot with subplots below
        if has2states:
            ax0.plot3D(x[:maxInd+1], y[:maxInd+1], z[:maxInd+1], 'k')
            ax0.plot3D(x[maxInd], y[maxInd], z[maxInd], 'ok')
            ax0.plot3D(x2[:maxInd+1], y2[:maxInd+1], z2[:maxInd+1], 'g')
            ax0.plot3D(x2[maxInd], y2[maxInd], z2[maxInd], 'og')
        else:
            ax0.plot3D(x[:maxInd+1], y[:maxInd+1], z[:maxInd+1], 'g')
            ax0.plot3D(x[maxInd], y[maxInd], z[maxInd], 'og')
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        ax0.set_zlabel('Alt (m)')
        ax0.set_title('Quadcopter Trajectory')


        if has2states:
            ax1.plot(timeVec[:maxInd+1], x[:maxInd+1], 'k')
            ax1.plot(timeVec[maxInd],    x[maxInd], 'ok')
            ax1.plot(timeVec[:maxInd+1], x2[:maxInd+1], 'g')
            ax1.plot(timeVec[maxInd],    x2[maxInd], 'og')         
        else:
            ax1.plot(timeVec[:maxInd+1], x[:maxInd+1], 'g')
            ax1.plot(timeVec[maxInd],    x[maxInd], 'og')
        ax1.set_ylim([minX, maxX])
        ax1.set_xlabel('t (sec)')
        ax1.set_ylabel('x (m)')


        if has2states:
            ax2.plot(timeVec[:maxInd+1], y[:maxInd+1], 'k')
            ax2.plot(timeVec[maxInd],    y[maxInd], 'ok')
            ax2.plot(timeVec[:maxInd+1], y2[:maxInd+1], 'g')
            ax2.plot(timeVec[maxInd],    y2[maxInd], 'og')

        else:
            ax2.plot(timeVec[:maxInd+1], y[:maxInd+1], 'g')
            ax2.plot(timeVec[maxInd],    y[maxInd], 'og')
        ax2.set_ylim([minY, maxY])
        ax2.set_xlabel('t (sec)')
        ax2.set_ylabel('y (m)')


        if has2states:
            ax3.plot(timeVec[:maxInd+1], z[:maxInd+1], 'k')
            ax3.plot(timeVec[:maxInd+1], z2[:maxInd+1], 'g')
            ax3.plot(timeVec[maxInd],    z[maxInd], 'ok')
            ax3.plot(timeVec[maxInd],    z2[maxInd], 'og')

            #only make legend if we're given array
            if not(isinstance(labels, str)): 
                ax3.legend(labels)
        else:
            ax3.plot(timeVec[:maxInd+1], z[:maxInd+1], 'g')
            ax3.plot(timeVec[maxInd],    z[maxInd], 'og')
            
        ax3.set_ylim([minZ, maxZ])
        ax3.set_xlabel('t (sec)')
        ax3.set_ylabel('Alt (m)')
            
        #And the plotting and saving stuff
        plt.draw()
        plt.pause(0.001)
        fileName = name+'_vistmp_%03d.png' % ii
        plt.savefig(fileName)

    #Now we need to assemble them into a gif
    searchName = name+'_vistmp_*.png'
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(searchName))]
    img.save(fp=name+'.gif', format='GIF', append_images=imgs,
             save_all=True, duration=50, loop=0)

    #Also, save the last one. We may actually want it alone
    plt.savefig(name+'_states.png')
    #And delete all the images
    for f in glob.glob(searchName):
        os.remove(f)

def accCompare(timeVec, accSet1, accSet2, runName='test', accTypeName='test'):
    # Plot the accels to compare
    lineNames = ['acc x', 'acc y', 'acc z', 'ang acc x', 'ang acc y', 'ang acc z']

    #Make our plot with subplots below
    fig = plt.figure(1, figsize=(10,9))
    spec = gridspec.GridSpec(ncols=3, nrows=2)
    for ii in range(0, 6):
        ax = fig.add_subplot(spec[ii])
        ax.plot(timeVec, accSet1[:,ii], 'k--')
        ax.plot(timeVec, accSet2[:,ii], 'g')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('acc val')
        ax.set_title(lineNames[ii])

    
    fileName = runName+'_accCompare_'+accTypeName+'.png'
    plt.savefig(fileName)
    plt.show()
    
