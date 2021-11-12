import os
import re
from tqdm import tqdm
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial

W = np.ones((16,16))
W /= np.sum(W.flatten())

cmap = matplotlib.cm.jet.copy()
cmap.set_bad('white', 1.)

cmap_dens = matplotlib.cm.binary.copy()
cmap_dens.set_bad('white', 1.)

cmap_sep = matplotlib.cm.Reds.copy()
cmap_sep.set_bad('white', 1.)

cmap_mix = matplotlib.cm.Blues.copy()
cmap_mix.set_bad('white', 1.)

def load_data(res_folder, components_equal=True):

    dirs = os.listdir(res_folder)

    assert(len(dirs) > 0)

    XB = []
    XR = []
    ST = []
    AA = []
    NN = []

    pattern = re.compile(r'lattice-([+-]?(?:[0-9]*[.])?[0-9]+)-([+-]?(?:[0-9]*[.])?[0-9]+)')

    for di in tqdm(dirs):

        m = pattern.match(di)
        if not m:
            raise(ValueError("Can't get values"))

        XB.append(float(m.group(1)))
        XR.append(float(m.group(2)))

        files = os.listdir(res_folder / di)

        A = [np.genfromtxt(res_folder / di / file, delimiter=',', dtype=np.int8) 
                for file in files]

        AA.append(A)
        NN.append(di)

        st = analyse_phase_sep(A, name=di, components_equal=components_equal)
        ST.append(st)

    XB = np.asarray(XB)
    XR = np.asarray(XR)
    ST = np.asarray(ST)

    data = np.concatenate([XB[:, np.newaxis], XR[:, np.newaxis], ST[:, np.newaxis]], axis=1)

    # data = data[np.lexsort((data[:,1], data[:,0]))]

    return data, AA

def iso_fraction_cut(data, phi, thr):
    ind = set()
    for i in np.arange(data.shape[0]):
        if np.abs(data[i,0] + data[i,1] - phi) < thr:
            ind.add(i)

    ind = np.array(list(ind))
    return ind

def phase_sep_boundary(data):
    # dist = sp.spatial.distance.pdist(data[:,0:1], metric='euclidean')
    # dist = sp.spatial.distance.squareform(dist, force='tomatrix')

    # plt.figure()
    # plt.imshow(dist)
    # plt.colorbar()

    ind = set()
    thr = 0.1
    for i in np.arange(data.shape[0]):
        for j in np.arange(data.shape[0]):
            d = np.sqrt((data[i,0] - data[j,0])**2 + (data[i,1] - data[j,1])**2);
            # d = dist[i,j]
            if (d > 0) and (d < thr):
                if int(data[i,2]) != int(data[j,2]):                
                    ind.add(i)
                    # break

    ind = np.array(list(ind))
    return ind

def plot_phase_sep(data, ind=None):
    plt.figure()
    plt.plot(data[data[:,2] == 0,0], data[data[:,2] == 0,1], 's', color='b', alpha=0.2)
    plt.plot(data[data[:,2] == 1,0], data[data[:,2] == 1,1], 's', color='r', alpha=0.2)
    if ind is not None:
        data = data[ind,:]
        plt.plot(data[data[:,2] == 0,0], data[data[:,2] == 0,1], 's', color='b')
        plt.plot(data[data[:,2] == 1,0], data[data[:,2] == 1,1], 's', color='r')
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

def analyse_phase_sep(AA, name, doplot='none', data=None, components_equal=True):

    if doplot == 'none':
        doplot = False
    elif doplot == 'all':
        doplot = True
    elif doplot == 'unknown':
        pass

    BB = []

    for A in AA:

        if components_equal == True:
            B = np.copy(A)
            B[B > 0] = 1

            B = sp.signal.convolve2d(B, W, 
                mode='same',
                boundary='wrap',
                fillvalue=0)

            BB.append(B)
        else:
            for comp in np.arange(1, np.max(A)):
                B = np.copy(A)
                B = B == comp

                B = sp.signal.convolve2d(B, W, 
                    mode='same',
                    boundary='wrap',
                    fillvalue=0)

                BB.append(B)

    BB = np.concatenate(BB)
    
    AA = np.concatenate(AA)
    AA = AA.flatten()

    xb = np.mean(AA == 1)
    xr = np.mean(AA == 2)
    phi = xr + xb

    AA[AA > 0] = 1

    nbins = 60
    # nbins = np.linspace(-0.1, 1.1, 100)

    hist, edges = np.histogram(BB.flatten(), bins=nbins)
    centers = (edges[1:] + edges[0:-1])/2

    # for i, (h, c) in enumerate(zip(hist, centers)):
    #     hist[i] *= abs(c - phi)

    # k = 1 
    # k = BB.size
    k = 0

    E = - np.log10(hist + k)
    
    nans = np.logical_or(np.isnan(E), np.isinf(E))
    x = centers[~nans]
    E = E[~nans]

    # print(E)

    pol2 = Polynomial.fit(x, E, deg=2)
    pol4 = Polynomial.fit(x, E, deg=4)

    dpol4 = pol4.deriv()
    ddpol4 = dpol4.deriv()
    # print(dpol4)
    # print(dpol4.roots())
    extrema = []
    maxmin = []
    for r in dpol4.roots():
        if not np.isreal(r):
            continue

        r = np.real(r)

        if r < centers[0] or r > centers[-1]:
            continue

        if ddpol4(r) < 0:
            extrema.append(r)
            maxmin.append(1)
        elif ddpol4(r) > 0:
            extrema.append(r)
            maxmin.append(-1)
     
    ext = np.sort(np.asarray(extrema))

    CC = centers[-1] - centers[0]

    if ext.size == 0 or (ext[0] - centers[0]) / CC >= 0.1:
        if dpol4(centers[0]) < 0:
            extrema.append(centers[0])
            maxmin.append(1)
        elif dpol4(centers[0]) > 0:
            extrema.append(centers[0])
            maxmin.append(-1)

    if ext.size == 0 or (- ext[-1] + centers[-1]) / CC >= 0.1:
        if dpol4(centers[-1]) > 0:
            extrema.append(centers[-1])
            maxmin.append(1)
        elif dpol4(centers[-1]) < 0:
            extrema.append(centers[-1])
            maxmin.append(-1)

    maxmin = np.asarray(maxmin, dtype=int)

    extrema = np.asarray(extrema)

    ii = np.argsort(extrema)
    extrema =  extrema[ii]
    maxmin = maxmin[ii]
    
    st = -1
    
    mm = maxmin

    if mm.size == 3:
        if np.all(mm == [1, -1, 1]): # max min max
            st = 0
            if doplot == 'unknown':
                doplot = False
        elif np.all(mm == [-1, 1, -1]): # min max min
            st = 1
            if doplot == 'unknown':
                doplot = False
        
    elif mm.size == 5:
        if np.all(mm == [1, -1, 1, -1, 1]):
            st = 1
            if doplot == 'unknown':
                doplot = False
        elif np.all(mm == [-1, 1, -1, 1, -1]):
            st = 1
            if doplot == 'unknown':
                doplot = False

    if st == -1 and doplot == 'unknown':
        doplot = True

    if st == -1 and doplot == False:
        # if we are not sure, assigne to separated
        st = 1
        
    # older conditions (backup)
    # st = 1 if sep
    # st = pol2.coef[2] < 0 
    # st = (centers[-1] - centers[0]) > 0.8
    # st = len(maxima) > 0

    if st == 1:
        state = "separated"
    elif st == 0:
        state = "mixed"
    else:
        state = "unknown"


    A = np.copy(A)
    A = np.asarray(A, dtype=float)

    if doplot is True:

        A[A == 0] = np.nan

        if st == 0:
            cmp = cmap_mix
            spec = 'bs'
        elif st == 1:
            cmp = cmap_sep
            spec = 'rs'
        else:
            cmp = cmap_dens
            spec = 'ks'

        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.imshow(A, cmap=cmp, vmin=0, vmax=2, interpolation='none')
        # plt.title(name)
        plt.title('lattice')
        plt.colorbar()

        plt.subplot(122)
        plt.imshow(B, cmap=cmap, interpolation='none')
        #plt.title(name)
        plt.title('conv.lattice')
        plt.colorbar()

        plt.figure(figsize=(12,3))
        ax = plt.subplot(131)
        ax.set_aspect('equal')
        # t2 = plt.Polygon(np.asarray([[0,0], [0,1], [1, 0]]), color=(0.9,0.9,0.9))
        # ax.add_patch(t2)

        plt.plot(data[data[:,2] == 0,0], data[data[:,2] == 0,1], 's', color='b', alpha=0.2)
        plt.plot(data[data[:,2] == 1,0], data[data[:,2] == 1,1], 's', color='r', alpha=0.2)

        plt.plot(xb, xr, spec)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('phase_diagram')
        plt.xlabel('vol. frac.')
        plt.ylabel('vol. frac.')

        plt.subplot(132)
        plt.plot(centers, hist / hist.max())
        plt.xlabel('Density')
        plt.ylabel('Probability')
        plt.title(state)

        plt.subplot(133)
        plt.plot(x, E)
        # plt.plot(centers, pol2(centers))
        plt.plot(centers, pol4(centers))
        t = ""
        # if len(maxima) > 0:
        #     t += f"max: " + ', '.join([f'{m:2f}' for m in maxima])
        # if len(minima) > 0:
        #     t += f"min: " + ', '.join([f'{m:2f}' for m in minima])
        plt.title(t)
        plt.xlabel('Density')
        plt.ylabel('Free Energy')
        # plt.title(c.coef)

    return st
