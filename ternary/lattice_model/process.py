import os
import re
from tqdm import tqdm
import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from numpy.polynomial import Polynomial

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = scipy.signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d /= np.sum(gkern2d.flatten())
    return gkern2d

# kernel = []

# if False:
#     for ws in [5, 9, 13, 17, 21]:
#         W = np.ones((ws,ws))
#         W /= np.sum(W.flatten())
#         kernel.append(W)
# else:
#     kernel.append(gkern(1+20,5))
#     # kernel.append(gkern(1+40,5*2))

# def plot_kernel():
#     plt.figure()
#     W = np.zeros_like(kernel[0])
#     for K in kernel:
#         W = W + K
#     W = W / len(kernel)
#     plt.imshow(W)
#     plt.colorbar()
#     # print(np.sum(W.flatten()))



cmap = matplotlib.cm.jet.copy()
cmap.set_bad('white', 1.)

cmap_dens = matplotlib.cm.binary.copy()
cmap_dens.set_bad('white', 1.)

cmap_sep = matplotlib.cm.Reds.copy()
cmap_sep.set_bad('white', 1.)

cmap_mix = matplotlib.cm.Blues.copy()
cmap_mix.set_bad('white', 1.)

def gauss(x, mu, sigma):
    return np.exp(-(((x - mu) / sigma)**2)/2) / (sigma * np.sqrt(2 * np.pi))

def load_data(res_folder, components_equal=False, load_fraction=1):

    dirs = os.listdir(res_folder)

    assert(len(dirs) > 0)

    XB = []
    XR = []
    ST = []
    AA = []
    NN = []

    results = []

    pattern = re.compile(r'lattice-([+-]?(?:[0-9]*[.])?[0-9]+)-([+-]?(?:[0-9]*[.])?[0-9]+)')

    for di in tqdm(dirs):

        if load_fraction < 1:
            if np.random.rand() > load_fraction:
                # skip this data
                continue

        m = pattern.match(di)
        if not m:
            raise(ValueError(f"Can't get values from folder {di}"))

        XB.append(float(m.group(1)))
        XR.append(float(m.group(2)))

        files = os.listdir(res_folder / di)

        A = [np.genfromtxt(res_folder / di / file, delimiter=',', dtype=np.int8) 
                for file in files]

        AA.append(A)
        NN.append(di)

        st, res = analyse_phase_sep(A, name=di, components_equal=components_equal)
        ST.append(st)
        results.append(res)

    XB = np.asarray(XB)
    XR = np.asarray(XR)
    ST = np.asarray(ST)

    data = np.concatenate([XB[:, np.newaxis], XR[:, np.newaxis], ST[:, np.newaxis]], axis=1)

    # data = data[np.lexsort((data[:,1], data[:,0]))]

    return data, AA, results

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

def renormalize_sep_norm(M):
    sep_thr = 0.2
    M[M > 1] = 1
    M = M - sep_thr
    M[M < 0] = M[M < 0] / sep_thr
    M[M > 0] = M[M > 0] / (1 - sep_thr)
    return M

def get_sep_norm(results):
    MM = []
    nc = 2
    for k in range(nc):
        n = len(results)
        M = np.zeros(n)
        for i in np.arange(n):
            M[i] = results[i][k]['sep_norm']
        M = renormalize_sep_norm(M)
        MM.append(M)
    return MM

def plot_phase_sep_norm(data, MM, ind=None):
    
    plt.figure()
    n = data.shape[0]
    for M in MM:
        plt.scatter(data[:,0], data[:,1], c=M,
            marker='s', alpha=0.5, cmap=matplotlib.cm.bwr,
            vmin=-1, vmax=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.colorbar()
    plt.show()

def save_phase_sep_norm(data, MM, file_name):
    MM = [M[:,np.newaxis] for M in MM]
    MM = np.concatenate(MM, axis=1)
    m = np.mean(MM, axis=1)[:, np.newaxis]
    MM = np.concatenate([data, m, MM], axis=1)
    np.savetxt(file_name, MM)

def analyse_phase_sep_core(BB, doplot):

    nbins = 100

    hist, edges = np.histogram(BB.flatten(), bins=nbins)
    centers = (edges[1:] + edges[0:-1])/2
    bin_sizes = (edges[1:] - edges[0:-1])

    hist_pdf = hist / np.sum(hist * bin_sizes)

    mu = np.mean(BB.flatten())
    sigma = np.std(BB.flatten())
    # print(mu, sigma)
    g = gauss(centers, mu, sigma)

    a = mu*(mu*(1-mu)/sigma**2 - 1)
    b = (1-mu)*(mu*(1-mu)/sigma**2 - 1)

    beta = scipy.stats.beta.pdf(centers, a, b)

    # popt, pcov = curve_fit(gauss, centers, hist / np.sum(hist))

    k = 0

    E = - np.log10(hist + k)
    
    nans = np.logical_or(np.isnan(E), np.isinf(E))
    x = centers[~nans]
    E = E[~nans]

    if False:

        # pol2 = Polynomial.fit(x, E, deg=2)
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

        # if st == -1 and doplot == False:
        #     # if we are not sure, assigne to separated
        #     st = 1
    
    else:
        pol4 = None

        d = np.sum(bin_sizes * np.abs(hist_pdf - beta))
        st = d >= 0.2

    # older conditions (backup)
    # st = 1 if sep
    # st = pol2.coef[2] < 0 
    # st = (centers[-1] - centers[0]) > 0.8
    # st = len(maxima) > 0

    res = {
        'sep_norm': d,
        'st': st,
        'doplot': doplot,
        'centers': centers,
        'bin_sizes': bin_sizes,
        'hist': hist,
        'hist_pdf': hist_pdf, 
        'gauss': g,
        'beta': beta,
        'x_energy': x,
        'energy': E,
        'pol4': pol4,
    }

    return res

def analyse_phase_sep(AA, name, doplot='none', data=None, components_equal=False):

    if doplot == 'none':
        doplot = False
    elif doplot == 'all':
        doplot = True
    elif doplot == 'unknown':
        pass

    ## -----

    BB = []

    if components_equal == True:
        nc = 1
        BB = [[],[]]

    else:
        nc = np.max(AA[0])
        BB = [[] for i in range(nc+1)]

    ## -----

    kernel = []
    for i, A in enumerate(AA):

        if components_equal == True:
            A = np.copy(A)
            A[A > 0] = 1

        for comp in np.arange(1, nc+1):
            B = np.copy(A)
            B = B == comp

            if i == 0:
                rho = np.mean(B)
                # if rho > 0.5 we are interested in the density of the rest
                rho = min([rho, 1-rho])
                assert(rho > 0)
                L = B.shape[0]
                # mean particle distance scales like 1/sqrt(concentration)
                l = 7 + 4*np.ceil(1/np.sqrt(rho))
                l = min([l, np.floor(L/np.sqrt(2))])
                W = gkern(int(l), l/3)
                kernel.append(W)

            D = sp.signal.convolve2d(B, kernel[comp-1], 
                mode='same',
                boundary='wrap',
                fillvalue=0)

            BB[comp].append(D)
    
    CC = np.copy(BB)

    AA = np.concatenate(AA)
    AA = AA.flatten()

    xb = np.mean(AA == 1)
    xr = np.mean(AA == 2)
    phi = xr + xb

    # AA[AA > 0] = 1

    ## -----
    results = []
    for comp in np.arange(1, nc+1):
        BB[comp] = np.concatenate(BB[comp])

        res = analyse_phase_sep_core(BB[comp], doplot)
        results.append(res)

    res_df = pd.DataFrame(results)

    ## --- logic

    doplot = np.any(np.asarray(res_df.doplot))
   
    st = np.asarray(res_df.st)
    if np.any(st == 1):
        st = 1
    elif np.any(st == 0):
        st = 0
    else:
        st = -1

    ## ---


    if st == 1:
        state = "separated"
    elif st == 0:
        state = "mixed"
    else:
        state = "unknown"


    A = np.copy(A)
    A = np.asarray(A, dtype=float)

    if doplot:

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

        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(A, cmap=cmp, vmin=0, vmax=2, interpolation='none')
        # plt.title(name)
        plt.title('lattice')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(CC[1][-1], cmap=cmap, interpolation='none')
        #plt.title(name)
        plt.title('conv.lattice')
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(CC[2][-1], cmap=cmap, interpolation='none')
        #plt.title(name)
        plt.title('conv.lattice')
        plt.colorbar()

        plt.figure(figsize=(12,2))
        plt.subplot(131)
        plt.imshow(kernel[0], cmap=cmap, interpolation='none')
        # plt.title(name)
        plt.title('kernel')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(kernel[1], cmap=cmap, interpolation='none')
        # plt.title(name)
        plt.title('kernel')
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
        colors = ['b', 'r']
        for comp in np.arange(0, nc):
            x = results[comp]['centers']
            y = results[comp]['hist_pdf']
            plt.plot(x, y, color=colors[comp], label=f"Hist c. {comp + 1}")
            y = results[comp]['beta']
            plt.plot(x, y, color=colors[comp], label=f"Beta c. {comp + 1}",  linestyle='dashed')
            y = results[comp]['gauss']
            plt.plot(x, y, color=colors[comp], label=f"Gauss c. {comp + 1}",  linestyle='dotted')

        plt.xlabel('Density')
        plt.ylabel('pdf')
        plt.legend()
        plt.title(state)

        plt.subplot(133)
        colors = ['b', 'r']
        for comp in np.arange(0, nc):
            plt.plot(results[comp]['x_energy'], results[comp]['energy'],
                color=colors[comp], label=f"comp. {comp + 1}")

            if results[comp]['pol4'] is not None:
                plt.plot(results[comp]['centers'], results[comp]['pol4'](results[comp]['centers']),
                    color=colors[comp],  linestyle='dashed', label=f"fit. {comp + 1}")
        plt.title(f"{np.asarray(res_df.st)}")
        plt.xlabel('Density')
        plt.ylabel('Free Energy')
        plt.legend()

        plt.figure(figsize=(12,3))

        plt.subplot(131)
        colors = ['b', 'r']
        for comp in np.arange(0, nc):
            x = results[comp]['centers']
            y = results[comp]['beta']
            plt.plot(x, y, color=colors[comp], label=f"Beta c. {comp + 1}")
            y = results[comp]['gauss']
            plt.plot(x, y, color=colors[comp], label=f"Gauss c. {comp + 1}",  linestyle='dashed')
        plt.xlabel('Density')
        plt.ylabel('Probability')
        plt.legend()
        plt.title(state)

        plt.subplot(132)
        colors = ['b', 'r']
        for comp in np.arange(0, nc):
            x = results[comp]['centers']
            # y = y / (eps + results[comp]['gauss'])
            # y = y / (eps + results[comp]['beta'])
            y = results[comp]['hist_pdf'] - results[comp]['beta']
            d = np.sum(results[comp]['bin_sizes'] * np.abs(y))
            plt.plot(x, y, color=colors[comp], label=f"c. {comp + 1}, e = {100*d:.1f}%",  linestyle='dashed')
        plt.xlabel('Density')
        plt.ylabel('Probability')
        plt.legend()
        plt.title(state)

        if results[comp]['pol4'] is not None:
            plt.figure(figsize=(12,3))

            plt.subplot(131)
            colors = ['b', 'r']
            for comp in np.arange(0, nc):
                dp = results[comp]['pol4'].deriv()
                plt.plot(results[comp]['centers'], dp(results[comp]['centers']),
                    color=colors[comp],  linestyle='dashed', label=f"fit. {comp + 1}")
            plt.title(f"{np.asarray(res_df.st)}")
            plt.xlabel('Density')
            plt.ylabel('Free Energy')
            plt.legend()

            plt.subplot(132)
            colors = ['b', 'r']
            for comp in np.arange(0, nc):
                dp = results[comp]['pol4'].deriv(2)
                plt.plot(results[comp]['centers'], dp(results[comp]['centers']),
                    color=colors[comp],  linestyle='dashed', label=f"fit. {comp + 1}")
            plt.title(f"{np.asarray(res_df.st)}")
            plt.xlabel('Density')
            plt.ylabel('Free Energy')
            plt.legend()

    return st, results
