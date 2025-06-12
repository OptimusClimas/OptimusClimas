# module for complex displaying/plotting

# import of needed libraries
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import cv2
import datetime


def implot(data, figxsize=12, figysize=18):
    # data: data to be plotten, np.array, 2 dimensional
    # figxsize: size of the figure in x direction, int
    # figysize: size of the figure in y direction, int

    # plot an image/matrix using imshow
    fig = plt.figure(figsize=(figxsize, figysize))
    size = 0.5
    alignement = 0.1
    # Patchwork
    ax_patchwork = fig.add_axes([alignement + 1.8 * size, 0.4, size, size])
    ax_patchwork.imshow(data)


def polarplor(phi, r):
    # phi: angular coordinates, np.array, one dimensional
    # r: radius coordinates, np.array, one dimensional
    # plot polar coordinates
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(phi, r)
    ax.grid(True)
    plt.show()


def arrayplot(array):
    # array: np.array of arrays to be plotted (2-dimensional)

    # plot an array of an array along the first axes
    for i in range(len(array)):
        plt.plot(array[i])


def visualizegridtemperature(predf, first=False, last=True, i=None, min=-30, max=40, diff=False,
                             nc=True, nominmax=False, awi=False, savediff=False, fig=None):
    # predf: temperature grid data to be displayed, np.array
    # first: whether the first year is to be displayed, boolean
    # last: whether the last year is to be displayed, boolean
    # i: which year is to be displayed (start year = 0), int
    # min: overall minimum, float
    # max: overall maximum, float
    # diff: if a difference map is to be displayed, boolean
    # nc: if the grid of the CMIP6 was used, boolean
    # nominmax: whether no minima or maxima are to be taken into account, boolean
    # awi: whether the grid of the AWI was used, boolean
    # savediff: whether to save under "heatmapdiff.jpg" or "heatmap.jpg", boolean
    # fig: matplotlib figure to display into

    # displays a heatmap of the given temperature grid
    if nominmax:
        # load lon/lat accordingly to the used grid
        if not nc:
            lats = np.load('climatesimulationAI/lats.npy',
                           mmap_mode='r+')
            lons = np.load('climatesimulationAI/lons,npy.npy',
                           mmap_mode='r+')
        else:
            if awi:
                lons = np.load('lonnetcdfawi.npy',
                               mmap_mode='r+')
                lats = np.load('../latnetcdfawi.npy',
                               mmap_mode='r+')
            else:
                lons = np.load('KlimaUi/lonnetcdfnmip6new.npy',
                               mmap_mode='r+')
                lats = np.load('KlimaUi/latnetcdfnmip6new.npy',
                               mmap_mode='r+')
        # determine year to be displayed
        if first:
            t = 0
        elif last:
            t = predf[:, 0].size - 1
        else:
            t = i
        predfnew = predf
        if diff:
            # calculate difference grid
            predfnew[t] = predf[t] - predf[0]
        # reshape temperature grid to original state (depending on used grid)
        if not nc:
            lo = 240
            la = 480
            reshapedpredf = np.ones((predf[:, 0].size, lo, la))
            reshapedpredf[t] = predfnew[t].reshape(lo, la)
            temp = predf[t].reshape(lo, la)[0, 0]
            predfscaled3 = np.ones((lats[:, 0].size, lats[0, :].size))
            for j in range(lats[:, 0].size - 1):
                for k in range(lats[0, :].size):
                    if k % 3:
                        temp = reshapedpredf[t][int(j / 3), int(k / 3)]
                    predfscaled3[j, k] = temp
        else:
            if awi:
                lo = 192
                la = 384
            else:
                lo = 192
                la = 288
            predfscaled3 = predfnew[t].reshape(lo, la)
        fig = plt.figure(figsize=(16, 35))
        # build map using Basemap with mercator projection
        m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, \
                    llcrnrlon=0, urcrnrlon=360, lat_ts=20, resolution='c')
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 360., 60.))
        m.drawcoastlines()
        m.drawmapboundary()
        # transform coordinates using map
        if not nc:
            x, y = m(lons, lats)
        else:
            xv, yv = np.meshgrid(lons, lats)
            x, y = m(xv, yv)
        print('plotting')
        plt.rcParams.update({'font.size': 30})
        # fill map with temperature values/plot heatmap
        cs = m.contourf(x, y, predfscaled3, 15, cmap=plt.cm.jet, extend='both')
        cb = m.colorbar(cs)
        if savediff:
            # save plot under "heatmapdiff.jpg"
            plt.savefig('heatmapdiff.jpg', dpi=150)
            print('heatmapdiff saved')
            img = cv2.imread('heatmapdiff.jpg')
            cropped_image = img[1919:3431, 227:2400]
            cv2.imwrite("heatmapdiff.jpg", cropped_image)
            print('heatmapdiff cropped')
        else:
            # save plot under "heatmap.jpg"
            plt.savefig('heatmap.jpg', dpi=150)
            print('heatmap saved')
            img = cv2.imread('heatmap.jpg')
            cropped_image = img[1919:3431, 227:2400]
            cv2.imwrite("heatmap.jpg", cropped_image)
            print('heatmap cropped')
            print(datetime.datetime.now())
    else:
        # load lon/lat accordingly to the used grid
        if not nc:
            lats = np.load('climatesimulationAI/lats.npy',
                           mmap_mode='r+')
            lons = np.load('climatesimulationAI/lons,npy.npy',
                           mmap_mode='r+')
        else:
            if awi:
                lons = np.load('lonnetcdfawi.npy',
                               mmap_mode='r+')
                lats = np.load('../latnetcdfawi.npy',
                               mmap_mode='r+')
            else:
                lons = np.load('KlimaUi/lonnetcdfnmip6new.npy',
                               mmap_mode='r+')
                lats = np.load('KlimaUi/latnetcdfnmip6new.npy',
                               mmap_mode='r+')
        # determine year to be displayed
        if first:
            t = 0
        elif last:
            t = predf[:, 0].size - 1
        else:
            t = i
        predfnew = predf
        if diff:
            # calculate difference grid
            predfnew[t] = predf[t] - predf[0]
            min = 0
            max = 8
        # reshape temperature grid to original state (depending on used grid)
        if not nc:
            lo = 240
            la = 480
            reshapedpredf = np.ones((predf[:, 0].size, lo, la))
            reshapedpredf[t] = predfnew[t].reshape(lo, la)
            temp = predf[t].reshape(lo, la)[0, 0]
            predfscaled3 = np.ones((lats[:, 0].size, lats[0, :].size))
            for j in range(lats[:, 0].size - 1):
                for k in range(lats[0, :].size):
                    if k % 3 == True:
                        temp = reshapedpredf[t][int(j / 3), int(k / 3)]
                    predfscaled3[j, k] = temp
        else:
            if awi:
                lo = 192
                la = 384
            else:
                lo = 192
                la = 288
            predfscaled3 = predfnew[t].reshape(lo, la)
        fig = plt.figure(figsize=(16, 35))
        # build map using Basemap with mercator projection
        m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, \
                    llcrnrlon=0, urcrnrlon=360, lat_ts=20, resolution='c')
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 360., 60.))
        m.drawcoastlines()
        m.drawmapboundary()
        if nc == False:
            x, y = m(lons, lats)
        else:
            xv, yv = np.meshgrid(lons, lats)
            x, y = m(xv, yv)
        print('plotting')
        plt.rcParams.update({'font.size': 30})
        # fill map with temperature values/plot heatmap
        cs = m.contourf(x, y, predfscaled3, 50, cmap=plt.cm.jet, extend='both', vmin=min, vmax=max)
        cb = m.colorbar(cs)
        # modify map accordingly to minimum and maximum given
        plt.clim(min, max)
        if savediff:
            # save plot under "heatmapdiff.jpg"
            plt.savefig('heatmapdiff.jpg', dpi=150)
            print('heatmapdiff saved')
            img = cv2.imread('heatmapdiff.jpg')
            cropped_image = img[1919:3431, 227:2400]
            cv2.imwrite("heatmapdiff.jpg", cropped_image)
            print('heatmapdiff cropped')
        else:
            # save plot under "heatmap.jpg"
            plt.savefig('heatmap.jpg', dpi=150)
            print('heatmap saved')
            img = cv2.imread('heatmap.jpg')
            cropped_image = img[1919:3431, 227:2400]
            cv2.imwrite("heatmap.jpg", cropped_image)
            print('heatmap cropped')
            print(datetime.datetime.now())
