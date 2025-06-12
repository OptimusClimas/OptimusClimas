import numpy as np
import pandas as pd
import tensorflow as tf
import math



def my_norm(a):
    ratio = 2 / (np.max(a) - np.min(a))
    #as you want your data to be between -1 and 1, everything should be scaled to 2,
    #if your desired min and max are other values, replace 2 with your_max - your_min
    shift = (np.max(a) + np.min(a)) / 2
    #now you need to shift the center to the middle, this is not the average of the values.
    return (a - shift) * ratio



def generateframes(length, series):
    frames = np.ones((int(series.size / length), length))
    j = 0

    for i in range((int(series.size / length))):
        frames[i] = series[j:j + length]
        if i < 19:
            j = j + length
        else:
            pass

    return frames


def generateiterateframes(serieswalks, length, amountofwalks):
    lenseries = serieswalks[0, :]
    framess = np.ones((amountofwalks + 1, (int(lenseries.size / length)), length))
    for i in range(amountofwalks + 1):
        framess[i] = generateframes(length, serieswalks[i, :])
        if i == amountofwalks:
            pass

    return framess


def activationforgaf(X, actfactor=0.7):
    # preactivation in preparation for the GAF transformation by applying the Sigmoid-Function stretched by a factor
    # X: data to be preactivated, actfactor: factor the Sigmoid-Function is stretched by, usually 0.7
    # only takes 1D arrays!
    return (np.array((tf.keras.activations.sigmoid(tf.constant([(X * actfactor)], dtype=tf.float32)) * 2) - 1))[0, :]


def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))


def cos_sum(a, b):
    """To work with tabulate."""
    return (math.cos(a + b))


def cos_diff(a, b):
    """To work with tabulate."""
    return (math.cos(a - b))


class GAF:

    def __init__(self):
        pass

    def transform(self, serie, arccoscheck=True):
        newserie = np.ones(serie.shape)
        if arccoscheck == True:
            for i in range(serie.size):
                if serie[i] == 1.:
                    newserie[i] = serie[i] - 0.0000000001
                else:
                    newserie[i] = serie[i] + 0.0000000001
                if serie[i] == -1:
                    newserie[i] = serie[i] + 0.0000000001
                else:
                    newserie[i] = serie[i] + 0.0000000001
            # Polar encoding
            phi = np.arccos(newserie)
            for i in range(phi.size):
                if np.isnan(phi[i]) == True:
                    phi[i] = np.arccos(serie[i] - 0.0000000001)

            #r = np.linspace(0, 1, len(newserie))

            # GAF Computation (every term of the matrix)
            gadf = tabulate(phi, phi, cos_diff)  # alternative: cos_sum
            gasf = tabulate(phi, phi, cos_sum)
            gaf = np.ones(gadf.shape)
            for i in range(gadf[:, 0].size):
                for j in range(gadf[0, :].size):
                    if i < j:
                        gaf[i, j] = gasf[i, j]
                    else:
                        gaf[i, j] = gadf[i, j]
            if np.any(np.isnan(gaf)) == True:
                print('gaf: ' + str(gaf) + ' newserie: ' + str(newserie) + ' phi: ' + str(phi))

        else:
            # Polar encoding
            phi = np.arccos(serie)
            # GAF Computation (every term of the matrix)
            gadf = tabulate(phi, phi, cos_diff)  # alternative: cos_sum
            gasf = tabulate(phi, phi, cos_sum)
            gaf = np.ones(gadf.shape)
            for i in range(gadf[:, 0].size):
                for j in range(gadf[0, :].size):
                    if i < j:
                        gaf[i, j] = gasf[i, j]
                    else:
                        gaf[i, j] = gadf[i, j]
            if np.any(np.isnan(gaf)) == True:
                print('gaf: ' + str(gaf) + ' newserie: ' + str(newserie) + ' phi: ' + str(phi))

        return (gaf)

    def rescale_gaf(self, framesallgaf, gafsizeorigin, lentrain, gafsize):
        framesall_rescaledTo1003 = np.ones(
            (lentrain, framesallgaf[0, :, 0, 0, 0].size, framesallgaf[0, 0, :, 0, 0].size, gafsizeorigin, gafsize))
        a = 0
        b = 0

        for z in range(lentrain):
            for a in range(framesallgaf[0, :, 0, 0, 0].size):
                for b in range(framesallgaf[0, 0, :, 0, 0].size):
                    for i in range(gafsizeorigin):
                        q = gafsize / gafsizeorigin
                        k = 0
                        for m in range(gafsize):
                            if m < q:
                                framesall_rescaledTo1003[z, a, b, i, m] = framesallgaf[z, a, b, i, k]
                            elif m == q:
                                q = q + (gafsize / gafsizeorigin)
                                k = k + 1
                                framesall_rescaledTo1003[z, a, b, i, m] = framesallgaf[z, a, b, i, k]

        framesall_rescaledTo100_2 = np.ones(
            (lentrain, framesallgaf[0, :, 0, 0, 0].size, framesallgaf[0, 0, :, 0, 0].size, gafsize, gafsize))

        x = 0

        for z in range(lentrain):
            for a in range(framesallgaf[0, :, 0, 0, 0].size):
                for b in range(framesallgaf[0, 0, :, 0, 0].size):
                    for i in range(gafsize):
                        for m in range(gafsize):
                            if m % (gafsize / gafsizeorigin) == 0:
                                framesall_rescaledTo100_2[z, a, b, i, m] = framesall_rescaledTo1003[
                                    z, a, b, int(i / (gafsize / gafsizeorigin)), m]
                                x = int(i / (gafsize / gafsizeorigin))
                            else:
                                framesall_rescaledTo100_2[z, a, b, i, m] = framesall_rescaledTo1003[z, a, b, x, m]
        return framesall_rescaledTo100_2

    def normalizerecaling(self, data, datas, interval, gafsize):
        data2 = np.ones((datas, interval, gafsize, gafsize))
        thismax = np.zeros((datas, interval))
        thismin = np.ones((datas, interval))
        for i in range(datas):
            for j in range(interval):
                for m in range(gafsize):
                    newmax = max(data[i, j, m])
                    newmin = min(data[i, j, m])
                    if newmax > thismax[i, j]:
                        thismax[i, j] = newmax
                    if newmin < thismin[i, j]:
                        thismin[i, j] = newmin

        for i in range(datas):
            for j in range(interval):
                for m in range(gafsize):
                    for k in range(gafsize):
                        data2[i, j, m, k] = (data[i, j, m, k] - thismin[i, j]) / (thismax[i, j] - thismin[i, j])

        if np.any(np.isnan(data2)) == True:
            print('data: ' + str(data2) + ' max: ' + str(thismax) + ' min: ' + str(thismin))

        return data2

    def generate_blockmatrices(self, framesallgaf, gafsize, size1, size2):
        allmatrix = np.ones((size1 * size2, gafsize, gafsize * len(framesallgaf)))
        x = 0

        for i in range(size1):
            for j in range(size2):
                allmatrix[x] = np.hstack(framesallgaf[:, i, j])
                x = x + 1

        return allmatrix


def producetarget6withoutwalks(normdata):
    target6 = np.ones(int(normdata.size / 5))
    v = 0
    w = 0

    for i in range(int(normdata.size / 5)):
        v = i * 5
        try:
            target6[w] = normdata[v + 1]
        except:
            try:
                target6[w] = normdata[v - 1]
            except Exception as e:
                print('fail')
                print(e)
        w = w + 1
    return target6

