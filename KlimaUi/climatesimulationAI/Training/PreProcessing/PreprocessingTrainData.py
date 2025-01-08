import numpy as np
import pandas as pd
import tensorflow as tf
import math


def extract_sealevel(rawdata):
    sea_level = []
    for i in range(0, 134):
        sea_level.append(rawdata.GMSL[i])
    return sea_level


def extract_co2(rawdata):
    co2_years_1880_2014 = []
    for i in range(24805, 24939):
        co2_years_1880_2014.append(rawdata.to_numpy()[i][3])
    return co2_years_1880_2014


def extract_ch4(earlydata, latedata):
    ch4early = np.ones(13)
    for i in range(1, 14):
        ch4early[i - 1] = sum(earlydata.to_numpy().transpose()[:][i])
    ch4late = np.ones(44)
    for i in range(3, 47):
        ch4late[i - 3] = sum(latedata.to_numpy().transpose()[:][i])
    xvals = np.linspace(0, 1, np.arange(1850, 1970).size)
    x = np.linspace(0, 1, ch4early.size)
    intpch4early = np.interp(xvals, x, ch4early)

    ch4 = np.ones(134)
    for i in range(30, 120):
        ch4[i - 30] = intpch4early[i]
    for i in range(44):
        ch4[i + 90] = ch4late[i]
    return ch4


def extract_pop(rawdata):
    population_years_1880_2014 = []
    for i in range(54739, 54873):
        population_years_1880_2014.append(rawdata['Population (historical estimates)'][i])
    return population_years_1880_2014


def extract_ghg(rawdata):
    ghg_world = rawdata[rawdata["Country"] == "World"]
    ghg_world = ghg_world.drop(columns="Country")
    drop_years = 1751
    for i in range(130):
        ghg_world = ghg_world.drop(columns=str(drop_years))
        drop_years += 1
    drop_years = 2015
    for i in range(3):
        ghg_world = ghg_world.drop(columns=str(drop_years))
        drop_years += 1
    np_ghg_world = []
    for i in range(134):
        np_ghg_world.append(ghg_world.to_numpy()[0][i])
    return np_ghg_world


def extract_temp(rawdata):
    temp = rawdata['Temperature']
    temperature_1880_2014 = []

    for i in range(0, 134):
        temp = (rawdata['Temperature'][i]) + 13.97
        temperature_1880_2014.append(temp)
    return np.array(temperature_1880_2014)


def extract_otherghg(rawdata, ghgsize):
    ghg_1880_2014 = np.ones(ghgsize)
    k = 0

    for i in range(1880, 2014):
        ghg_1880_2014[k] = sum(rawdata['X' + str(i)])
        k = k + 1
    return ghg_1880_2014


def extract_grib(ecmwf=True):
    tempgrib_mean03 = np.load('../climatesimulationAI/Training/PreProcessing/targetdata/tempgrib/tempgrib_mean03.npy',
                              mmap_mode='r+')
    tempgrib = np.load(
        '../climatesimulationAI/Training/PreProcessing/targetdata/tempgrib/temperature_grb_1940_present.npy',
        mmap_mode='r+')
    div = 3

    tempgrib_mean = np.ones(
        (tempgrib[:, 0, 0].size, int((tempgrib[0, :, 0].size - 1) / div), tempgrib_mean03[0, 0, :].size))
    for i in range(tempgrib[:, 0, 0].size):
        for j in range(int((tempgrib[0, :, 0].size - 1) / div)):
            for k in range(tempgrib_mean03[0, 0, :].size):
                tempmean = np.ones(div)
                for m in range(div):
                    tempmean[m] = tempgrib[i, j * m, k]
                tempgrib_mean[i, j, k] = np.mean(tempmean)
    train_data = np.ones((tempgrib_mean[:, 0, 0].size, tempgrib_mean[0, :, 0].size * tempgrib_mean[0, 0, :].size))
    org_ind_tempgrib = np.ones(
        (tempgrib_mean[:, 0, 0].size, tempgrib_mean[0, :, 0].size * tempgrib_mean[0, 0, :].size, 3))
    for i in range(tempgrib_mean[:, 0, 0].size):
        x = 0
        for j in range(tempgrib_mean[0, :, 0].size):
            for k in range(tempgrib_mean[0, 0, :840].size):
                train_data[i, x] = tempgrib_mean[i, j, k]
                org_ind_tempgrib[i, x][0] = i
                org_ind_tempgrib[i, x][1] = j
                org_ind_tempgrib[i, x][2] = k
                x = x + 1
    train_data = train_data.transpose()[:, :74]
    return train_data


def importdata():
    # useddata: data used until int, 1: GHG, 2: CO2, 3: CH4, 4: BC, 5: SO2, 6: OC, 7: Temperature, 8: Sea Level , 9: Population
    try:
        paths = ["../PreProcessing/Data/SeaLevelData/gmsl2.csv", '../PreProcessing/Data/GHGData/owid-co2-data.csv',
                 '../PreProcessing/Data/GHGData/CH4_Extension_CEDS_emissions_by_sector_v2017_05_18.csv',
                 '../PreProcessing/Data/GHGData/CH4_CEDS_emissions_by_sector_v2017_05_18.csv',
                 '../PreProcessing/Data/other/world-population-by-world-regions-post-1820.csv',
                 '../PreProcessing/Data/GHGData/emission data.csv', '../PreProcessing/Data/other/temperature_NOAA2.csv',
                 '../PreProcessing/Data/GHGData/BC_CEDS_emissions_by_sector_v2016_07_26.csv',
                 '../PreProcessing/Data/GHGData/SO2_CEDS_emissions_by_sector_v2016_07_26.csv',
                 '../PreProcessing/Data/GHGData/OC_CEDS_emissions_by_sector_v2016_07_26.csv']
        raw_data = []
        for i in range(len(paths)):
            raw_data.append(pd.read_csv(paths[i]))
    except FileNotFoundError:
        paths = ["../climatesimulationAI/Training/PreProcessing/Data/SeaLevelData/gmsl2.csv",
                 '../climatesimulationAI/Training/PreProcessing/Data/GHGData/owid-co2-data.csv',
                 '../climatesimulationAI/Training/PreProcessing/Data/GHGData/CH4_Extension_CEDS_emissions_by_sector_v2017_05_18.csv',
                 '../climatesimulationAI/Training/PreProcessing/Data/GHGData/CH4_CEDS_emissions_by_sector_v2017_05_18.csv',
                 '../climatesimulationAI/Training/PreProcessing/Data/other/world-population-by-world-regions-post-1820.csv',
                 '../climatesimulationAI/Training/PreProcessing/Data/GHGData/emission data.csv',
                 '../climatesimulationAI/Training/PreProcessing/Data/other/temperature_NOAA2.csv',
                 '../climatesimulationAI/Training/PreProcessing/Data/GHGData/BC_CEDS_emissions_by_sector_v2016_07_26.csv',
                 '../climatesimulationAI/Training/PreProcessing/Data/GHGData/SO2_CEDS_emissions_by_sector_v2016_07_26.csv',
                 '../climatesimulationAI/Training/PreProcessing/Data/GHGData/OC_CEDS_emissions_by_sector_v2016_07_26.csv']
        raw_data = []
        for i in range(len(paths)):
            raw_data.append(pd.read_csv(paths[i]))
    extracted_data = []
    extracted_data.append(extract_ghg(raw_data[5]))
    extracted_data.append(extract_co2(raw_data[1]))
    extracted_data.append(extract_ch4(raw_data[2], raw_data[3]))
    for i in range(3):
        extracted_data.append(extract_otherghg(raw_data[7 + i], len(extracted_data[2])))
    extracted_data.append(extract_temp(raw_data[6]))
    extracted_data.append(extract_sealevel(raw_data[0]))
    extracted_data.append(extract_pop(raw_data[4]))

    del raw_data
    #del extracted_data[useddata:]
    return (np.array(extracted_data))


def approxdata(startwert, years, endwert):
    approxdatas = np.ones(years)
    c = endwert / startwert
    a = float(c) ** (1 / years)
    for i in range(years):
        approxdatas[i] = startwert * a ** i

    return approxdatas


def approxdataall(data, intervall):
    approxdates = np.ones(data.shape)
    approxdate = np.ones((int((data.size) / intervall), intervall))
    h = 0
    k = intervall
    for i in range(int((data.size) / intervall)):
        try:
            approxdate[i] = approxdata(data[h], intervall, data[k])
            for l in range(10):
                if np.isnan(approxdate[i, l]) == True:
                    print('try, i: ' + str(i) + ' l: ' + str(l) + ' h: ' + str(h) + ' k: ' + str(k))
        except:
            temp = approxdata(data[h], intervall - 1, data[k - 1])
            for q in range(intervall - 1):
                approxdate[i, q] = temp[q]
            approxdate[i, intervall - 1] = data[k - 1]
            for l in range(intervall):
                if np.isnan(approxdate[i, l]) == True:
                    print('ex, i: ' + str(i) + ' l: ' + str(l))
        h = h + intervall
        k = h + intervall

    m = 0
    for i in range(int((data.size) / intervall)):
        for k in range(intervall):
            approxdates[m] = approxdate[i, k]
            m = m + 1
    try:
        approxdates[int((data.size) / intervall) * intervall:] = approxdata(
            data[int((data.size) / intervall) * intervall], data[int((data.size) / intervall) * intervall:].size,
            data[data.size - 1])
    except:
        pass  #print('failed line 213')
    return approxdates


def my_norm(a):
    ratio = 2 / (np.max(a) - np.min(a))
    #as you want your data to be between -1 and 1, everything should be scaled to 2,
    #if your desired min and max are other values, replace 2 with your_max - your_min
    shift = (np.max(a) + np.min(a)) / 2
    #now you need to shift the center to the middle, this is not the average of the values.
    return (a - shift) * ratio


def generateWalks(walklength, steplength, AmountOfWalks, series):
    walks = np.ones((AmountOfWalks + 1, walklength))
    for i in range(AmountOfWalks + 1):
        try:
            j = i * steplength
            select = series[j:j + walklength]
            walks[i] = select[:, 0]
        except:
            j = i * steplength
            select = series[j:j + walklength]
            walks[i] = select[:]

    return walks


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


def generate_target6(normdata, walklen):
    walkstarget = generateWalks(walklen, 1, normdata.size - walklen, normdata)

    target6 = np.ones((normdata.size - walklen + 1) * 15 - 5)
    v = 0
    w = 0

    for c in range(normdata.size - walklen + 1):
        for i in range(1, 15):
            v = i * 5
            try:
                target6[w] = walkstarget[c, v + 1]
            except:
                try:
                    target6[w] = walkstarget[c, v - 1]
                except:
                    print('fail')
            w = w + 1
    return target6


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


def preprocessing(useddata=7, gafsize=20, walklen=70, actfactor=0.8, approxinterval=20, numtargets=2,
                  timestart=1940, timeend=2014, ssp1=False, ssp3=False, ssp5=False, altdataused=False, altdata=None,
                  tempgrib=False, endaftertargetcalc=False, gafwisenorm=False, withoutwalks=False):
    # useddata: data used until int, 0: GHG, 1: CO2, 2: CH4, 3: BC, 4: SO2, 5: OC, 6: Temperature, 7: Sea Level , 8: Population
    # gafsize: int, x length of a single gaf matrix, int*int = shape of a single gaf matrix
    # walklen: int, lenght of a single walk
    # actfactor: float, factor multiplied with the normalised data before activation with sigmoid before the gaf transformation
    # approxinterval: int, interval over which data is averaged
    # numtargets: int, number of needed targets, 1: temperature, 2: sea level
    # timestart: int, startyear for using observational data
    # timeend: int, endyear for using observational data
    # ssp1: boolean, if True: uses simulated data for scenario ssp1 until 2100
    # ssp3: boolean, if True: uses simulated data for scenario ssp3 until 2100
    # ssp5: boolean, if True: uses simulated data for scenario ssp5 until 2100
    # altdataused: boolean, if True: don't use standard data but given data in altdata
    # altdata: numpy array, alternative data to be processed IF otherdatauses IS ACTIVATED !, needed shape: (spatial dimension/different features,temporal dimension)
    # tempgrib: boolean, if True: use the temperature grib data from the ECMWF if altdataused is activated
    # endaftertargetcalc: if True: only calculates the target6 and stops afterwards
    if altdataused == False:
        train_data = importdata()[:, timestart - 1880:134 - (2014 - timeend)]
    else:
        if tempgrib == False:
            if (altdata == None).all():
                raise Exception(
                    "There's no alternative data given in altdata to be processed! Did you accidentally activated alternative data to be used?")
            else:
                train_data = altdata
        elif tempgrib == True:
            train_data = extract_grib()
        else:
            train_data = altdata

    min_train = np.ones(len(train_data))
    max_train = np.ones(len(train_data))

    for i in range(len(train_data)):
        min_train[i] = min(train_data[i])
        max_train[i] = max(train_data[i])
    try:
        np.save('../PreProcessing/normdata/min_train.npy', min_train)
        np.save('../PreProcessing/normdata/max_train.npy', max_train)
    except FileNotFoundError:
        np.save('../climatesimulationAI/Training/PreProcessing/normdata/min_train.npy', min_train)
        np.save('../climatesimulationAI/Training/PreProcessing/normdata/max_train.npy', max_train)

    if altdataused == False:
        positive_seaLevel = np.ones(np.array(train_data[6]).size)
        for i in range(np.array(train_data[6]).size):
            positive_seaLevel[i] = np.array(train_data[6])[i] - min(np.array(train_data[6])) + 0.0000000000001

    train_data_approx = np.ones(train_data.shape)
    for i in range(len(train_data_approx)):
        train_data_approx[i] = approxdataall(train_data[i], approxinterval)
    if altdataused == False:
        if len(train_data_approx) < 9 and len(train_data_approx) > 7:
            train_data_approx[8] = approxdataall(positive_seaLevel[60:134], approxinterval)

    train_data_norm_for_gaf = np.ones(train_data.shape)
    for i in range(len(train_data_approx)):
        train_data_norm_for_gaf[i] = my_norm(train_data_approx[i])

    for j in range(len(train_data_approx)):
        for i in range(train_data[0, :].size):
            if train_data_norm_for_gaf[j, i] < -1:
                train_data_norm_for_gaf[j, i] = train_data_norm_for_gaf[j, i] + (train_data_norm_for_gaf[j, i] + 1)
            if train_data_norm_for_gaf[j, i] > 1:
                train_data_norm_for_gaf[j, i] = train_data_norm_for_gaf[j, i] - (train_data_norm_for_gaf[j, i] - 1)
    if altdataused == False:
        targets6 = np.ones((numtargets, (train_data[0, :].size - walklen + 1) * 15 - 5))
        for i in range(numtargets):
            if tempgrib == False:
                targets6[i] = generate_target6(train_data_norm_for_gaf[7 + i], walklen)
            else:
                targets6[i] = generate_target6(train_data_norm_for_gaf[i], walklen)
    else:
        if withoutwalks == True:
            targets6 = producetarget6withoutwalks(train_data_norm_for_gaf[0, :])
        else:
            targets6 = np.ones((numtargets, (train_data_norm_for_gaf[0, :].size - walklen + 1) * 15 - 5))
            for i in range(numtargets):
                if tempgrib == False:
                    targets6[i] = generate_target6(train_data_norm_for_gaf[i], walklen)

    train_data_norm_for_gaf = train_data_norm_for_gaf[:useddata, :]
    if endaftertargetcalc == False:

        framesall5 = np.ones(
            (len(train_data_norm_for_gaf), train_data[0, :].size - walklen + 1, int(walklen / 5), 5))
        for i in range(len(train_data_norm_for_gaf)):
            framesall5[i] = generateiterateframes(
                generateWalks(walklen, 1, train_data[0, :].size - walklen, train_data_norm_for_gaf[i]), 5,
                train_data[0, :].size - walklen)

        framesall10 = np.ones(
            (len(train_data_norm_for_gaf), train_data[0, :].size - walklen + 1, int(walklen / 10), 10))
        for i in range(len(train_data_norm_for_gaf)):
            framesall10[i] = generateiterateframes(
                generateWalks(walklen, 1, train_data[0, :].size - walklen, train_data_norm_for_gaf[i]), 10,
                train_data[0, :].size - walklen)

        framesall5_act = activationforgaf(actfactor, framesall5)
        framesall10_act = activationforgaf(actfactor, framesall10)

        gaf = GAF

        framesall5_gaf = np.ones(
            (len(train_data_norm_for_gaf), framesall5[0, :, 0, 0].size, framesall5[0, 0, :, 0].size, 5, 5))

        for k in range(len(train_data_norm_for_gaf)):
            for i in range(framesall5[0, :, 0, 0].size):
                for j in range(framesall5[0, 0, :, 0].size):
                    framesall5_gaf[k, i, j, :, :] = gaf.transform(gaf, framesall5_act[k, i, j, :])

        framesall10_gaf = np.ones(
            (len(train_data_norm_for_gaf), framesall10[0, :, 0, 0].size, framesall10[0, 0, :, 0].size, 10, 10))

        for k in range(len(train_data_norm_for_gaf)):
            for i in range(framesall10[0, :, 0, 0].size):
                for j in range(framesall10[0, 0, :, 0].size):
                    framesall10_gaf[k, i, j, :, :] = gaf.transform(gaf, framesall10_act[k, i, j, :])

        framesall5_rescaledTo100 = gaf.rescale_gaf(gaf, framesall5_gaf, 5, len(train_data_norm_for_gaf), gafsize)
        framesall10_rescaledTo100 = gaf.rescale_gaf(gaf, framesall10_gaf, 10, len(train_data_norm_for_gaf), gafsize)

        if gafwisenorm == True:
            framesall5_rescaledTo100 = np.ones(framesall5_rescaledTo100.shape)
            for i in range(len(train_data_norm_for_gaf)):
                framesall5_rescaledTo100[i] = gaf.normalizerecaling(gaf, framesall5_rescaledTo100[i],
                                                                    framesall5[0, :, 0, 0].size,
                                                                    framesall5[0, 0, :, 0].size, gafsize)

            framesall10_rescaledTo100 = np.ones(framesall10_rescaledTo100.shape)
            for i in range(len(train_data_norm_for_gaf)):
                framesall10_rescaledTo100[i] = gaf.normalizerecaling(gaf, framesall10_rescaledTo100[i],
                                                                     framesall10[0, :, 0, 0].size,
                                                                     framesall10[0, 0, :, 0].size, gafsize)

        all5matrix = gaf.generate_blockmatrices(gaf, framesall5_rescaledTo100, gafsize, framesall5[0, :, 0, 0].size,
                                                framesall5[0, 0, :, 0].size)
        all10matrix = gaf.generate_blockmatrices(gaf, framesall10_rescaledTo100, gafsize,
                                                 framesall10[0, :, 0, 0].size, framesall10[0, 0, :, 0].size)

        return (all5matrix, all10matrix, targets6.transpose())
    else:
        return targets6.transpose()
