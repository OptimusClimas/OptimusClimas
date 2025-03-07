# main module for the simulation with Optimus Climas
# all simulation intern functions are based here

# imports of needed libraries for the simulation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy
from numba import jit
from scipy.ndimage import gaussian_filter
import math
# import of needed functions from other modules
from KlimaUi.KlimaUi.climatesimulationAI.Training.Training import Training
from KlimaUi.KlimaUi.climatesimulationAI.Training.PreProcessing import PreprocessingTrainData

def creatediff(predfin):
    diffs = np.ones(55296)
    for i in range(55296):
        diffs[i] = predfin[99, i] - predfin[0, i]
    return diffs
def calc_future_emissions(emission, years, ghgchanges):
    # generate emission time series based on exponential development, assumed increase or decrease and current values
    # of greenhouse gas (ghg) emissions emission: index for chosen ghg, integer:
    # [0] ghg, [1] co2, [2] ch4, [3] bc, [4] so2, [5] oc
    # years: timescale to generate time
    # series for, numpy array shape (length, ) (usually generated with np.arange)
    # ghgchanges: assumed emission increases or decreases in procent
    #            (positive values = increase, negative values = decrease),
    #             list, shape (6)

    # current values of the ghgs in order of the used indices
    feature_today = [1.470000e+12, 35452.459, 369772.2615919564, 8035.338362034104, 112684.1359768876,
                     19657.40319671115, 14.637567415953079, 200.2, 7380117870.0]
    # init variables
    ghg_future = []
    procent = 0

    for i in range(years.size):
        # calculate increase from one year to another based on linear increase and the assumed scenario
        if ghgchanges[emission] > 0:
            procent = 100 + (i * (abs(ghgchanges[emission]) / years.size))
        elif ghgchanges[emission] < 0:
            procent = 100 - (i * (abs(ghgchanges[emission]) / years.size))

        ghg_future.append(feature_today[emission] * procent / 100)  # calculate emission value for the concerning year

    ghg_futuren = np.ones(years.size)

    # calculate increase from one year to another based on exponential increase and the assumed scenario
    q = (ghg_future[years.size - 1] / feature_today[emission]) ** (1 / years.size)

    for i in range(years.size):
        ghg_futuren[i] = feature_today[emission] * q ** i  # calculate emission value for the concerning year

    return ghg_futuren


def my_norm(a, kind):
    # normalising to a scale from -1 to 1
    # minima and maxima for normalising from generating training data
    # a: data to be normalised, np.array,
    # kind: index for value being normalised (needed for the corresponding minima and maxima), int
    # [0] ghg, [1] co2, [2] ch4, [3] bc, [4] so2, [5] oc
    # ONLY FOR POSITIVE INPUTS POSSIBLE!!!

    # loading minima and maxima from training data
    maxtrainReal = np.load(
        '../KlimaUI/climatesimulationAI/ForPreprocessing/max_train.npy',
        mmap_mode='r+')
    mintrainReal = np.load(
        '../KlimaUI/climatesimulationAI/ForPreprocessing/min_train.npy',
        mmap_mode='r+')

    ratio = 2 / (maxtrainReal[kind] - mintrainReal[kind])
    # as you want your data to be between -1 and 1, everything should be scaled to 2,
    # if your desired min and max are other values, replace 2 with your_max - your_min
    shift = (maxtrainReal[kind] + mintrainReal[kind]) / 2
    # now you need to shift the center to the middle, this is not the average of the values.
    return (a - shift) * ratio


def denorm(a, startvalue, sealevel, nccmip6, new=False, fred93=False, newsea=True):
    # denormalisation of model results
    # a: data to be denormalised, np.array
    # startvalue: value of the denormalised time series at time step 0, float
    # sealevel: if input data a is sea level data, boolean
    # nccmip6: if grid of cmip6 imported in a netCDF format is used, boolean
    # new: if another factor is needed for a model (e.g. FrederikeSSTGADFGRIBhist101), boolean (optional)
    # fred93: if the model FrederikeSSTGADFGRIBwithssp2_93 is used, boolean (optional)
    # newsea: if a sea level model with regionalised temperature input is used (e.g. kalaSST104), boolean (optional)

    # defining the different factors needed for different models
    if not nccmip6:
        factor = 13500
    else:
        if fred93:
            factor = 30000
        else:
            factor = 500000
    if new:
        factor = 800000
    if startvalue == 14.64:
        factor = 13500

    # shifting the input data to positive values (reversing the shift in the normalisation)
    positiveshift = a + 1
    # scaling the shifted data with the needed factor for denormalisation
    denormalized_temp = positiveshift * factor
    if newsea:
        denormalized_sealevel = positiveshift * (2200000 * 2.67 * 3.7 * 65)
    else:
        denormalized_sealevel = positiveshift * (2200000 * 2.67 * 3.7 * 3)
    # shifting the scaled data by the needed calculated rate
    if sealevel:
        denormrate = startvalue - denormalized_sealevel[0]
        denormalized = denormalized_sealevel + denormrate
    else:
        denormrate = startvalue - denormalized_temp[0]
        denormalized = denormalized_temp + denormrate
    # additional shift if needed (based on model and predicted value)
    if not nccmip6:
        denormfinal = denormalized
    elif startvalue != 14.64:
        if not sealevel:
            if new:
                denormfinal = denormalized
            else:
                denormfinal = denormalized + 273.15
        else:
            denormfinal = denormalized
    elif startvalue == 14.64:
        denormfinal = denormalized

    return denormfinal


def postprocessing(pred_data, years, i, sealevel=False, withcmip6=False, new=False, awi=False, newsea=True):
    # processing of the raw model output data including denormalisiation
    # and interpolating predictions for each year (instead of 5 year intervals)
    # pred_data: data to be processed, np.array, shape depending on the used model
    # years: time span to be processed, np.array, 1 D Array, usually generated with np.arange
    # i:
    # sealevel: if sea level predictions are to processed, boolean (optional)
    # withcmip6: if grid of cmip6 imported in a netCDF format is used, boolean (optional)
    # new: if another factor is needed for another model (which model ?!?), boolean (optional)
    # awi, if the grid of the model from the AWI is used (optional)
    # newsea: if the new sea level model (which one ?!?) is used, boolean (optional)

    # defining the start value based on chosen model and parameter
    if not withcmip6:
        if i == 115200 or i == 115201:
            startvalue = 14.64  # start value for global mean temperature
        else:
            # start values for grid points imported from training data
            startvalue = np.load(
                '../KlimaUi/climatesimulationAI/Training/PreProcessing/trainingdata/train_data_1880_2014.npy',
                mmap_mode='r+')[
                             i, 83] - 273.15
    else:
        if awi:
            u = 73728
        else:
            u = 55296
        if i == u or i == (u + 1):
            startvalue = 14.64  # start value for global mean temperature
        else:
            if awi:
                startvalue = np.load(
                    'climatesimulationAI/Training/PreProcessing/trainingdata/train_data_2014_2100_awi.npy',
                    mmap_mode='r+')[
                                 i, 83] - 273.15
            else:
                startvalue = np.load(
                    'climatesimulationAI/Training/PreProcessing/trainingdata/train_data_2014_2100_ssp2.npy',
                    mmap_mode='r+')[i, 83] - 273.15
    if sealevel:
        startvalue = 210.9  # start value for global mean sea level

    # interpolating data because only every 6 years' value gets predicted
    xvals = np.linspace(0, 1, years.size)
    x = np.linspace(0, 1, pred_data[:].size)
    intpData = np.interp(xvals, x, pred_data[:])  # interpolating over the 5 year intervals
    # denormalisation
    output = denorm(intpData, startvalue, sealevel, nccmip6=withcmip6, new=new, newsea=newsea)

    return output


def preprocessing(future_inputa, years, gafsize, numberoffeatures, actfactora, s=None):
    # processing future emission time series to final gaf matrices (final input for the neural network)
    # future_inputa: future emission time series, np.array, shape (6, 100)
    # years: time span of the emission time series, np.array 1d, usually generated with np.arange
    # gafsize: squared is the size of the final gaf matrices, int
    # numberoffeatures: amount of features in the inputs, int
    # actfactora: factor the sigmoid is stretched by in the preactivation, float
    # s: int, for what? (optional)

    # init array for 5 year frames
    framesall5 = np.ones((future_inputa[:, 0].size, int(years.size / 5), 5))

    # generate frames already normalized and preactivated
    for i in range(future_inputa[:, 0].size):
        if s is not None:
            framesall5[i] = PreprocessingTrainData.generateframes(5,
                                                                  PreprocessingTrainData.activationforgaf(
                                                                      my_norm(future_inputa[i], s), actfactora))
        else:
            framesall5[i] = PreprocessingTrainData.generateframes(5,
                                                                  PreprocessingTrainData.activationforgaf(
                                                                      my_norm(future_inputa[i], i), actfactora))

    gaf = PreprocessingTrainData.GAF  # init GAF class for GAF transformation

    # init array for gaf matrices
    framesall5_gaf = np.ones((future_inputa[:, 0].size, int(years.size / 5), 5, 5))

    # transform generated frames to gaf matrices
    for i in range(future_inputa[:, 0].size):
        for j in range(framesall5[i, :, 0].size):
            framesall5_gaf[i, j, :, :] = gaf.transform(gaf, framesall5[i, j, :], True)

    # rescaling gaf matrices to gafsize x gafsize
    framesall5_rescaled = gaf.normalizerecaling(gaf, gaf.rescale_gaf(gaf, framesall5_gaf.reshape(
        (1, future_inputa[:, 0].size, int(years.size / 5), 5, 5)), 5, 1, gafsize)[0, :, :, :], framesall5[:, 0, 0].size,
                                                framesall5[0, :, 0].size, gafsize)

    # combine all matrices to a block matrix
    all5matrixtemp = generate_blockmatrices(gaf, framesall5_rescaled, gafsize, framesall5[0, :, 0].size)

    # reshape for final input for the neural network
    X = all5matrixtemp
    X = X.reshape(int(years.size / 5), gafsize, gafsize * numberoffeatures, 1)

    return X


def generate_blockmatrices(self, framesallgaf, gafsize, size1):
    allmatrix = np.ones((size1, gafsize, gafsize * len(framesallgaf)))
    x = 0

    for i in range(size1):
        allmatrix[x] = np.hstack(framesallgaf[:, i])
        x = x + 1

    return allmatrix


@jit
def movingaverage_new(y, N=15, only_historical=False, m=2, fut=False):
    # Applies the moving average filter to the input data
    # y: input data, np.array
    # N: number of points used to calculate the moving average, int (optional)
    # only_historical: if True, use only historical data (before the current point) for the average, boolean (optional)
    # m: length of the interval used for the approximating tangent for the values
    #    outside the given timeline, int (optional)
    # fut: if True, use the last values; if False, use the first values
    #      for the approximating tangent outside the timeline, boolean (optional)

    if only_historical:
        # Initialize array for averaged time series
        y_average = np.ones(len(y))

        # Calculate tangent for approximating the N values before the timeline
        if fut:
            # Calculate gradient based on the last m points
            tangente_factor = -1 * (y[len(y) - 1] - y[len(y) - m - 1]) / m
            # Calculate tangent for N points after the given timeline
            y_appendeda = tangente_factor * np.arange(len(y), len(y) + N)
        else:
            # Calculate gradient based on the first m points
            tangente_factor = -1 * (y[0] - y[m - 1]) / m
            # Calculate tangent for N points before the given timeline
            y_appendeda = tangente_factor * np.arange(-N, 0)

        # Initialize time series with appended approximated N points before the given timeline
        y_appended = np.ones(len(y) + N)
        y_appended[:N] = y_appendeda
        y_appended[N:] = y

        # Calculate moving average for each point using the appended values
        x = 0
        for i in range(N, len(y_appended)):
            y_average[x] = np.mean(y_appended[i - N:i])
            x = x + 1

        # Handle the first N points separately by averaging progressively more points
        for i in range(N):
            y_average[i] = np.mean(y[0:i + 1])

        # Adjust the result to align with the original data
        return y_average + (y[N] - y_average[N])
    else:
        # Placeholder for non-historical average implementation
        pass


def movingaverage(y, N=15):
    # Y: input data, np.array
    # N: number of points used to calculate the moving average, int
    # applies moving average also using data points after in the time series
    y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode='edge')
    return np.convolve(y_padded, np.ones((N,)) / N, mode='valid')


def calc_withrainforestemissions(future_inputs, rainforestemission_co2, temperature_pred, X, start, years, gafsize,
                                 numberoffeatures, actfactor, p_rainforest):
    # future_inputs: emissions in all used ghgs as calculated before, np.array, shape (numberoffeatures, years.size)
    # rainforestemission_co2: amount of co2 emissions released due to die-off of the amazonas rainforest, float
    # temperature_pred: predicted temperature (normalised), np.array, shape (100)
    # X: input for the neural network (combined GAF) as used before, np.array
    # start: start year of the simulation, int
    # years: years to be simulated, np.array, shape (100)
    # gafsize: size of the GAF matrices, int
    # numberoffeatures: amount of features (input parameters) to be used for the simulation, int
    # actfactor: factor used for pre-activation to multiply sigmoid
    # p_rainforest: geo-reference points for the amazon rainforest (translated into internal coordinates), np.array

    # calculate the trigger year of and influenced emissions by the tipping point
    # die-off of the amazon rainforest and the corresponding gaf matrices as final input for the model
    # to proceed the simulation
    tippingpointrainforest = 30.26  # regional temperature risen by 4 °C relative to pre-industrial level
    rainforesttriggerd = False

    # denormalise predicted temperature
    temperature_denormtemp = np.ones(temperature_pred.shape)
    for i in range(temperature_pred[0, :].size - 1):
        temperature_denormtemp[:, i] = denorm(temperature_pred[:, i],
                                              np.load('climatesimulationAI/Training/PreProcessing/trainingdata/train_data_2014_2100_ssp2.npy', mmap_mode='r+')[i, 0] - 273.15,
                                              sealevel=False,
                                              nccmip6=True,
                                              fred93=True)
    temperature_denormtemp[:, 55296] = denorm(temperature_pred[:, 55296], 14.64, sealevel=False, nccmip6=True,
                                              fred93=True)
    # apply filter to denormalised temperature (postprocessing)
    new = np.ones(temperature_denormtemp.shape)
    for i in range(temperature_denormtemp[0, :].size):
        new[:, i] = movingaverage(temperature_denormtemp[:, i], N=10)
    temperature_denormtemp = new

    # iterate over all 5 year frames
    for i in range(1, X[:, 0, 0, 0].size):
        # get temperature in reference points
        temps = np.ones(len(p_rainforest))
        for j in range(len(p_rainforest)):
            temps[j] = temperature_denormtemp[i, p_rainforest[j]]
        tempalt = temps[temps != np.min(temps)]
        temp = tempalt[tempalt != np.min(tempalt)].mean()
        # check if tipping point was triggerd
        if temp > tippingpointrainforest and rainforesttriggerd == False:
            rainforesttriggerd = True
            triggeryearrainforest = (i * 5) - 1
            print('Amazon Rainforest dieback was triggerd in year ' + str(start + triggeryearrainforest))

            #calculate emissions with the additional emissions from the tipping point
            withrainforest_future_inputs = copy.deepcopy(future_inputs)
            durationrainforest = 100
            # initial emission
            withrainforest_future_inputs[:2, triggeryearrainforest] = future_inputs[:2, triggeryearrainforest] + (
                    rainforestemission_co2 / durationrainforest)
            # emission longterm
            for j in range(years.size - triggeryearrainforest):
                if triggeryearrainforest + j < years.size:
                    withrainforest_future_inputs[:2, triggeryearrainforest + j] = future_inputs[:2,
                                                                                  triggeryearrainforest + j] + (
                                                                                          rainforestemission_co2 / durationrainforest)
            # calculate new combined GAF as input for the neural network
            Xnew = preprocessing(withrainforest_future_inputs, years, gafsize, numberoffeatures, actfactor)
    # return calculated values if the tipping point was triggerd
    if rainforesttriggerd:
        return Xnew, withrainforest_future_inputs, triggeryearrainforest


def calc_withpermafrostemissions(future_inputs, p, permafrostemission_co2, temperature_pred, X, start, years, gafsize,
                                 numberoffeatures, actfactor, anerobe=False, permafrostemission_ch4=None):
    # future_inputs: emissions in all used ghgs as calculated before, np.array, shape (numberoffeatures, years.size)
    # p: geo-reference point (tranformed into internal coordinated), int
    # permafrostemission_co2: amount of co2 emissions released due to collapse of the boreal permafrost, float
    # temperature_pred: predicted temperature (normalised), np.array, shape (100)
    # X: input for the neural network (combined GAF) as used before, np.array
    # start: start year of the simulation, int
    # years: years to be simulated, np.array, shape (100)
    # gafsize: size of the GAF matrices, int
    # numberoffeatures: amount of features (input parameters) to be used for the simulation, int
    # actfactor: factor used for pre-activation to multiply sigmoid
    # anerobe: whether anarobe conditions are to be assumed or not, boolean (optional)
    # permafrostemission_ch4: amount of ch4 emissions released due to collapse of the boreal permafrost (under anarobe conditions), float (optional)

    # calculate the trigger year of and influenced emissions by the tipping point
    # collapse of the boreal permafrost and the corresponding gaf matrices as final input for the model
    # to proceed the simulation

    # init variables
    Xnew = None
    triggeryearpermafrost = 100
    withpermafrost_future_inputs = None
    permafrosttriggerd = False
    tippingpointpermafrost = 5.0  # local temperature in Northeastern Siberia

    # denormalise predicted temperature
    temperature_denormtemp = np.ones(temperature_pred.shape)
    for i in range(temperature_pred[0, :].size - 1):
        temperature_denormtemp[:, i] = denorm(temperature_pred[:, i],
                                              np.load('climatesimulationAI/Training/PreProcessing/trainingdata/train_data_2014_2100_ssp2.npy', mmap_mode='r+')[i, 0] - 273.15,
                                              sealevel=False,
                                              nccmip6=True,
                                              fred93=True)
    temperature_denormtemp[:, 55296] = denorm(temperature_pred[:, 55296], 14.64, sealevel=False, nccmip6=True,
                                              fred93=True)
    # apply filter to denormalised temperature (postprocessing)
    new = np.ones(temperature_denormtemp.shape)
    for i in range(temperature_denormtemp[0, :].size):
        new[:, i] = movingaverage(temperature_denormtemp[:, i], N=10)
    temperature_denormtemp = new
    # iterate over all 5 year frames
    for i in range(1, X[:, 0, 0, 0].size):
        # get temperature in reference points
        temps = np.ones(len(p))
        for j in range(len(p)):
            temps[j] = temperature_denormtemp[i, p[j]]
        if len(p) == 3:
            temp = temps.mean()
        else:
            tempalt = temps[temps != np.min(temps)]
            temp = tempalt[tempalt != np.min(tempalt)].mean()
        # check if tipping point was triggerd
        if temp > tippingpointpermafrost and permafrosttriggerd == False:
            permafrosttriggerd = True
            triggeryearpermafrost = (i * 5) - 1
            print('Permafrost was triggerd in year ' + str(start + triggeryearpermafrost))
            # calculate emissions with the additional emissions from the tipping point
            withpermafrost_future_inputs = copy.deepcopy(future_inputs)
            if anerobe:  # with assumed anaerobe conditions -> 50% of Carbon reacts to CH4
                withpermafrost_future_inputs[2, triggeryearpermafrost] = future_inputs[2, triggeryearpermafrost] + (
                        permafrostemission_ch4 / 40)
                for j in range(years.size - triggeryearpermafrost):
                    if triggeryearpermafrost + j < years.size:
                        withpermafrost_future_inputs[2, triggeryearpermafrost + j] = future_inputs[
                                                                                         2, triggeryearpermafrost + j] + (
                                                                                             permafrostemission_ch4 / 40)
            # additional CO2 emissions
            withpermafrost_future_inputs[:2, triggeryearpermafrost] = future_inputs[:2, triggeryearpermafrost] + (
                    permafrostemission_co2 / 40)
            for j in range(years.size - triggeryearpermafrost):
                if triggeryearpermafrost + j < years.size:
                    withpermafrost_future_inputs[:2, triggeryearpermafrost + j] = future_inputs[:2,
                                                                                  triggeryearpermafrost + j] + (
                                                                                          permafrostemission_co2 / 40)
            # calculate new combined GAF for model input
            Xnew = preprocessing(withpermafrost_future_inputs, years, gafsize, numberoffeatures, actfactor)
    # return calculated values if tipping points was triggerd
    if permafrosttriggerd:
        return Xnew, withpermafrost_future_inputs, triggeryearpermafrost


def calc_wais_sealevel(temperature, start=2014):
    # temperature: predicted temperature (denormalized and processed), np.array
    # start: start year of the simulation, int (optional)

    # calculate the trigger year of and the additional global mean sea level rise due to
    # of the tipping point collapse of the west-antarctic ice-shield

    # init variables
    triggerd = False
    end = False

    u = temperature[0, :].size - 1  # index for global temperatures

    # iterate over the simulated time span
    for i in range(temperature[:, 0].size):
        # check if temperature (globally) crossed threshold for tipping point
        if temperature[i, u] >= (1.656 + (14.64 - 0.9)) and triggerd == False:
            triggerd = True
            triggeryear = i
            print('WAIS was triggerd in year ' + str(start + triggeryear))

        # check if temperature (globally) crossed threshold for the end of the implication of the tipping point
        if temperature[i, u] >= (1.8768 + (14.64 - 0.9)) and end == False:
            end = True
            endyear = i
            print('endyear: ' + str(start + endyear))

    # calculate the additional sea level rise if the tipping point was triggerd
    if triggerd:
        ges_sealevel = 100  # amount of additional sea level in total
        duration = endyear - triggeryear  # duration of the melting based on thresholds (calculated before)
        add_sealevel = np.zeros(len(temperature))
        print('duration: ' + str(duration))
        print('endyear: ' + str(endyear))
        # iterate for the duration of the melting
        for i in range(duration):
            # calculate additional sea level rise for each year
            add_sealevel[i + triggeryear] = i * (ges_sealevel / duration)
        # additional sea level stays the same after the duration (because it doesn't magically sink after)
        add_sealevel[triggeryear + duration:] = add_sealevel[triggeryear + duration - 1]
        # return calculated additional sea level due to the tipping point
        return add_sealevel


def transformgeocoordinates(lat, lon):
    # lat: latitude to be transformed, float
    # lon: longitude to be transformer, float

    # load lon/lat grids
    lons = np.load('lonnetcdfnmip6new.npy', mmap_mode='r+') - 180
    lats = np.load('latnetcdfnmip6new.npy', mmap_mode='r+')
    # init variables
    lat_in = None
    lon_in = None
    # lat: N -> -
    # lon: E -> -
    k = 0.7
    j = 0.5
    o = 0.1

    # transform lat
    for i in range(192):
        if abs(int(lats[i]) + lat) < j:
            lat_in = i

    # transform lon
    for i in range(288):
        if abs(lons[i] + lon) < k:
            lon_in = i

    # print if transformation failed
    if lat_in is None:
        print('failed geo coordinaten transformation lat of ' + str(lat))

    if lon_in is None:
        print('failed geo coordinaten transformation lon of ' + str(lon))

    # return lat/lon if transformation succeeded
    if lon_in is not None and lat_in is not None:
        return (288 * lat_in) + lon_in


def pred(ghgchanges, start=2014, end=2114, numberofghgs=6,
         numberoffeatures=7, gafsize=20, outputsize=55297, modelname="FrederikeSSTGADFGRIBhist108.h5",
         method_SST=True, method_Conv=False, onlyindata=False, onlyemidata=False, withtippingpoints=False,
         actfactor=0.8,
         predsea=False, modelnamesea='kalaSST104.h5', nccmip6=True, anaerobe=True,
         rainforestused=True, with_oldmodel=True, new=False, wais=True, awi=False, newsea=True,
         partly_anaerobe=False, partanaeorbe=None, withpermafrost=True):
    # ghgchanges: changes in ghgs (characteristic of the scenario), list of integers, shape (ghg, co2, ch4, bc,
    # so2, oc)
    # start: start year of the simulation, int
    # end: end year of the simulation, int
    # numberofghgs: amount of ghgs used for the simulation (optional)
    # numberoffeatures: amount of features (input parameters) used for the simulation (optional)
    # gafsize: size of the GAF matrices (optional)
    # outputsize: size of the output (temperature grid) (optional)
    # modelname: filename of the used neural network for the temperature simulation (optional)
    # methodSST: whether to use Spatial-Spectral Transformer/mVit, boolean (default true)
    # methodConv: whether to use CNN, boolean (default false),  EXPERIMENTAL!,
    #   ESPECIALLY WITH TIPPING POINTS! ONLY WORKS WITH MODELS TRAINED FOR CNN ARCHITECTURE!!!
    # onlyindata: whether to only calculate the gaf matrices (input for the model), boolean (default false)
    # onlyemidata: whether to only calculate the future emission developments, boolean (default false)
    # withtippingpoints: whether to consider tipping points in the simulation, boolean (default false)
    # actfactor: factor sigmoid is multiplied by in the pre-activation, float (default 0.8, determined experimentally)
    # predsea: whether to simulate global mean sea level, boolean (default false)
    # modelnamesea: filename of the used neural network for the sea level simulation (optional)
    # nccmip6: if grid of cmip6 imported in a netCDF format is used, boolean (default true),
    # anaerobe: whether anaerobe conditions for the boreal permafrost collapse are to be assumed, boolean (default true)
    # rainforestused: whether to take the tipping point die-off of the amazonas rainforest into account, boolean (default true)
    # with_oldmodel: whether to use multiple model versions (results in more accurate result), boolean (default true)
    # new: whether to use a certain model architecture (for temperature model), boolean
    # wais: whether to consider the tipping point collapse of the west-antarctic ice sheet
    # awi: if grid format of the AWI is used, boolean (default false)
    # newsea: whether to use a certain model architecture (for sea level model), boolean
    # partly_anaerobe: whether certain parts are to be assumed under anaerobe conditions
    #   for the boreal permafrost collapse, boolean (default false),  HAS EFFECT ONLY IF anaerobe=true
    # partanaeorbe: how many is to be assumed anaerobe of the boreal permafrost when collapsing,
    #   in %, float, HAS EFFECT ONLY IF partly_anaerobe=true!
    # withpermafrost: whether to take the tipping point collapse of the boreal permafrost into account, boolean (default true)


    # set outputsize accordingly to the used grid
    if not nccmip6:
        outputsize = 115201
    withcmip6 = nccmip6

    # if only emission data is to be calculated, only input needs to be calculated
    if onlyemidata:
        onlyindata = True

    # if ghg change is given as integer, use that integer for all ghgs
    if type(ghgchanges) != int:
        pass
    else:
        ghgchangese = np.ones(numberofghgs)
        for i in range(len(ghgchangese)):
            ghgchangese[i] = ghgchanges
        ghgchanges = ghgchangese
    # check if the model emission range was exceeded
    exceededmodelrange = False
    for i in range(len(ghgchanges)):
        if ghgchanges[i] > 450:
            exceededmodelrange = True
        if ghgchanges[i] < -90:
            exceededmodelrange = True
    # only continue if the model emission range was not exceeded
    if not exceededmodelrange:
        # model is unstable for emission increase from 0 to 5 % therefore all emissions originally in this
        # range are set to 5
        for i in range(len(ghgchanges)):
            if 5 > ghgchanges[i]:
                if ghgchanges[i] > -0.0001:
                    ghgchanges[i] = 5

        over200 = False
        if ghgchanges[1] > 200:
            modelnamesea = 'kalaSST100.h5'
            newsea = False
            over200 = True
        if ghgchanges[2] > 200:
            modelnamesea = 'kalaSST100.h5'
            newsea = False
            over200 = True


        years = np.arange(start, end)
        # calculate the future emission development based on the given ghg changes
        future_inputs = np.ones((numberoffeatures, years.size))
        for i in range(numberofghgs):
            future_inputs[i] = calc_future_emissions(i, years, ghgchanges)
        if numberoffeatures > numberofghgs:
            future_inputs[6] = future_inputs[1]
        # calculate combined GAF for model input
        X = preprocessing(future_inputs, years, gafsize, numberoffeatures, actfactor)
        modelnametippingpoints = 'FrederikeSSTGADFGRIBwithssp2_93.h5'
        output = None
        # start simulating if not only input data is to be calculated
        if not onlyindata:
            # build models accordingly to intended use
            if method_SST:
                # build model using SST/mViT architecture
                if new:
                    buildnew = False
                else:
                    buildnew = True
                if withtippingpoints:
                    frederike = Training.buildSST(gafsize=gafsize, outputsize=outputsize, printsum=False,
                                                  modelname=modelname, features=X, new=False)
                    frederike.load_weights('../KlimaUi/climatesimulationAI/models/' + modelnametippingpoints)
                else:
                    frederike = Training.buildSST(gafsize=gafsize, outputsize=outputsize, printsum=False,
                                                  modelname=modelname, features=X, new=buildnew)
                    frederike.load_weights('../KlimaUi/climatesimulationAI/models/' + modelname)
            elif method_Conv:
                # build model using CNN architecture
                frederike = Training.buildConv(gafsize=gafsize, inputshape=X[0].shape, printsum=False)
                if withtippingpoints:
                    frederike.load_weights('../KlimaUi/climatesimulationAI/models/' + modelnametippingpoints)
                else:
                    frederike.load_weights('../KlimaUi/climatesimulationAI/models/' + modelname)

            # init variables
            temperature_pred = np.ones((X[:, 0, 0, 0].size, outputsize))
            permafrosttriggerd = False

            # first temperature prediction
            temperature_pred = frederike.predict(X)
            oldX = copy.deepcopy(X)
            # init variable
            rainforesttriggerd = False

            # tipping point simulation
            if withtippingpoints == True and permafrosttriggerd == False:
                print('tipping point simulation starting')
                if withpermafrost:
                    # init variables
                    triggeryearspermafrost = []
                    triggeryearpermafrosttemp = 2018
                    permafrosttriggerd_parzelle = [False, False, False, False, False]

                    # lat: N -> -
                    # lon: E -> -
                    p_geo_lat = [None] * 5
                    p_geo_lon = [None] * 5
                    # init geo reference points for permafrost, see Appendix A
                    p_geo_lat[0] = np.array([-67, -68, -71, -68, -66])
                    p_geo_lon[0] = np.array([-36, -31, -54, -44, -56])
                    p_geo_lat[1] = np.array([-73, -72, -75, -76, -68, -68, -72])
                    p_geo_lon[1] = np.array([-105, -70, -92, -108, -82, -98, -90])
                    p_geo_lat[2] = np.array([-69, -70, -72, -69, -68])
                    p_geo_lon[2] = np.array([-161, -155, -126, -141, -171, -144])
                    p_geo_lat[3] = np.array([-68, -68, -69])
                    p_geo_lon[3] = np.array([147, 173, 152])
                    p_geo_lat[4] = np.array([-66, -63, -80, -76, -71, -70, -62, -72])
                    p_geo_lon[4] = np.array([52, 109, 59, 27, 82, 109, 92, 53])
                    # transform geo coordinates for permafrost into internal coordinates
                    p = [None] * len(p_geo_lat)
                    for i in range(len(p_geo_lat)):
                        p_temp = [None] * len(p_geo_lat[i])
                        for j in range(len(p_geo_lat[i])):
                            p_temp[j] = transformgeocoordinates(p_geo_lat[i][j], p_geo_lon[i][j])
                        p[i] = np.array(p_temp)

                    # different regions for permafrost collapse
                    parzellen_names = ['eurasia', 'russia west', 'russia east', 'alaska', 'north amerika']
                    # carbon (to be released) stored in the different permafrost regions
                    permafrost_carbon_stored_pg = np.array([493.9, 186.5, 186.5, 2.6, 279.2])

                    # handle edge cases (100% and 0%) of only parts of the boreal permafrost are to be assumed anaerobe
                    if partly_anaerobe:
                        if partanaeorbe == 100:
                            partly_anaerobe = False
                            anaerobe = True
                        if partanaeorbe == 0:
                            partly_anaerobe = False
                            anaerobe = False
                    # calculate co2 and ch4 emissions for the boreal permafrost collapse
                    # (if under anaerobe conditions 50% of carbon react to CH4)
                    if anaerobe:
                        permafrostemission_co2 = (((permafrost_carbon_stored_pg / 10) / 2) * 3.67) * 1000
                        permafrostemission_ch4 = (((permafrost_carbon_stored_pg / 10) / 2) * 1.34) * 10000
                    elif partly_anaerobe:
                        # for partly anerobe conditions calculate emissions according to the anaeorbe part
                        factor_anaerobe = partanaeorbe / 100
                        factor_aerobe = 1 - factor_anaerobe
                        permafrostemission_co2 = ((((
                                                            permafrost_carbon_stored_pg / 10) * factor_aerobe) * 3.67) * 1000) + (
                                                         (((permafrost_carbon_stored_pg / 10) * (
                                                                 factor_anaerobe / 2)) * 3.67) * 1000)
                        permafrostemission_ch4 = (((permafrost_carbon_stored_pg / 10) * (
                                factor_anaerobe / 2)) * 1.34) * 10000
                    else:
                        permafrostemission_co2 = ((permafrost_carbon_stored_pg / 10) * 3.67) * 1000

                if rainforestused:
                    # init geo reference points for amazonian rainforest
                    p_rainforest = np.array([(288 * 95) + 90, (288 * 93) + 98, (288 * 87) + 97,
                                             (288 * 90) + 91])

                    # carbon (to be released) stored in the amazonian rainforest
                    rainforestemission_c_pg = 30
                    # CO2 (to be released) stored in the amazonian rainforest (calculated with the stored carbon)
                    rainforestemission_co2 = (rainforestemission_c_pg * 3.67) * 1000

                    # check if die-off of the amazonian rainforest was triggerd and
                    # calculate released emissions if it was
                    rainforest = calc_withrainforestemissions(future_inputs, rainforestemission_co2, temperature_pred,
                                                              X, start, years, gafsize, numberoffeatures, actfactor,
                                                              p_rainforest)
                    if rainforest is not None:
                        # update input for the model and trigger variable if amazonian rainforest die-off was triggerd
                        rainforesttriggerd = True
                        with_oldmodel = True
                        future_inputs = rainforest[1]
                        triggerrainforest = rainforest[2]
                        # calculate new combined GAF for model input
                        Xnew = preprocessing(future_inputs, years, gafsize, numberoffeatures, actfactor)

                if withpermafrost:
                    # iterate over all regions
                    for i in range(len(p)):
                        if not permafrosttriggerd_parzelle[i]:
                            # if permafrost region wasn't triggerd already
                            # check if collapse of the boreal permafrost was triggerd and
                            # calculate released emissions if it was
                            # (while taking into account if anaerobe conditions are assumed or not)
                            if anaerobe:
                                permafrost_parzelle = calc_withpermafrostemissions(future_inputs, p[i],
                                                                                   permafrostemission_co2[i],
                                                                                   temperature_pred, X, start, years,
                                                                                   gafsize, numberoffeatures, actfactor,
                                                                                   anerobe=True, permafrostemission_ch4=
                                                                                   permafrostemission_ch4[i])
                            else:
                                permafrost_parzelle = calc_withpermafrostemissions(future_inputs, p[i],
                                                                                   permafrostemission_co2[i],
                                                                                   temperature_pred, X, start, years,
                                                                                   gafsize, numberoffeatures, actfactor)
                        else:
                            permafrost_parzelle = None

                        if permafrost_parzelle is not None:
                            # if collapse of the boreal permafrost was triggerd in the concerning region
                            # print info, update input for the model and trigger variables accordingly
                            print('Triggerd Permafrost Parzelle ' + parzellen_names[i])
                            permafrosttriggerd = True
                            with_oldmodel = True
                            permafrosttriggerd_parzelle[i] = True
                            permafrost_parzelle_saved = permafrost_parzelle
                            future_inputs = permafrost_parzelle[1]

                            if permafrost_parzelle[2] != triggeryearpermafrosttemp:
                                triggeryearpermafrosttemp = permafrost_parzelle[2]
                                triggeryearspermafrost.append(permafrost_parzelle[2])
                                break
                    try:
                        # calculate new combined GAF for model input
                        Xnew = preprocessing(future_inputs, years, gafsize, numberoffeatures,
                                             actfactor)
                        triggeryearpermafrost = permafrost_parzelle_saved[2]
                    except:
                        pass

            if permafrosttriggerd or rainforesttriggerd:
                with_oldmodel = True
            # init variable
            newtrigger = True
            # repeat checking for triggering of tipping points as long as new get triggerd
            while permafrosttriggerd == True and False in permafrosttriggerd_parzelle and newtrigger == True and withtippingpoints == True:
                newtrigger = False
                if rainforestused:
                    # check for triggering of the die-off of the amazonian rainforest as before if not triggered before
                    if not rainforesttriggerd:
                        rainforest = calc_withrainforestemissions(future_inputs, rainforestemission_co2,
                                                                  temperature_pred, X, start, years, gafsize,
                                                                  numberoffeatures, actfactor, p_rainforest)
                        if rainforest is not None:
                            # updata input and trigger variable if tipping point was triggered
                            newtrigger = True
                            rainforesttriggerd = True
                            with_oldmodel = True
                            future_inputs = rainforest[1]
                            triggerrainforest = rainforest[2]
                            Xnew = preprocessing(future_inputs, years, gafsize, numberoffeatures, actfactor)
                # init variables
                triggeryearpermafrosttemp = 2018
                if permafrosttriggerd or rainforesttriggerd:
                    with_oldmodel = True
                # temperature prediction with modified input due to tipping points
                temperature_pred = frederike.predict(Xnew)
                if withpermafrost:
                    # iterate over all permafrost regions
                    for i in range(len(p)):
                        if not permafrosttriggerd_parzelle[i]:
                            # check for triggering of collapse of the boreal permafrost in the concering region as before
                            if anaerobe:
                                permafrost_parzelle = calc_withpermafrostemissions(future_inputs, p[i],
                                                                                   permafrostemission_co2[i],
                                                                                   temperature_pred, Xnew, start, years,
                                                                                   gafsize, numberoffeatures, actfactor,
                                                                                   anerobe=True, permafrostemission_ch4=
                                                                                   permafrostemission_ch4[i])
                            else:
                                permafrost_parzelle = calc_withpermafrostemissions(future_inputs, p[i],
                                                                                   permafrostemission_co2[i],
                                                                                   temperature_pred, Xnew, start, years,
                                                                                   gafsize, numberoffeatures, actfactor)
                        else:
                            permafrost_parzelle = None
                        if permafrost_parzelle is not None:
                            # if permafrost region was triggerd print info and updata input
                            # and trigger variables accordingly
                            print('Triggerd Permafrost Parzelle ' + parzellen_names[i])
                            newtrigger = True
                            permafrosttriggerd_parzelle[i] = True
                            future_inputs = permafrost_parzelle[1]

                            if permafrost_parzelle[2] != triggeryearpermafrosttemp:
                                triggeryearpermafrosttemp = permafrost_parzelle[2]
                                triggeryearspermafrost.append(permafrost_parzelle[2])
                                break
                # calculate new modified GAF
                Xnew = preprocessing(future_inputs, years, gafsize, numberoffeatures,
                                     actfactor)
                try:
                    triggeryearpermafrost = permafrost_parzelle_saved[2]
                except:
                    pass
                if permafrosttriggerd or rainforesttriggerd:
                    with_oldmodel = True

            if withtippingpoints:
                triggeryearspermafrost = np.array(triggeryearspermafrost)
                # final temperature prediction with chosen combination of models
                # (while taking new emissions of tipping points into account)
                if over200:
                    frederike = Training.buildSST(gafsize=gafsize, outputsize=outputsize, printsum=False,
                                                  modelname=modelname, features=X, new=False)
                    frederike.load_weights('../KlimaUi/climatesimulationAI/models/' + modelnametippingpoints)
                else:
                    frederike = Training.buildSST(gafsize=gafsize, outputsize=55297, printsum=False,
                                                  modelname=modelnametippingpoints, features=Xnew, new=buildnew)
                    frederike.load_weights('../KlimaUi/climatesimulationAI/models/' + modelname)
                temperature_pred = frederike.predict(Xnew)
                temperature_pred_old = frederike.predict(oldX)
                frederike = Training.buildSST(gafsize=gafsize, outputsize=115201, printsum=False,
                                              modelname="FrederikeSSTGADFGRIBwithact88.h5", features=Xnew, new=False)
                frederike.load_weights('../KlimaUi/climatesimulationAI/models/' + "FrederikeSSTGADFGRIBwithact88.h5")
                temperature_pred[:, 55296] = frederike.predict(Xnew)[:, 115200]
                temperature_pred_old[:, 55296] = frederike.predict(oldX)[:, 115200]

            else:
                if with_oldmodel:
                    # final temperature prediction with chosen combination of models
                    # (while NOT taking new emissions of tipping points into account)
                    if over200:
                        frederike = Training.buildSST(gafsize=gafsize, outputsize=outputsize, printsum=False,
                                                      modelname=modelname, features=X, new=False)
                        frederike.load_weights(
                            '../KlimaUi/climatesimulationAI/models/' + modelnametippingpoints)
                        temperature_pred = frederike.predict(X)
                    else:
                        pass
                    frederike = Training.buildSST(gafsize=gafsize, outputsize=115201, printsum=False,
                                                  modelname="FrederikeSSTGADFGRIBwithact88.h5", features=X, new=False)
                    frederike.load_weights(
                        '../KlimaUi/climatesimulationAI/models/' + "FrederikeSSTGADFGRIBwithact88.h5")
                    temperature_pred[:, 55296] = frederike.predict(X)[:, 115200]
                    temperature_pred_old = copy.deepcopy(temperature_pred)
                    temperature_pred_old[:, 55296] = frederike.predict(oldX)[:, 115200]

            if permafrosttriggerd:
                # init variable
                triggeryearspermafrost = np.array(triggeryearspermafrost)

            # denormalisation and postprocessing of the temperature prediction
            temperature_denorm = np.ones((years.size, temperature_pred[0, :].size))
            cmip6postproc = True
            for i in range(temperature_pred[0, :].size):
                temperature_denorm[:, i] = postprocessing(temperature_pred[:, i], years, i, withcmip6=cmip6postproc,
                                                          new=new, awi=awi)
            postprocessingtemp = copy.deepcopy(temperature_denorm)

            if with_oldmodel:
                # seperatedenormalisation and postprocessing of the temperature prediction for other model
                temperature_denorm_old = np.ones((years.size, temperature_pred[0, :].size))
                for i in range(temperature_pred_old[0, :].size):
                    temperature_denorm_old[:, i] = postprocessing(temperature_pred_old[:, i], years, i,
                                                                  withcmip6=cmip6postproc, new=new, awi=awi)
                postprocessingtemp_old = copy.deepcopy(temperature_denorm_old)

            tippingpointtriggerd = False
            if withtippingpoints:
                if permafrosttriggerd or rainforesttriggerd:
                    tippingpointtriggerd = True

            if tippingpointtriggerd:
                # combine trigger years of tipping points
                triggeryears = []
                if rainforesttriggerd and rainforestused:
                    triggeryears.append(triggerrainforest)
                for i in range(len(triggeryearspermafrost)):
                    triggeryears.append(triggeryearspermafrost[i])

                # apply different filters to denormalised temperature (special filter for tipping point simulation)
                tempsave = temperature_denorm
                new = np.ones(temperature_denorm.shape)
                for i in range(temperature_denorm[0, :].size):
                    new[:, i] = movingaverage_new(temperature_denorm[:, i], N=9, only_historical=True)
                newsave = new
                temperature_denorm = new

                newer = copy.deepcopy(temperature_denorm)
                for i in range(newer[0, :].size):
                    ked = triggeryears[len(triggeryears) - 1] + 8
                    led = triggeryears[len(triggeryears) - 1] + 2
                    med = triggeryears[len(triggeryears) - 1] + 20
                    if ked>99:
                        ked = 99
                    if led>99:
                        led = 99
                    if (temperature_denorm[ked, i] - temperature_denorm[led, i]) < 0:
                    #if (temperature_denorm[triggeryears[len(triggeryears) - 1] + 8, i] - temperature_denorm[
                     #   triggeryears[len(triggeryears) - 1] + 2, i]) < 0:
                        newer[led:, i] = (temperature_denorm[
                                                                              led:,
                                                                              i] - (temperature_denorm[ked, i] -
                                                                                    temperature_denorm[led, i])) + 0.5
                    else:
                        newer[led:, i] = (temperature_denorm[
                                                                              led:,
                                                                              i] + (temperature_denorm[ked, i] -
                                                                                    temperature_denorm[led, i])) + 0.5
                    newer[ked:med,
                    i] = movingaverage_new(
                        newer[ked:med, i],
                        N=10, only_historical=True)

                    newer[triggeryears[0]:, i] = movingaverage(newer[triggeryears[0]:, i], N=15)

                    newer[:, i] = movingaverage_new(newer[:, i], N=4, only_historical=True)

                    if rainforesttriggerd:
                        newer[:triggeryears[0], i] = postprocessingtemp[:triggeryears[0], i]
                        newer[:, i] = movingaverage_new(newer[:, i], N=8, only_historical=True)

                    fed = triggeryears[0] + 33
                    if fed > 99:
                        fed = 99
                    if rainforesttriggerd and permafrosttriggerd:
                        if triggeryears[1] - triggeryears[0] > 23:
                            newer[triggeryears[0]:fed, i] = newer[triggeryears[0]:fed,
                                                                             i] + 0.6
                            newer[:, i] = movingaverage(newer[:, i],
                                                        N=7)
                        else:
                            newer[triggeryears[0]:triggeryears[1], i] = newer[triggeryears[0]:triggeryears[1], i] + 0.6
                            newer[:, i] = movingaverage(newer[:, i], N=7)

                        if len(triggeryears) > 3:
                            newer[triggeryears[len(triggeryears) - 2]:triggeryears[len(triggeryears) - 1], i] = newer[
                                                                                                                triggeryears[
                                                                                                                    len(triggeryears) - 2]:
                                                                                                                triggeryears[
                                                                                                                    len(triggeryears) - 1],
                                                                                                                i] + 0.45
                            newer[:, i] = movingaverage(newer[:, i], N=7)
                    if rainforesttriggerd:
                        newer[:triggeryears[0], i] = postprocessingtemp_old[:triggeryears[0],
                                                     i]
                        newer[:, i] = movingaverage_new(newer[:, i], N=8, only_historical=True)
                        newer[:10, i] = postprocessingtemp[:10, i]
                    else:
                        newer[:triggeryears[0], i] = postprocessingtemp[:triggeryears[0], i]
                        newer[triggeryears[0] - 5:triggeryears[0], i] = newer[triggeryears[0] - 5:triggeryears[0],
                                                                        i] + 0.3
                        newer[:, i] = movingaverage_new(newer[:, i], N=10, only_historical=True)
                        newer[:46, i] = postprocessingtemp[:46, i]
            # final filter for temperature simulation
            if tippingpointtriggerd:
                output = newer
                newtemp = np.ones(temperature_denorm_old.shape)
                for i in range(temperature_denorm_old[0, :].size):
                    newtemp[:, i] = movingaverage(temperature_denorm_old[:, i], N=2)
            else:
                newtemp = np.ones(temperature_denorm.shape)
                for i in range(temperature_denorm[0, :].size):
                    newtemp[:, i] = movingaverage(temperature_denorm[:, i], N=2)
                #output = newtemp
            oldtemp = copy.deepcopy(newtemp)
            if withcmip6:
                if awi:
                    u = 73728
                else:
                    u = 55297
            else:
                u = 115201
            newpredf = np.ones(newtemp.shape)
            for i in range(100):
                newpredf[i, :u-1] = newtemp[i, :u-1] - (math.sqrt(99 - (i)) * 0.85)
            newpredf[:, :u-1] = newpredf[:, :u-1] + 17
            newtemp[:, :u-1] = gaussian_filter(newpredf[:, :u-1], sigma=2)
            output = newtemp

            if predsea:
                # sea level simulation

                # init variables
                if permafrosttriggerd:
                    X = copy.deepcopy(Xnew)
                gafsizesea = 20
                # outputsize accordingly to used grid
                temperature_denorm = output

                # calculate combined GAF only for global temperature
                all5temperature = preprocessing(np.array(newtemp[:, u - 1:u].transpose()), years, gafsizesea,
                                                1, 0.8, s=6)
                # calculate whether tipping point collapse of the west-antarctic ice shield was triggerd
                # and its implications (if this tipping point is turned on)
                additionalsealevel = None
                if withtippingpoints and wais:
                    additionalsealevel = calc_wais_sealevel(temperature_denorm)
                # combine global temperature GAF and ghg GAF
                Xsea = copy.deepcopy(X)
                X[:, :, 120:] = all5temperature[:]
                if method_SST:
                    if newsea:
                        # reformat and normalised predicted temperature grid for input of the sea level simulation
                        lo = 192
                        la = 288
                        temperature_norm = np.ones((len(X), temperature_denorm[0, :].size))
                        for k in range(temperature_denorm[0, :].size):
                            temperature_norm[:, k] = producetarget6withoutwalks(
                                my_norm_tempgrib(newtemp[:, k]))
                        tempgribmaped = temperature_norm[:, :outputsize - 1].reshape(temperature_norm[:, 0].size, lo,
                                                                                     la, 1)
                        # build sea level simulation model (taking into account regionalised temperature)
                        kala = Training.buildSST(gafsize=gafsize, outputsize=1, printsum=False, modelname=modelnamesea,
                                                 features=[X, tempgribmaped], new=True, sea=False, seasea=newsea)
                        kala.load_weights('../KlimaUi/climatesimulationAI/models/' + modelnamesea)
                    else:
                        # build sea level simulation model (WITHOUT taking into account regionalised temperature)
                        kala = Training.buildSST(gafsize=gafsize, outputsize=1, printsum=False, modelname=modelnamesea,
                                                 features=X, num_heads=4, old=True, new=False)
                        kala.load_weights('../KlimaUi/climatesimulationAI/models/' + modelnamesea)
                elif method_Conv:
                    # build sea level simulation model with CNN architecture
                    kala = Training.buildConv(gafsize=gafsize, inputshape=X[0].shape, printsum=False)
                    kala.load_weights('../KlimaUi/climatesimulationAI/models/' + modelnamesea)

                # sea level prediction (different input depending on model)
                if newsea:
                    sealevel_pred = kala.predict([X, tempgribmaped])
                else:
                    sealevel_pred = kala.predict(Xsea)
                # apply filter to sea level predction
                newseas = sealevel_pred
                for i in range(sealevel_pred[0, :].size):
                    newseas[:, i] = movingaverage_new(sealevel_pred[:, i], N=10, only_historical=True)
                for i in range(sealevel_pred[0, :].size):
                    newseas[:, i] = movingaverage(newseas[:, i], N=5)
                # denormalisation and postprocessing of sea level prediction
                sealevel_denorm = np.ones((years.size, newseas[0, :].size))
                for i in range(newseas[0, :].size):
                    sealevel_denorm[:, i] = postprocessing(newseas[:, i], years, i, sealevel=True,
                                                           withcmip6=withcmip6, awi=awi, newsea=newsea)
                sealevel_denorm = sealevel_denorm[:, 0]
                sealevel_without = copy.deepcopy(sealevel_denorm)
                # add additional sea level rise from tipping point
                # collapse of the west-antarctic ice shield (if triggerd and turned on)
                if wais == True and additionalsealevel is not None:
                    for i in range(len(sealevel_denorm)):
                        sealevel_denorm[i] = sealevel_denorm[i] + additionalsealevel[i]
                    # apply filter
                    sealevel_denorm = movingaverage_new(sealevel_denorm[:], N=3, only_historical=True)
                    sealevel_denorm[:4] = sealevel_without[:4]

                # handly fragment/edge cases of the sea level simulation
                if sealevel_denorm[0] > sealevel_denorm[len(sealevel_denorm) - 1]:
                    print(sealevel_denorm[0])
                    sea_end = sealevel_denorm[0] + (sealevel_denorm[0] - sealevel_denorm[len(sealevel_denorm) - 1])
                    print(sea_end)
                    sea_cen = sealevel_denorm[0] + ((sea_end - sealevel_denorm[0]) / 2) - (
                            (sea_end - sealevel_denorm[0]) / 10)
                    print(sea_cen)
                    sealevel_denorm = np.interp(np.linspace(0, len(sealevel_denorm), len(sealevel_denorm)),
                                                [0, len(sealevel_denorm) / 2, len(sealevel_denorm)],
                                                [sealevel_denorm[0], sea_cen, sea_end])
                    sealevel_denorm = movingaverage(sealevel_denorm, N=15)
                    output = [temperature_denorm, sealevel_denorm]
                elif max(sealevel_denorm) > (sealevel_denorm[99] * 1.02):
                    print('fit')
                    years_val = [0, 99]
                    deg = 1
                    val = [sealevel_denorm[0], sealevel_denorm[99]]
                    fit = np.polyfit(years_val, val, deg)
                    years = np.arange(100)
                    all_val = []
                    for k in range(100):
                        all_val.append(np.polyval(fit, years[k]))

                    sealevel_denorm = np.array(all_val)
                    output = [temperature_denorm, sealevel_denorm]

                else:
                    output = [temperature_denorm, sealevel_denorm[:]]

        else:
            # return combined GAF or only emission developments if wished
            output = X
            if onlyemidata:
                output = future_inputs/1000

        return output


def producetarget6withoutwalks(normdata):
    # normdata: data to be used, np.array, one dimensional
    # takes every 6th value of normdata and combines them
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


def my_norm_tempgrib(a):
    # normalisation for the temperature grib on a scale from -1 to 1

    # overall minima and maxima
    maxi = 46.706146
    mini = -55.40129
    ratio = 2 / (maxi - mini)
    # as you want your data to be between -1 and 1, everything should be scaled to 2,
    # if your desired min and max are other values, replace 2 with your_max - your_min
    shift = (maxi + mini) / 2
    # now you need to shift the center to the middle, this is not the average of the values.
    return (a - shift) * ratio
