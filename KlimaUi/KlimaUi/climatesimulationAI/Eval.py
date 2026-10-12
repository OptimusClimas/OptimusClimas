# module for automated evaluations

# import of needed libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from KlimaUi.climatesimulationAI import simulation


def meandifference(mysim, vglsim):
    # mysim, vglsim: np.array
    # calculates the absolute mean difference
    diff = abs(mysim - vglsim)
    return np.mean(diff)


def get_sub(x):
    # x: character to get subscript

    # subscript of a character
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


def comparisontippingpoints(modelname, onlytrigger=False, nccmip6=True, sea=False,
                            modelnamesea='kalaSST100.h5', anaerobe=True, rainforest=True, wais=False):
    # modelname: filename of the temperature model to be evaluated, string
    # onlytrigger: whether to only plot the co2 and ch4 emissions, boolean
    # nccmip6: if grid of cmip6 imported in a netCDF format is used, boolean (default true)
    # sea: whether sea level is to be predicted (and evaluated), boolean
    # modelnamesea: filename of the sea level model to be evaluated, string (only if sea=true)
    # anaerobe: whether anaerobe conditions are to be assumed for the collapse of the permafrost, boolean
    # rainforest: whether the tipping point die-off of the amazonian rainforest is to be taken into account, boolean
    # wais: only plots scenarios SSP3-7.0 and SSP5-8.5 otherwise same functionality as sea, boolean
    if wais:
        sea = True
    fontsizelabel = 17
    
    # plots the predictions of a model for different scenarios with and without taking tipping points into account
    if not sea:
        # temperature evaluation
        # outputsize accordingly to use grid
        if nccmip6:
            u = 55296
        else:
            u = 115200

        # init variables

        # init scenarios
        ghgchangesssp5 = [200, 200, 50, -60, -60, -43]
        ghgchangespermafrost3 = [300, 300, 100, -50, -50, -33]
        ghgchangespermafrost4 = [400, 400, 150, -40, -40, -38]
        ghgchangespermafrost5 = [500, 500, 200, -30, -30, -23]

        # predictions in different scenarios with and without consideration of tipping points
        predfssp5 = simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                    withtippingpoints=True, nccmip6=nccmip6, anaerobe=anaerobe,
                                    rainforestused=rainforest, onlyemidata=onlytrigger)
        predfssp5without = simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                           withtippingpoints=False, nccmip6=nccmip6, onlyemidata=onlytrigger)
        predfperma3 = simulation.pred(ghgchanges=ghgchangespermafrost3, start=2014, end=2114, modelname=modelname,
                                      withtippingpoints=True, nccmip6=nccmip6, anaerobe=anaerobe,
                                      rainforestused=rainforest, onlyemidata=onlytrigger)
        predfperma3without = simulation.pred(ghgchanges=ghgchangespermafrost3, start=2014, end=2114,
                                             modelname=modelname, withtippingpoints=False, nccmip6=nccmip6,
                                             onlyemidata=onlytrigger)
        predfperma4without = simulation.pred(ghgchanges=ghgchangespermafrost4, start=2014, end=2114,
                                             modelname=modelname, withtippingpoints=False, nccmip6=nccmip6,
                                             onlyemidata=onlytrigger)
        predfperma4 = simulation.pred(ghgchanges=ghgchangespermafrost4, start=2014, end=2114, modelname=modelname,
                                      withtippingpoints=True, nccmip6=nccmip6, anaerobe=anaerobe,
                                      rainforestused=rainforest, onlyemidata=onlytrigger)
        plt.figure(figsize=(9, 6))
        if not onlytrigger:
            # plot temperature predictions and calculate (and print) absolut and relative difference
            plt.plot(np.arange(2014, 2114), predfssp5without[:, u], 'b', linestyle='--')
            plt.plot(np.arange(2014, 2114), predfssp5[:, u], 'r')
            print('diff absolut ssp5: ' + str(predfssp5[99, u] - predfssp5[0, u]))
            print('diff relative ssp5: ' + str(predfssp5[99, u] - predfssp5without[99, u]))
            plt.plot(np.arange(2014, 2114), predfperma3without[:, u], 'steelblue', linestyle='--')
            plt.plot(np.arange(2014, 2114), predfperma3[:, u], 'magenta')
            print('diff absolut 4x: ' + str(predfperma3[99, u] - predfperma3[0, u]))
            print('diff relative 4x: ' + str(predfperma3[99, u] - predfperma3without[99, u]))
            plt.plot(np.arange(2014, 2114), predfperma4without[:, u], 'mediumturquoise', linestyle='--')
            plt.plot(np.arange(2014, 2114), predfperma4[:, u], 'blueviolet')
            print('diff absolut 5x: ' + str(predfperma4[99, u] - predfperma4[0, u]))
            print('diff relative 5x: ' + str(predfperma4[99, u] - predfperma4without[99, u]))
        else:
            # plot predicted emission developments for CO2
            fig, ax1 = plt.subplots(figsize=(9, 6))
            color = 'black'
            ax1.set_xlabel('year', fontsize=fontsizelabel)
            ax1.set_ylabel('CH{}'.format(get_sub('4')) + ' emissions in Gt', color=color, fontsize=fontsizelabel)
            ax1.plot(np.arange(2014, 2114), predfssp5without[2, :] / 1000, 'b', linestyle='--')
            ax1.plot(np.arange(2014, 2114), predfssp5[2, :] / 1000, 'r')
            ax1.plot(np.arange(2014, 2114), predfperma3without[2, :] / 1000, 'steelblue', linestyle='--')
            ax1.plot(np.arange(2014, 2114), predfperma3[2, :] / 1000, 'magenta')
            ax1.plot(np.arange(2014, 2114), predfperma4without[2, :] / 1000, 'mediumturquoise', linestyle='--')
            ax1.plot(np.arange(2014, 2114), predfperma4[2, :] / 1000, 'blueviolet')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(['CH{}'.format(get_sub('4')) + ' SSP 5-8.5 without tipping points',
                        'CH{}'.format(get_sub('4')) + ' SSP 5-8.5 with tipping points',
                        'CH{}'.format(get_sub('4')) + ' 4x without tipping points',
                        'CH{}'.format(get_sub('4')) + ' 4x with tipping points',
                        'CH{}'.format(get_sub('4')) + ' 5x without tipping points',
                        'CH{}'.format(get_sub('4')) + ' 5x with tipping points',
                        'CH{}'.format(get_sub('4')) + ' 6x without tipping points',
                        'CH{}'.format(get_sub('4')) + ' 6x with tipping points'], loc='upper left',
                       fontsize=fontsizelabel*0.65)  #'CH{}'.format(get_sub('4')) + '

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            # plot predicted emission developments for CH4
            color = 'black'
            ax2.set_ylabel('CO{}'.format(get_sub('2')) + ' emissions in Gt', color=color,
                           fontsize=fontsizelabel)  # we already handled the x-label with ax1
            ax2.plot(np.arange(2014, 2114), predfssp5without[1, :] / 1000, 'royalblue', linestyle='--')
            ax2.plot(np.arange(2014, 2114), predfssp5[1, :] / 1000, 'crimson')
            ax2.plot(np.arange(2014, 2114), predfperma3without[1, :] / 1000, 'deepskyblue', linestyle='--')
            ax2.plot(np.arange(2014, 2114), predfperma3[1, :] / 1000, 'darkmagenta')
            ax2.plot(np.arange(2014, 2114), predfperma4without[1, :] / 1000, 'mediumaquamarine', linestyle='--')
            ax2.plot(np.arange(2014, 2114), predfperma4[1, :] / 1000, 'mediumorchid')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(['CO{}'.format(get_sub('2')) + ' SSP 5-8.5 without tipping points',
                        'CO{}'.format(get_sub('2')) + '  SSP 5-8.5 with tipping points',
                        'CO{}'.format(get_sub('2')) + '  4x without tipping points',
                        'CO{}'.format(get_sub('2')) + ' 4x with tipping points',
                        'CO{}'.format(get_sub('2')) + ' 5x without tipping points',
                        'CO{}'.format(get_sub('2')) + ' 5x with tipping points',
                        'CO{}'.format(get_sub('2')) + ' 6x without tipping points',
                        'CO{}'.format(get_sub('2')) + ' 6x with tipping points'], loc='upper right', fontsize=fontsizelabel*0.65)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.xticks(fontsize=fontsizelabel*0.75)
        plt.yticks(fontsize=fontsizelabel*0.75)
        if not onlytrigger:
            plt.legend(
                ['SSP 5-8.5 without tipping points', 'SSP 5-8.5 with tipping points', '4x without tipping points',
                 '4x with tipping points', '5x without tipping points', '5x with tipping points',
                 '6x without tipping points', '6x with tipping points'], loc='upper left', fontsize=fontsizelabel*0.65)
            plt.title('temperature simulation while considering tipping points', fontsize=fontsizelabel)
            plt.ylabel('global mean tempeature in °C', fontsize=fontsizelabel)
            plt.xlabel('year', fontsize=fontsizelabel)

    else:
        # sea level evaluation
        # outputsize accordingly to used grid
        if nccmip6:
            u = 55296
        else:
            u = 115200


        # init scenarios
        ghgchangesssp3 = [100, 100, 100, -1, -10, -1]
        ghgchangesssp5 = [200, 200, 50, -60, -60, -43]
        ghgchangespermafrost3 = [300, 300, 100, -50, -50, -33]
        ghgchangespermafrost4 = [400, 400, 150, -40, -40, -38]
        ghgchangespermafrost5 = [500, 500, 200, -30, -30, -23]
        # predictions in different scenarios with and without consideration of tipping points
        if wais:
            predfssp3 = simulation.pred(ghgchanges=ghgchangesssp3, start=2014, end=2114, modelname=modelname,
                                        withtippingpoints=True, predsea=True, modelnamesea=modelnamesea,
                                        nccmip6=nccmip6,
                                        onlyemidata=onlytrigger)
            predfssp3without = simulation.pred(ghgchanges=ghgchangesssp3, start=2014, end=2114, modelname=modelname,
                                               withtippingpoints=False, predsea=True, modelnamesea=modelnamesea,
                                               nccmip6=nccmip6, onlyemidata=onlytrigger)
        predfssp5 = simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                    withtippingpoints=True, predsea=True, modelnamesea=modelnamesea, nccmip6=nccmip6,
                                    onlyemidata=onlytrigger)
        predfssp5without = simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                           withtippingpoints=False, predsea=True, modelnamesea=modelnamesea,
                                           nccmip6=nccmip6, onlyemidata=onlytrigger)
        if not wais:
            predfperma3 = simulation.pred(ghgchanges=ghgchangespermafrost3, start=2014, end=2114, modelname=modelname,
                                          withtippingpoints=True, predsea=True, modelnamesea=modelnamesea, nccmip6=nccmip6,
                                          onlyemidata=onlytrigger)
            predfperma3without = simulation.pred(ghgchanges=ghgchangespermafrost3, start=2014, end=2114,
                                                 modelname=modelname, withtippingpoints=False, predsea=True,
                                                 modelnamesea=modelnamesea, nccmip6=nccmip6, onlyemidata=onlytrigger)
            predfperma4without = simulation.pred(ghgchanges=ghgchangespermafrost4, start=2014, end=2114,
                                                 modelname=modelname, withtippingpoints=False, predsea=True,
                                                 modelnamesea=modelnamesea, nccmip6=nccmip6, onlyemidata=onlytrigger)
            predfperma4 = simulation.pred(ghgchanges=ghgchangespermafrost4, start=2014, end=2114, modelname=modelname,
                                          withtippingpoints=True, predsea=True, modelnamesea=modelnamesea, nccmip6=nccmip6,
                                          onlyemidata=onlytrigger)
        plt.figure(figsize=(9, 6))
        if not onlytrigger:
            if wais:
                plt.plot(np.arange(2014, 2114), predfssp3without[1], 'b', linestyle='--')
                plt.plot(np.arange(2014, 2114), predfssp3[1], 'r')
                plt.plot(np.arange(2014, 2114), predfssp5without[1], 'steelblue', linestyle='--')
                plt.plot(np.arange(2014, 2114), predfssp5[1], 'magenta')
            if not wais:
                plt.plot(np.arange(2014, 2114), predfssp5without[1], 'b', linestyle='--')
                plt.plot(np.arange(2014, 2114), predfssp5[1], 'r')
                plt.plot(np.arange(2014, 2114), predfperma3without[1], 'steelblue', linestyle='--')
                plt.plot(np.arange(2014, 2114), predfperma3[1], 'magenta')
                plt.plot(np.arange(2014, 2114), predfperma4without[1], 'mediumturquoise', linestyle='--')
                plt.plot(np.arange(2014, 2114), predfperma4[1], 'blueviolet')

        plt.xticks(fontsize=fontsizelabel*0.75)
        plt.yticks(fontsize=fontsizelabel*0.75)
        plt.xlabel('year',fontsize=fontsizelabel)
        if wais:
            plt.legend(
                ['SSP 3-7.0 without tipping points', 'SSP 3-7.0 with tipping points',
                 'SSP 5-8.5 without tipping points', 'SSP 5-8.5 with tipping points',], loc='upper left',
                fontsize=fontsizelabel * 0.65)
        else:
            plt.legend(['SSP 5-8.5 without tipping points', 'SSP 5-8.5 with tipping points', '4x without tipping points',
                        '4x with tipping points', '5x without tipping points', '5x with tipping points',
                        '6x without tipping points', '6x with tipping points'], loc='upper left', fontsize=fontsizelabel*0.65)
        if not onlytrigger:
            plt.title('Sea level simulation while considering tipping points', fontsize=fontsizelabel)
            plt.ylabel('Rise in mm since 1880', fontsize=fontsizelabel)


def rmse(pred, ref):
    # root mean square error between a prediction and a reference series
    # both series are cut to the shorter length, so differing lengths don't raise an error
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n = min(len(pred), len(ref))
    return float(np.sqrt(np.mean((pred[:n] - ref[:n]) ** 2)))


def comparisonssp(modelname=None, complete=False, sealevel=False, modelnamesea=None, actfactore=0.8, nccmip6=False,
                  withconv=False):
    # modelname: filename of the temperature model to be evaluated, string
    # complete: whether to simulate and plot all ssp scenarios or just ssp3 and ssp5, boolean
    # sealevel: whether to simulate and plot the sea level, boolean
    # modelnamesea: filename of the sea level model to be evaluated, string
    # actfactore: factor sigmoid is multiplied by in the pre-activation, float (default 0.8, determined experimentally)
    # nccmip6: if grid of cmip6 imported in a netCDF format is used, boolean (default true),
    # withConv: whether to include simulations with models with CNN architecture, boolean

    # simulates and plots in SSP scenarios to compare them with the results of the CMIP6
    # outputsize accordingly to used grid
    if nccmip6:
        u = 55296
    else:
        u = 115200

        # comparison data from AC-SSP (Clyne, Kaltenborn, Nowack et al. 2026) in scenario SSP 5-8.5
    acssp_585 = np.array([    np.nan, 287.3055, 287.7174, 287.6072, 287.6850, 287.6687, 287.6535,
             287.7515, 287.8687, 287.9452, 287.9281, 287.8873, 287.9887, 287.8922,
             287.8052, 287.8077, 288.2756, 287.9176, 287.8686, 287.8796, 288.2068,
             288.4895, 288.3094, 288.4645, 288.4551, 288.6043, 288.6341, 288.8525,
             288.5998, 288.4703, 288.5724, 288.7175, 289.0687, 288.7529, 288.9186,
             289.2954, 289.1680, 289.4060, 289.3069, 289.4037, 289.4236, 289.5782,
             289.8005, 289.7238, 289.3828, 289.8306, 289.8087, 289.6198, 289.7526,
             290.0502, 289.9124, 290.2156, 289.9640, 290.3358, 290.4853, 290.7931,
             290.6207, 290.5571, 290.6600, 290.6107, 290.9875, 290.9047, 290.7757,
             291.1861, 291.0694, 291.2997, 291.0657, 291.3198, 291.4979, 291.4727,
             291.8157, 291.7326, 291.8054, 291.9039, 292.2302, 292.1447, 292.1682,
             291.9789, 292.1664, 292.4657, 292.6799, 292.3438, 292.6275, 292.3598,
             292.8136])-273.15
    dates_acssp = pd.DatetimeIndex(['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01',
                   '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01',
                   '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01',
                   '2027-01-01', '2028-01-01', '2029-01-01', '2030-01-01',
                   '2031-01-01', '2032-01-01', '2033-01-01', '2034-01-01',
                   '2035-01-01', '2036-01-01', '2037-01-01', '2038-01-01',
                   '2039-01-01', '2040-01-01', '2041-01-01', '2042-01-01',
                   '2043-01-01', '2044-01-01', '2045-01-01', '2046-01-01',
                   '2047-01-01', '2048-01-01', '2049-01-01', '2050-01-01',
                   '2051-01-01', '2052-01-01', '2053-01-01', '2054-01-01',
                   '2055-01-01', '2056-01-01', '2057-01-01', '2058-01-01',
                   '2059-01-01', '2060-01-01', '2061-01-01', '2062-01-01',
                   '2063-01-01', '2064-01-01', '2065-01-01', '2066-01-01',
                   '2067-01-01', '2068-01-01', '2069-01-01', '2070-01-01',
                   '2071-01-01', '2072-01-01', '2073-01-01', '2074-01-01',
                   '2075-01-01', '2076-01-01', '2077-01-01', '2078-01-01',
                   '2079-01-01', '2080-01-01', '2081-01-01', '2082-01-01',
                   '2083-01-01', '2084-01-01', '2085-01-01', '2086-01-01',
                   '2087-01-01', '2088-01-01', '2089-01-01', '2090-01-01',
                   '2091-01-01', '2092-01-01', '2093-01-01', '2094-01-01',
                   '2095-01-01', '2096-01-01', '2097-01-01', '2098-01-01',
                   '2099-01-01'],
                  dtype='datetime64[us]', freq='YS-JAN')

    if modelname is not None:
        # 0: GHG, 1: CO2, 2: CH4, 3: BC, 4: SO2, 5: OC
        # init scenarios
        ghgchangesssp19 = [-100, -100, -60, -80, -85, -58]
        ghgchangesssp126 = [-90, --90, -60, -77, -88, -58]
        ghgchangesssp2 = [-75, -75, -25, -55, -60, -55]
        ghgchangesssp3 = [100, 100, 100, -1, -10, -1]
        ghgchangesssp5 = [200, 200, 50, -60, -60, -43]
        years = np.arange(2020, 2100)
        if not sealevel:
            if complete:
                plt.figure(figsize=(9, 6))
                # predict, plot and calculate deviation to CMIP6 results in negative SSP scenarios
                simpredt = np.array(
                    simulation.pred(ghgchanges=ghgchangesssp19, start=2014, end=2114, modelname=modelname,
                                    method_SST=True, method_Conv=False))
                simpred = simpredt[:80, u]
                plt.plot(years, np.array(simpred) - 0.2, 'c')
                plt.plot(years, np.load("ssp19temp.npy", mmap_mode="r+") - 0.2, 'c', linestyle='--')
                print('vgl ssp19: ' + str(meandifference(simpred, np.load("ssp19temp.npy", mmap_mode="r+"))))
                print('rmse ssp19: ' + str(rmse(simpred, np.load("ssp19temp.npy", mmap_mode="r+"))))
                simpredt = np.array(
                    simulation.pred(ghgchanges=ghgchangesssp126, start=2014, end=2114, modelname=modelname,
                                    method_SST=True, method_Conv=False))
                simpred = simpredt[:80, u]
                plt.plot(years, np.array(simpred) - 0.2, 'blue')
                print('vgl ssp126: ' + str(meandifference(simpred, np.load("ssp126temp.npy", mmap_mode="r+"))))
                print('rmse ssp126: ' + str(rmse(simpred, np.load("ssp126temp.npy", mmap_mode="r+"))))
                plt.plot(years, np.load("ssp126temp.npy", mmap_mode="r+") - 0.2, 'blue', linestyle='--')
                simpredt = np.array(
                    simulation.pred(ghgchanges=ghgchangesssp2, start=2014, end=2114, modelname=modelname,
                                    method_SST=True, method_Conv=False))
                simpred = simpredt[:80, u]
                plt.plot(years, np.array(simpred) - 0.2, 'orange')
                print('vgl ssp2: ' + str(meandifference(simpred, np.load("ssp2temp.npy", mmap_mode="r+"))))
                print('rmse ssp2: ' + str(rmse(simpred, np.load("ssp2temp.npy", mmap_mode="r+"))))
                plt.plot(years, np.load("ssp2temp.npy", mmap_mode="r+") - 0.2, 'orange', linestyle='--')
                plt.legend(
                    ['SSP1-1.9 VT', 'SSP1-1.9 IPCC', 'SSP1-2.6 VT', 'SSP1-2.6 IPCC', 'SSP2-4.5 VT', 'SSP2-4.5 IPCC',
                     'SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC'], loc='upper left')
            # predict, plot and calculate deviation to CMIP6 results in positive SSP scenarios
            # metrics use exactly the curves that are plotted (incl. the -0.2 / +0.75 offsets)
            simpredt = np.array(
                simulation.pred(ghgchanges=ghgchangesssp3, start=2014, end=2114, modelname=modelname,
                                method_SST=True, method_Conv=False))[:80]
            simpred = simpredt[:80, u]
            ssp3_shown_pred = np.array(simpred) - 0.2
            ssp3_shown_ref = np.load("all_ssp3Temperatur.npy", mmap_mode="r+")[:80]
            plt.plot(years, ssp3_shown_pred, 'g')
            plt.plot(years, ssp3_shown_ref, 'g', linestyle='--')

            print('vgl ssp3: ' + str(meandifference(ssp3_shown_pred, ssp3_shown_ref)))
            print('rmse ssp3: ' + str(rmse(ssp3_shown_pred, ssp3_shown_ref)))
            if withconv:
                ssp3_shown_conv = np.load('predconvtempglobalssp3.npy', mmap_mode='r+')[:80] + 0.75
                plt.plot(np.arange(2020, 2100), ssp3_shown_conv, 'olive')
                print('vgl ssp3 with conv: ' + str(meandifference(ssp3_shown_conv, ssp3_shown_ref)))
                print('rmse ssp3 with conv: ' + str(rmse(ssp3_shown_conv, ssp3_shown_ref)))
            simpredt = np.array(
                simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                method_SST=True, method_Conv=False))
            simpred = simpredt[:80, u]
            ssp5_shown_pred = np.array(simpred) - 0.2
            ssp5_shown_ref = np.load("ipccSSP5Temperature1.npy", mmap_mode="r+")[:80]
            plt.plot(years, ssp5_shown_pred, 'r')
            plt.plot(years, ssp5_shown_ref, 'r', linestyle='--')
            plt.plot(dates_acssp.year, acssp_585, 'orange')
            
            print('vgl ssp5 vs ac-ssp: ' + str(meandifference(acssp_585[5:], ssp5_shown_ref[:])))
            print('vgl ssp5: ' + str(meandifference(ssp5_shown_pred, ssp5_shown_ref)))
            print('rmse ssp5: ' + str(rmse(ssp5_shown_pred, ssp5_shown_ref)))
            if withconv:
                ssp5_shown_conv = np.load('predconvtempglobalssp5.npy', mmap_mode='r+')[:80] + 0.75
                plt.plot(np.arange(2020, 2100), ssp5_shown_conv, 'maroon')
                print('vgl ssp5 with conv: ' + str(meandifference(ssp5_shown_conv, ssp5_shown_ref)))
                print('rmse ssp5 with conv: ' + str(rmse(ssp5_shown_conv, ssp5_shown_ref)))
            if not complete:
                plt.yticks(ticks=(np.arange(14.5, 21, 0.5)))
            else:
                plt.yticks(ticks=(np.arange(13, 21, 0.5)))
            # plot error range of the CMIP6
            plt.fill_between(np.arange(2020, 2100), np.load('all_ssp3Temperaturlow.npy', mmap_mode="r+")[:80],
                             np.load('all_ssp3Temperaturhigh.npy', mmap_mode="r+")[:80], color='springgreen',
                             alpha=.1)
            plt.fill_between(np.arange(2020, 2100), np.load('all_ssp5Temperaturlow.npy', mmap_mode="r+")[:80],
                             np.load('all_ssp5Temperaturhigh.npy', mmap_mode="r+")[:80], color='lightcoral',
                             alpha=.1)
            plt.title('temperature simulation in SSP scenarios in comparison', fontsize=17)
            plt.ylabel('global mean temperature in °C', fontsize=17)
            plt.xlabel('year', fontsize=17)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if not complete:
                plt.legend(['SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC', 'SSP5-8.5 AC-SSP'], loc='upper left')
                if withconv:
                    plt.legend(['SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP3-7.0 Conv', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC',
                                'SSP5-8.5 Conv'], loc='upper left')
            else:
                plt.legend(
                    ['SSP1-1.9 VT', 'SSP1-1.9 IPCC', 'SSP1-2.6 VT', 'SSP1-2.6 IPCC', 'SSP2-4.5 VT', 'SSP2-4.5 IPCC',
                     'SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP3-7.0 IPCC error', 'SSP5-8.5 VT',
                     'SSP5-8.5 IPCC', 'SSP5-8.5 IPCC error'], loc='upper left')
            plt.savefig("eval.jpg",dpi=500)
        else:
            # sea level comparison
            # predict and plot sea level in positive ssp scenarios
            predf = simulation.pred(ghgchanges=ghgchangesssp3, start=2014, end=2114, modelname=modelname,
                                    actfactor=actfactore, predsea=True, modelnamesea=modelnamesea)
            ssp3sealevel = np.load('ssp3sealevel.npy', mmap_mode='r+')
            ssp3sealevellow = np.load('ssp3sealevellow.npy', mmap_mode='r+')
            ssp3sealevelhigh = np.load('ssp3sealevelhigh.npy', mmap_mode='r+')
            plt.plot(np.arange(2020, 2110), predf[1][6:96], 'g')
            plt.plot(np.arange(2020, 2110), ssp3sealevel, 'g', linestyle='--')
            print(
                'vgl sea ssp3: ' + str(meandifference(predf[1][6:96], ssp3sealevel)))
            print(
                'rmse sea ssp3: ' + str(rmse(predf[1][6:96], ssp3sealevel)))

            if withconv:
                ssp3_shown_conv = np.load('predconvseaglobalssp3.npy', mmap_mode='r+')[:80] + 0.75
                plt.plot(np.arange(2020, 2100), ssp3_shown_conv, 'olive')
                print('vgl ssp3 with conv: ' + str(meandifference(ssp3_shown_conv, ssp3sealevel)))
                print('rmse ssp3 with conv: ' + str(rmse(ssp3_shown_conv, ssp3sealevel)))
            predf = simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                    actfactor=actfactore, predsea=True, modelnamesea=modelnamesea)
            ssp5sealevel = np.load('ssp5sealevel.npy', mmap_mode='r+')
            ssp5sealevellow = np.load('ssp5sealevellow.npy', mmap_mode='r+')
            ssp5sealevelhigh = np.load('ssp5sealevelhigh.npy', mmap_mode='r+')
            plt.plot(np.arange(2020, 2110), predf[1][6:96], 'r')
            plt.plot(np.arange(2020, 2110), ssp5sealevel, 'r', linestyle='--')
            print('vgl sea ssp5: ' + str(
                meandifference(predf[1][6:96], ssp5sealevel)))
            print('rmse sea ssp5: ' + str(
                rmse(predf[1][6:96], ssp5sealevel)))
            if withconv:
                ssp5_shown_conv = np.load('predconvseaglobalssp5.npy', mmap_mode='r+')[:80] + 0.75
                plt.plot(np.arange(2020, 2100), ssp5_shown_conv, 'maroon')
                print('vgl ssp5 with conv: ' + str(meandifference(ssp5_shown_conv, ssp5sealevel)))
                print('rmse ssp5 with conv: ' + str(rmse(ssp5_shown_conv, ssp5sealevel)))
            plt.fill_between(np.arange(2020, 2110), ssp3sealevellow, ssp3sealevelhigh, color='g', alpha=.1)
            plt.fill_between(np.arange(2020, 2110), ssp5sealevellow, ssp5sealevelhigh, color='red', alpha=.1)
            plt.title('sea level simulation in SSP scenarios in comparison', fontsize=17)
            plt.ylabel('rise in mm since 1880', fontsize=17)
            plt.xlabel('year',fontsize=17)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(['SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC'], loc='upper left')
            if withconv:
                plt.legend(['SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP3-7.0 Conv', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC',
                            'SSP5-8.5 Conv'], loc='upper left')