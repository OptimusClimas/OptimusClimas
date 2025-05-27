# module for automated evaluations

# import of needed libraries
import numpy as np
import matplotlib.pyplot as plt

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
                simpredt = np.array(
                    simulation.pred(ghgchanges=ghgchangesssp126, start=2014, end=2114, modelname=modelname,
                                    method_SST=True, method_Conv=False))
                simpred = simpredt[:80, u]
                plt.plot(years, np.array(simpred) - 0.2, 'blue')
                print('vgl ssp126: ' + str(meandifference(simpred, np.load("ssp126temp.npy", mmap_mode="r+"))))
                plt.plot(years, np.load("ssp126temp.npy", mmap_mode="r+") - 0.2, 'blue', linestyle='--')
                simpredt = np.array(
                    simulation.pred(ghgchanges=ghgchangesssp2, start=2014, end=2114, modelname=modelname,
                                    method_SST=True, method_Conv=False))
                simpred = simpredt[:80, u]
                plt.plot(years, np.array(simpred) - 0.2, 'orange')
                print('vgl ssp2: ' + str(meandifference(simpred, np.load("ssp2temp.npy", mmap_mode="r+"))))
                plt.plot(years, np.load("ssp2temp.npy", mmap_mode="r+") - 0.2, 'orange', linestyle='--')
                plt.legend(
                    ['SSP1-1.9 VT', 'SSP1-1.9 IPCC', 'SSP1-2.6 VT', 'SSP1-2.6 IPCC', 'SSP2-4.5 VT', 'SSP2-4.5 IPCC',
                     'SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC'], loc='upper left')
            # predict, plot and calculate deviation to CMIP6 results in positive SSP scenarios
            simpredt = np.array(
                simulation.pred(ghgchanges=ghgchangesssp3, start=2014, end=2114, modelname=modelname,
                                method_SST=True, method_Conv=False))[:80]
            simpred = simpredt[:80, u]
            plt.plot(years, np.array(simpred) - 0.2, 'g')
            plt.plot(years, np.load("all_ssp3Temperatur.npy", mmap_mode="r+")[:80], 'g', linestyle='--')

            print('vgl ssp3: ' + str(meandifference(simpred, np.load("ssp3temp.npy", mmap_mode="r+"))))
            if withconv:
                plt.plot(np.arange(2020, 2100), np.load('predconvtempglobalssp3.npy', mmap_mode='r+')[:80] + 0.75,
                         'olive')
                print('vgl ssp3 with conv: ' + str(
                    meandifference(np.load('predconvtempglobalssp3.npy', mmap_mode='r+')[:80],
                                   np.load("ssp3temp.npy", mmap_mode="r+"))))
            simpredt = np.array(
                simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                method_SST=True, method_Conv=False))
            simpred = simpredt[:80, u]
            plt.plot(years, np.array(simpred) - 0.2, 'r')
            plt.plot(years, np.load("ipccSSP5Temperature1.npy", mmap_mode="r+")[:80], 'r', linestyle='--')

            print('vgl ssp5: ' + str(meandifference(simpred, np.load("ssp5temp.npy", mmap_mode="r+"))))
            if withconv:
                plt.plot(np.arange(2020, 2100), np.load('predconvtempglobalssp5.npy', mmap_mode='r+')[:80] + 0.75,
                         'maroon')
                print('vgl ssp5 with conv: ' + str(
                    meandifference(np.load('predconvtempglobalssp5.npy', mmap_mode='r+')[:80],
                                   np.load("ssp5temp.npy", mmap_mode="r+"))))
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
                plt.legend(['SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC'], loc='upper left')
                if withconv:
                    plt.legend(['SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP3-7.0 Conv', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC',
                                'SSP5-8.5 Conv'], loc='upper left')
            else:
                plt.legend(
                    ['SSP1-1.9 VT', 'SSP1-1.9 IPCC', 'SSP1-2.6 VT', 'SSP1-2.6 IPCC', 'SSP2-4.5 VT', 'SSP2-4.5 IPCC',
                     'SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP3-7.0 IPCC error', 'SSP5-8.5 VT',
                     'SSP5-8.5 IPCC', 'SSP5-8.5 IPCC error'], loc='upper left')
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
                'vgl sea ssp3: ' + str(meandifference(predf[1][6:96], np.load("ssp3sealevel.npy", mmap_mode="r+"))))

            if withconv:
                plt.plot(np.arange(2020, 2100), np.load('predconvseaglobalssp3.npy', mmap_mode='r+')[:80] + 0.75,
                         'olive')
                print('vgl ssp3 with conv: ' + str(
                    meandifference(np.load('predconvglobalssp3.npy', mmap_mode='r+')[:80],
                                   np.load("ssp3sealevel.npy", mmap_mode="r+"))))
            predf = simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                    actfactor=actfactore, predsea=True, modelnamesea=modelnamesea)
            ssp5sealevel = np.load('ssp5sealevel.npy', mmap_mode='r+')
            ssp5sealevellow = np.load('ssp5sealevellow.npy', mmap_mode='r+')
            ssp5sealevelhigh = np.load('ssp5sealevelhigh.npy', mmap_mode='r+')
            plt.plot(np.arange(2020, 2110), predf[1][6:96], 'r')
            plt.plot(np.arange(2020, 2110), ssp5sealevel, 'r', linestyle='--')
            print('vgl sea ssp5: ' + str(
                meandifference(predf[1][6:96], np.load("ssp5sealevel.npy", mmap_mode="r+"))))
            if withconv:
                plt.plot(np.arange(2020, 2100), np.load('predconvseaglobalssp5.npy', mmap_mode='r+')[:80] + 0.75,
                         'maroon')
                print('vgl ssp5 with conv: ' + str(
                    meandifference(np.load('predconvglobalssp5.npy', mmap_mode='r+')[:80],
                                   np.load("ssp5sealevel.npy", mmap_mode="r+"))))
            plt.fill_between(np.arange(2020, 2110), ssp3sealevellow, ssp3sealevelhigh, color='g', alpha=.1)
            plt.fill_between(np.arange(2020, 2110), ssp5sealevellow, ssp5sealevelhigh, color='red', alpha=.1)
            plt.title('sea level simulation in SSP scenarios in comparison', fontsize=17)
            plt.ylabel('rise in mm since 1880', fontsize=17)
            plt.xlabel('year',fontsize=)
            plt.xticks(fontsize=12)17
            plt.yticks(fontsize=12)
            plt.legend(['SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC'], loc='upper left')
            if withconv:
                plt.legend(['SSP3-7.0 VT', 'SSP3-7.0 IPCC', 'SSP3-7.0 Conv', 'SSP5-8.5 VT', 'SSP5-8.5 IPCC',
                            'SSP5-8.5 Conv'], loc='upper left')