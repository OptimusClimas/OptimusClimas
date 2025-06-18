# -*- coding: utf-8 -*-
# Main script for GUI, start this script to use GUI

# importing libraries for displaying UI
import os
import time
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget
import threading
# importing libraries and modules needed for the simulation
import numpy as np
from climatesimulationAI.other import visuals
from climatesimulationAI import simulation
import matplotlib.pyplot as plt
import datetime


def displayheatmap():
    # Displays heatmap using vis from visuals.py of absolute temperature
    # in the year given by the Slider.
    # Differentiates with try ... except between the cases
    # sea level was simulated (adds an extra dimension to the output list)
    # and was not simulated.
    print('new heatmap in progress ...')
    try:
        visuals.visualizegridtemperature(predf[0][:, :outputsize], last=False, first=False,
                                         i=uiOutput.Slider_year.sliderPosition(),
                                         fig=fig, nominmax=True)
    except:
        visuals.visualizegridtemperature(predf[:, :outputsize], last=False, first=False,
                                         i=uiOutput.Slider_year.sliderPosition(),
                                         fig=fig, nominmax=True)
    # updates the year label using the Slider Position
    uiOutput.label_selyear.setText(str(2014 + uiOutput.Slider_year.sliderPosition()))
    # updates label used as variable to "a" for absolute temperature heatmap is displayed
    uiOutput.label_valheatmap.setText("a")


def displayheatmap_diff():
    # displays heatmap with calculated difference in temperature between selected year and 2014
    # with using presaved heatmapdiff.
    differ = True
    print('new heatmap (diff) in progress ...')
    threaddiffheatmap = threaddiffmap(3, "thread-diffmap", 3)
    threaddiffheatmap.daemon = True
    threaddiffheatmap.start()
    time.sleep(20)
    # updates label used as variable to "d" for difference temperature heatmap is displayed
    print('plotting new diffmap')
    uiOutput.label_valheatmap.setText("d")
    uiOutput.label_empty.setPixmap(QPixmap(u"heatmapdiff.jpg"))


class Ui_MainWindow(object):
    # Class for the UI Input Window
    # initiates and displays GUI using PyQT

    def closeEvent(self):
        sys.exit(0)

    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        # Creates QObjects and sets Name, Position and Geometry
        MainWindow.resize(1015, 715)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        # init label_start for "enter Scenario to Start"
        self.label_start = QLabel(self.centralwidget)
        self.label_start.setObjectName(u"label_start")
        self.label_start.setGeometry(QRect(190, 150, 411, 21))
        # Creates new Font
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.label_start.setFont(font)
        # init label_co2  for "CO2"
        self.label_co2 = QLabel(self.centralwidget)
        self.label_co2.setObjectName(u"label_co2")
        self.label_co2.setGeometry(QRect(190, 200, 71, 21))
        self.label_co2.setFont(font)
        # init label_ch4 for "CH4"
        self.label_ch4 = QLabel(self.centralwidget)
        self.label_ch4.setObjectName(u"label_ch4")
        self.label_ch4.setGeometry(QRect(190, 250, 71, 21))
        self.label_ch4.setFont(font)
        # init label_bc for "BC"
        self.label_bc = QLabel(self.centralwidget)
        self.label_bc.setObjectName(u"label_bc")
        self.label_bc.setGeometry(QRect(200, 300, 71, 21))
        self.label_bc.setFont(font)
        # init label_so2 for "SO2"
        self.label_so2 = QLabel(self.centralwidget)
        self.label_so2.setObjectName(u"label_so2")
        self.label_so2.setGeometry(QRect(190, 350, 71, 21))
        self.label_so2.setFont(font)
        # init label_oc for "OC"
        self.label_oc = QLabel(self.centralwidget)
        self.label_oc.setObjectName(u"label_oc")
        self.label_oc.setGeometry(QRect(200, 400, 71, 21))
        self.label_oc.setFont(font)
        # init lineEdit_co2 for input for the co2 emissions increase or decrease
        self.lineEdit_co2 = QLineEdit(self.centralwidget)
        self.lineEdit_co2.setObjectName(u"lineEdit_co2")
        self.lineEdit_co2.setGeometry(QRect(270, 195, 113, 36))
        # init lineEdit_ch4 for input for the co2 emissions increase or decrease
        self.lineEdit_ch4 = QLineEdit(self.centralwidget)
        self.lineEdit_ch4.setObjectName(u"lineEdit_ch4")
        self.lineEdit_ch4.setGeometry(QRect(270, 245, 113, 36))
        # init lineEdit_bc for input for the co2 emissions increase or decrease
        self.lineEdit_bc = QLineEdit(self.centralwidget)
        self.lineEdit_bc.setObjectName(u"lineEdit_bc")
        self.lineEdit_bc.setGeometry(QRect(270, 295, 113, 36))
        # init lineEdit_so2 for input for the co2 emissions increase or decrease
        self.lineEdit_so2 = QLineEdit(self.centralwidget)
        self.lineEdit_so2.setObjectName(u"lineEdit_so2")
        self.lineEdit_so2.setGeometry(QRect(270, 345, 113, 36))
        # init lineEdit_oc for input for the co2 emissions increase or decrease
        self.lineEdit_oc = QLineEdit(self.centralwidget)
        self.lineEdit_oc.setObjectName(u"lineEdit_oc")
        self.lineEdit_oc.setGeometry(QRect(270, 395, 113, 36))
        # init checkBox_tippingpoints for "use tipping points for simulation"
        self.checkBox_tippingpoints = QCheckBox(self.centralwidget)
        self.checkBox_tippingpoints.setObjectName(u"checkBox_tippingpoints")
        self.checkBox_tippingpoints.setGeometry(QRect(540, 200, 401, 31))
        self.checkBox_tippingpoints.setFont(font)
        # init new font
        font1 = QFont()
        font1.setPointSize(14)
        # init checkBox_permafrost for "use permafrost"
        self.checkBox_permafrost = QCheckBox(self.centralwidget)
        self.checkBox_permafrost.setObjectName(u"checkBox_permafrost")
        self.checkBox_permafrost.setGeometry(QRect(560, 225, 401, 31))
        self.checkBox_permafrost.setFont(font1)
        font2 = QFont()
        font2.setPointSize(12)
        # init checkBox_anaerobe for  "complete anaerobe conditions"
        self.checkBox_anaerobe = QRadioButton(self.centralwidget)
        self.checkBox_anaerobe.setObjectName(u"checkBox_anaerobe")
        self.checkBox_anaerobe.setGeometry(QRect(580, 250, 401, 31))
        self.checkBox_anaerobe.setFont(font2)
        # init checkBox_aeroebe for "complete aerobe conditions"
        self.checkBox_aeroebe = QRadioButton(self.centralwidget)
        self.checkBox_aeroebe.setObjectName(u"checkBox_aeroebe")
        self.checkBox_aeroebe.setGeometry(QRect(580, 270, 401, 31))
        self.checkBox_aeroebe.setFont(font2)
        # init checkbox_partlyaerobe for "partly aerobe conditions"
        self.checkbox_partlyaerobe = QRadioButton(self.centralwidget)
        self.checkbox_partlyaerobe.setObjectName(u"checkbox_partlyaerobe")
        self.checkbox_partlyaerobe.setGeometry(QRect(580, 290, 401, 31))
        self.checkbox_partlyaerobe.setFont(font2)
        # init lineEdit_partanerobe for the input of the percentage of the area assumed anaerobe conditions
        self.lineEdit_partanerobe = QLineEdit(self.centralwidget)
        self.lineEdit_partanerobe.setObjectName(u"lineEdit_partanerobe")
        self.lineEdit_partanerobe.setGeometry(QRect(820, 295, 41, 21))
        # init checkBox_amazonas for "consider amazonas rainforest"
        self.checkBox_amazonas = QCheckBox(self.centralwidget)
        self.checkBox_amazonas.setObjectName(u"checkBox_amazonas")
        self.checkBox_amazonas.setGeometry(QRect(560, 310, 401, 31))
        self.checkBox_amazonas.setFont(font1)
        # init CheckBox_wais for "consider west antarctic ice shield"
        self.CheckBox_wais = QCheckBox(self.centralwidget)
        self.CheckBox_wais.setObjectName(u"CheckBox_wais")
        self.CheckBox_wais.setGeometry(QRect(560, 340, 401, 31))
        self.CheckBox_wais.setFont(font1)
        # init checkBox_sealevel for "predict also global mean sea level"
        self.checkBox_sealevel = QCheckBox(self.centralwidget)
        self.checkBox_sealevel.setObjectName(u"checkBox_sealevel")
        self.checkBox_sealevel.setGeometry(QRect(540, 380, 461, 31))
        self.checkBox_sealevel.setFont(font)
        # init label for "use SSP-Scenarios"
        self.checkBox_SSP = QLabel(self.centralwidget)
        self.checkBox_SSP.setObjectName(u"checkBox_SSP")
        self.checkBox_SSP.setGeometry(QRect(540, 420, 401, 31))
        self.checkBox_SSP.setFont(font)
        # init buttongroup for ssp scenarios
        ssp_group = QButtonGroup(self.centralwidget)
        # init checkBox_ssp2 for "SSP 2-4.5"
        self.checkBox_ssp2 = QRadioButton(self.centralwidget)
        self.checkBox_ssp2.setObjectName(u"checkBox_ssp2")
        self.checkBox_ssp2.setGeometry(QRect(560, 510, 401, 31))
        self.checkBox_ssp2.setFont(font1)
        ssp_group.addButton(self.checkBox_ssp2)
        # init checkBox_ssp1 for "SSP 1-2.6"
        self.checkBox_ssp1 = QRadioButton(self.centralwidget)
        self.checkBox_ssp1.setObjectName(u"checkBox_ssp1")
        self.checkBox_ssp1.setGeometry(QRect(560, 480, 401, 31))
        self.checkBox_ssp1.setFont(font1)
        ssp_group.addButton(self.checkBox_ssp1)
        # init checkBox_ssp19 for "SSP 1-1.9"
        self.checkBox_ssp19 = QRadioButton(self.centralwidget)
        self.checkBox_ssp19.setObjectName(u"checkBox_ssp19")
        self.checkBox_ssp19.setGeometry(QRect(560, 450, 401, 31))
        self.checkBox_ssp19.setFont(font1)
        ssp_group.addButton(self.checkBox_ssp19)
        # init checkBox_ssp3 for "SSP 3-7.0"
        self.checkBox_ssp3 = QRadioButton(self.centralwidget)
        self.checkBox_ssp3.setObjectName(u"checkBox_ssp3")
        self.checkBox_ssp3.setGeometry(QRect(560, 540, 401, 31))
        self.checkBox_ssp3.setFont(font1)
        ssp_group.addButton(self.checkBox_ssp3)
        # init checkBox_ssp5 for "SSP 5-8.5"
        self.checkBox_ssp5 = QRadioButton(self.centralwidget)
        self.checkBox_ssp5.setObjectName(u"checkBox_ssp5")
        self.checkBox_ssp5.setGeometry(QRect(560, 570, 401, 31))
        self.checkBox_ssp5.setFont(font1)
        ssp_group.addButton(self.checkBox_ssp5)
        # init checkBox_ownscenario for "own scenario"
        self.checkBox_ownscenario = QRadioButton(self.centralwidget)
        self.checkBox_ownscenario.setObjectName(u"checkBox_ownscenario")
        self.checkBox_ownscenario.setGeometry(QRect(560, 600, 401, 31))
        self.checkBox_ownscenario.setFont(font1)
        ssp_group.addButton(self.checkBox_ssp5)
        font2 = QFont()
        font2.setPointSize(15)
        # init button_start for "start simulation"
        self.button_start = QPushButton(self.centralwidget)
        self.button_start.setObjectName(u"button_start")
        self.button_start.setGeometry(QRect(750, 560, 161, 71))
        self.button_start.setFont(font2)
        self.button_start.clicked.connect(startsim)
        font3 = QFont()
        font3.setPointSize(13)
        font3.setBold(False)
        # init label_info for information on the input units
        self.label_info = QLabel(self.centralwidget)
        self.label_info.setObjectName(u"label_info")
        self.label_info.setGeometry(QRect(120, 550, 331, 101))
        self.label_info.setFont(font3)
        self.label_info.setWordWrap(True)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # initiates UI elements with retranslateUI

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi
    def retranslateUi(self, MainWindow):
        # translates text for gui elements
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Optimus Klimas", None))
        self.label_start.setText(QCoreApplication.translate("MainWindow", u"Enter Scenario to start:", None))
        self.label_co2.setText(QCoreApplication.translate("MainWindow", u"CO2*:", None))
        self.label_ch4.setText(QCoreApplication.translate("MainWindow", u"CH4*:", None))
        self.label_bc.setText(QCoreApplication.translate("MainWindow", u" BC*:", None))
        self.label_so2.setText(QCoreApplication.translate("MainWindow", u"SO2*:", None))
        self.label_oc.setText(QCoreApplication.translate("MainWindow", u"OC*:", None))
        self.checkBox_tippingpoints.setText(
            QCoreApplication.translate("MainWindow", u"use tipping points for simulation", None))
        self.checkBox_permafrost.setText(QCoreApplication.translate("MainWindow", u"use permafrost", None))
        self.checkBox_anaerobe.setText(QCoreApplication.translate("MainWindow", u"complete anaerobe conditions", None))
        self.checkBox_aeroebe.setText(QCoreApplication.translate("MainWindow", u"complete aerobe conditions", None))
        self.checkbox_partlyaerobe.setText(
            QCoreApplication.translate("MainWindow", u"partly aerobe conditions, in %:", None))
        self.checkBox_amazonas.setText(QCoreApplication.translate("MainWindow", u"consider amazonas rainforest", None))
        self.CheckBox_wais.setText(
            QCoreApplication.translate("MainWindow", u"consider west antarctic ice shield", None))
        self.checkBox_sealevel.setText(
            QCoreApplication.translate("MainWindow", u"predict also global mean sea level", None))
        self.checkBox_SSP.setText(QCoreApplication.translate("MainWindow", u"use SSP-Scenarios", None))
        self.checkBox_ssp2.setText(QCoreApplication.translate("MainWindow", u"SSP 2-4.5", None))
        self.checkBox_ssp1.setText(QCoreApplication.translate("MainWindow", u"SSP 1-2.6", None))
        self.checkBox_ssp19.setText(QCoreApplication.translate("MainWindow", u"SSP 1-1.9", None))
        self.checkBox_ssp3.setText(QCoreApplication.translate("MainWindow", u"SSP 3-7.0", None))
        self.checkBox_ssp5.setText(QCoreApplication.translate("MainWindow", u"SSP 5-8.5", None))
        self.button_start.setText(QCoreApplication.translate("MainWindow", u"start simulation", None))
        self.checkBox_ownscenario.setText(QCoreApplication.translate("MainWindow", u"own scenario", None))
        self.label_info.setText(QCoreApplication.translate("MainWindow",
                                                           u"* All Values are used as relative changes in terms of emissions in the respective greenhousegas until 2100 and are therefore to be given in percentage values.",
                                                           None))


# defines stylesheet for the Main Windows
stylesheet = """
        QMainWindow {
            border-image: url("KlimaUi/climatesimulationAI/designUIoptimusklimas.png") 0 0 0 0 stretch stretch; 
            background-position: center;
            background-attachment: fixed;
        }
    """

def show_info_messagebox():
   msg = QMessageBox()
   msg.setIcon(QMessageBox.Icon.Information)
   msg.setText("All processes of the last simulation will finish soon. Please wait for about 10 seconds.")
   msg.setWindowTitle("Restarting simulation")
   msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
   retval = msg.exec()

def restart():
    # restarts whole application
    try:
        threadheatmap.stop()
    except:
        pass
    MainOutput.close()
    show_info_messagebox()
    time.sleep(10)
    MainInput.show()


class myThreadheatmap(threading.Thread):
    # thread for regularly updating the heatmap
    def __init__(self, threadID, name, counter,*args, **kwargs):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        super().__init__(*args, **kwargs)
        self._stopper = threading.Event()


    def run(self):
        # updates the heatmap in 5 seconds intervals if the difference option is not displayed
        # (checks via label_valheatmap equals "d")
        while True:
            if self.stopped():
                return
            time.sleep(5)
            if str(uiOutput.label_valheatmap.text()) == "d":
                pass
            else:
                uiOutput.label_empty.setPixmap(QPixmap(u"heatmap.jpg"))

    def stop(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.is_set()


class threaddiffmap(threading.Thread):
    # thread for saving a difference heatmap (which is on other levels not always possible)
    # not a continues loop!
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        #  init variables and objects for generating new heatmaps
        print('save diffmap')
        u = 55296
        diff = np.ones(u)
        seause = False
        # calculate difference heatmap (differentiates if sea level was predicted or not)
        for i in range(u):
            if seause:
                diff[i] = predf[0][uiOutput.Slider_year.sliderPosition(), i] - predf[0][0, i]
            else:
                try:
                    diff[i] = predf[uiOutput.Slider_year.sliderPosition(), i] - predf[0, i]
                except:
                    diff[i] = predf[0][uiOutput.Slider_year.sliderPosition(), i] - predf[0][0, i]
                    seause = True
        # generate, safe and display image file using vis from visuals.py
        try:
            visuals.visualizegridtemperature(np.expand_dims(diff, 0), first=True, last=False, nc=True,
                                             i=uiOutput.Slider_year.sliderPosition(),
                                             fig=fig, min=0, max=11,
                                             diff=False, nominmax=True, savediff=True)
        except:
            visuals.visualizegridtemperature(np.expand_dims(diff, 0), first=True, last=False, nc=True,
                                             i=uiOutput.Slider_year.sliderPosition(), min=0, max=11,
                                             fig=fig,
                                             diff=False, nominmax=False, savediff=True)
        print('generated new diff map')
        # set label (functioning as variable) to "d" for difference temperature heatmap
        uiOutput.label_valheatmap.setText("d")

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

class myThreadSim(threading.Thread):
    # thread for handling the simulation
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        # print current time for later check of the simulation duration
        print(datetime.datetime.now())
        # init variables for the simulation
        global differ
        global nc
        differ = False
        global predf
        modelname = "FrederikeSSTGADFGRIBhist108.h5"
        modelnamesea = 'kalaSST104.h5'
        newsea = True

        # init emissions for ssp scenarios
        ghgchangesssp19 = [-98, -98, -60, -80, -85, -58]
        ghgchangesssp126 = [-90, -90, -60, -77, -88, -58]
        ghgchangesssp2 = [-75, -75, -25, -55, -60, -55]
        ghgchangesssp3 = [100, 100, 100, -1, -10, -1]
        ghgchangesssp5 = [200, 200, 50, -60, -60, -43]

        # init variables for aerobe or anaerobe conditions (for permafrost simulation)
        if aerobe:
            useanaerobe = False
        elif anaerobe:
            useanaerobe = True
        else:
            useanaerobe = True
        nc = False
        exceededmodelrange = False
        # check if a ssp scenario was selected and if start simulation of the concerning ssp scenario
        # print if it is a negative ssp scenario and select different model
        # !!! model for negative emission scenarios is not tested and evaluated as much,
        # Optimus Climas is mainly optimised on positive emission scenarios !!!

        stopssim = False
        if ssp19:
            print('negative emission scenario')
            modelname = "FrederikeSSTGADFGRIBhist101.h5"
            nc = True
            predf = simulation.pred(ghgchanges=ghgchangesssp19, start=2014, end=2114, modelname=modelname,
                                    withtippingpoints=tippingpoints, predsea=sea, modelnamesea=modelnamesea,
                                    anaerobe=useanaerobe, rainforestused=rainforest,
                                    partly_anaerobe=partlyanaerobe, partanaeorbe=partanaerobe, new=True,
                                    with_oldmodel=False,
                                    awi=False, wais=wais)
        elif ssp12:
            print('negative emission scenario')
            modelname = "FrederikeSSTGADFGRIBhist101.h5"
            nc = True
            predf = simulation.pred(ghgchanges=ghgchangesssp126, start=2014, end=2114, modelname=modelname,
                                    withtippingpoints=tippingpoints, predsea=sea, modelnamesea=modelnamesea,
                                    anaerobe=useanaerobe, rainforestused=rainforest,
                                    partly_anaerobe=partlyanaerobe, partanaeorbe=partanaerobe, new=True,
                                    with_oldmodel=False,
                                    wais=wais)
        elif ssp2:
            print('negative emission scenario')
            modelname = "FrederikeSSTGADFGRIBhist101.h5"
            nc = True
            predf = simulation.pred(ghgchanges=ghgchangesssp2, start=2014, end=2114, modelname=modelname,
                                    withtippingpoints=tippingpoints, predsea=sea, modelnamesea=modelnamesea,
                                    anaerobe=useanaerobe, rainforestused=rainforest,
                                    partly_anaerobe=partlyanaerobe, partanaeorbe=partanaerobe, new=True,
                                    with_oldmodel=False,
                                    wais=wais)
        elif ssp3:
            predf = simulation.pred(ghgchanges=ghgchangesssp3, start=2014, end=2114, modelname=modelname,
                                    withtippingpoints=tippingpoints, predsea=sea, modelnamesea=modelnamesea,
                                    anaerobe=useanaerobe, rainforestused=rainforest,
                                    partly_anaerobe=partlyanaerobe, partanaeorbe=partanaerobe, awi=False,
                                    wais=wais)
        elif ssp5:
            predf = simulation.pred(ghgchanges=ghgchangesssp5, start=2014, end=2114, modelname=modelname,
                                    withtippingpoints=tippingpoints, predsea=sea, modelnamesea=modelnamesea,
                                    anaerobe=useanaerobe, rainforestused=rainforest,
                                    partly_anaerobe=partlyanaerobe, partanaeorbe=partanaerobe, awi=False,
                                    wais=wais)
        else:
            # prepare simulation with manually entered scenario #
            # init variables for manual simulation
            ghgs = [ghg, co2, ch4, bc, so2, oc]
            newuse = False
            oldmodeluse = True
            exceededmodelrange = False
            # check if the model emission range was exceeded
            for i in range(len(ghgs)):
                if ghgs[i] > 450:
                    exceededmodelrange = True
                if ghgs[i] < -90:
                    exceededmodelrange = True
            if not exceededmodelrange:
                # select other model if a negative emission scenario was chosen
                # !!! model for negative emission scenarios is not tested and evaluated as much,
                # Optimus Climas is mainly optimised on positive emission scenarios !!!
                for i in range(2):
                    if ghgs[i] < 0:
                        print('negative emission scenario')
                        modelname = "FrederikeSSTGADFGRIBhist101.h5"
                        newuse = True
                        oldmodeluse = False
                        nc = True
                        if tippingpoints:
                            print('simulation of negative emission scenario with tipping points is not possible!')
                            stopssim = True
                if not stopssim:
                    # start prepared simulation with manually entered scenario
                    predf = simulation.pred(ghgchanges=ghgs, start=2014, end=2114, modelname=modelname,
                                            withtippingpoints=tippingpoints, predsea=sea, modelnamesea=modelnamesea,
                                            anaerobe=useanaerobe, rainforestused=rainforest,
                                            partly_anaerobe=partlyanaerobe, partanaeorbe=partanaerobe, new=newuse,
                                            with_oldmodel=oldmodeluse, awi=False, wais=wais, newsea=newsea)
            else:
                # print error message if model emission range was exceeded
                print('The possible range of changes from -90 % to 450 % was exceeded -> No Simulation possible!')
        if not exceededmodelrange and (not stopssim):
            # init variables for displaying results of the simulation
            years = np.arange(2014, 2114)
            # back-up of the simulation results via file saving as and .npy
            np.save('KlimaUi/climatesimulationAI/other/predf.npy', predf[0])
            sim = np.ones((2, 100))
            global outputsize
            # change outputsize if model with "nc" grid is used
            if nc:
                print('nc')
                outputsize = 55296
            else:
                # change outputsize (to "nc" grid) if certain model is used
                if modelname == "FrederikeSSTGADFGRIBhist108.h5":
                    outputsize = 55296
                    nc = True
                else:
                    outputsize = 115200

            if sea:
                # split simulation output onto different list if sea level was simulated
                sim[0] = predf[0][:, outputsize]
                sim[1] = predf[1][:]
                # display simulated global sea level rise onto graph
                Ui_Output.draw(uiOutput, years, sim[1], "SeaLevel")
            else:
                sim[0] = predf[:, outputsize]

            # display simulated global temperature rise onto graph
            Ui_Output.draw(uiOutput, years, sim[0], "Temperatur")
            print(datetime.datetime.now())

            # generate heatmap image file using vis from visuals.py (input differentiating whether sea level was
            # simulated)

            print(uiOutput.Slider_year.sliderPosition())
            if sea:
                print(nc)
                visuals.visualizegridtemperature(predf[0][:, :outputsize], last=False,
                                                 i=uiOutput.Slider_year.sliderPosition(),
                                                 fig=fig, diff=False, nc=nc, nominmax=True, awi=False)
            else:
                visuals.visualizegridtemperature(predf[:, :outputsize], last=False,
                                                 i=uiOutput.Slider_year.sliderPosition(),
                                                 fig=fig, diff=False, nc=nc, nominmax=True, awi=False)
            # display heatmap image file in label
            uiOutput.label_empty.setPixmap(QPixmap(u"heatmap.jpg"))
            # start threads for displaying heatmaps
            threaddiffheatmap = threaddiffmap(3, "thread-diffmap", 3)
            threaddiffheatmap.daemon = True
            threaddiffheatmap.start()
            global threadheatmap
            threadheatmap = myThreadheatmap(2, "Thread-2", 2)
            threadheatmap.daemon = True
            threadheatmap.start()
            uiOutput.button_simotherscen.setVisible(True)

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


class Ui_Output(object):
    # Class for Output GUI
    # Initiates GUI elements with name, size and position

    def closeEvent(self):
        sys.exit(0)

    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1144, 735)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        font = QFont()
        font.setPointSize(19)
        font.setBold(True)
        font2 = QFont()
        font2.setPointSize(12)
        font2.setBold(True)
        # init label_scenario for displaying the used scenario for the current simulation
        self.label_scenario = QLabel(self.centralwidget)
        self.label_scenario.setObjectName(u"label_scenario")
        self.label_scenario.setGeometry(QRect(180, 130, 941, 51))
        self.label_scenario.setFont(font)
        self.label_scenario.setStyleSheet("QLabel { color: 'black'; }")
        # init label_globalres for "Global mean results"
        self.label_globalres = QLabel(self.centralwidget)
        self.label_globalres.setObjectName(u"label_globalres")
        self.label_globalres.setGeometry(QRect(140, 180, 291, 51))
        self.label_globalres.setFont(font)
        self.label_globalres.setStyleSheet("QLabel { color: 'black'; }")
        # init button_simotherscen for start simulate another scenario
        self.button_simotherscen = QPushButton(self.centralwidget)
        self.button_simotherscen.setObjectName(u"button_simotherscen")
        self.button_simotherscen.setGeometry(QRect(570, 650, 381, 71))
        self.button_simotherscen.setFont(font)
        self.button_simotherscen.setVisible(False)
        self.button_simotherscen.clicked.connect(restart)
        # init button_absolutetemp for change to absolute temperature map
        self.button_absolutetemp = QPushButton(self.centralwidget)  # 710, 210, 221, 16
        self.button_absolutetemp.setObjectName(u"button_absolutetemp")
        self.button_absolutetemp.setGeometry(QRect(550, 180, 251, 30))
        self.button_absolutetemp.setFont(font2)
        self.button_absolutetemp.clicked.connect(displayheatmap)
        # init button_differencetemp for change to difference temperature map
        self.button_differencetemp = QPushButton(self.centralwidget)
        self.button_differencetemp.setObjectName(u"button_differencetemp")
        self.button_differencetemp.setGeometry(QRect(800, 180, 221, 30))
        self.button_differencetemp.setFont(font2)
        self.button_differencetemp.clicked.connect(displayheatmap_diff)
        # init Slider_year for input for the year of the displayed heatmap
        self.Slider_year = QSlider(self.centralwidget)
        self.Slider_year.setObjectName(u"Slider_year")
        self.Slider_year.setGeometry(QRect(580, 580, 351, 31))
        self.Slider_year.setMaximum(100)
        self.Slider_year.setSliderPosition(86)
        self.Slider_year.setOrientation(Qt.Horizontal)
        self.Slider_year.sliderMoved.connect(displayheatmap)
        # init label_selyear for the selected year to display the heatmap
        self.label_selyear = QLabel(self.centralwidget)
        self.label_selyear.setStyleSheet("QLabel { color: 'black'; }")
        self.label_selyear.setObjectName(u"label_selyear")
        self.label_selyear.setGeometry(QRect(740, 620, 56, 15))
        self.label_selyear.setFont(font2)
        # init label_2014 for 2014
        self.label_2014 = QLabel(self.centralwidget)
        self.label_2014.setObjectName(u"label_2014")
        self.label_2014.setStyleSheet("QLabel { color: 'black'; }")
        self.label_2014.setGeometry(QRect(580, 610, 56, 15))
        self.label_2014.setFont(font2)
        # init label_2114 for "2114"
        self.label_2114 = QLabel(self.centralwidget)
        self.label_2114.setObjectName(u"label_2114")
        self.label_2114.setStyleSheet("QLabel { color: 'black'; }")
        self.label_2114.setGeometry(QRect(910, 610, 56, 15))
        self.label_2114.setFont(font2)
        # init label_empty for " "
        self.label_empty = QLabel(self.centralwidget)
        self.label_empty.setObjectName(u"label_empty")
        self.label_empty.setGeometry(QRect(560, 230, 441, 341))
        self.label_empty.setPixmap(QPixmap(u"simulating.jpg"))
        self.label_empty.setScaledContents(True)
        # init label_tempC for temperatures in degree Celsius
        self.label_tempC = QLabel(self.centralwidget)
        self.label_tempC.setObjectName(u"label_tempC")
        self.label_tempC.setGeometry(QRect(710, 210, 221, 16))
        self.label_tempC.setFont(font2)
        self.label_tempC.setStyleSheet("QLabel { color: 'black'; }")
        # init graphicsView_temp for displaying plots of global mean temperature
        scene = QGraphicsScene()
        scene.addText("Hello, world!")
        self.graphicsView_temp = PlotWidget(MainOutput)
        self.graphicsView_temp.setObjectName(u"graphicsView_temp")
        self.graphicsView_temp.setGeometry(QtCore.QRect(140, 240, 381, 211))  # 260
        self.graphicsView_temp.setBackground('w')
        self.graphicsView_temp.setLabel('left', 'temperature in ° C')
        self.graphicsView_temp.setLabel('bottom', 'years')
        self.graphicsView_temp.show()
        # init label_globalmeantemp for "global mean temperature"
        self.label_globalmeantemp = QLabel(self.centralwidget)
        self.label_globalmeantemp.setObjectName(u"label_globalmeantemp")
        self.label_globalmeantemp.setStyleSheet("QLabel { color: 'black'; }")
        self.label_globalmeantemp.setGeometry(QRect(260, 220, 220, 16))
        self.label_globalmeantemp.setFont(font2)
        # init graphicsView_sea for displaying graphs of global mean sea level
        self.graphicsView_sea = PlotWidget(MainOutput)
        self.graphicsView_sea.setObjectName(u"graphicsView_sea")
        self.graphicsView_sea.setGeometry(QtCore.QRect(140, 480, 381, 221))
        self.graphicsView_sea.setBackground('w')
        self.graphicsView_sea.setLabel('left', 'sea level rise in mm since 1880')
        self.graphicsView_sea.setLabel('bottom', 'years')
        self.graphicsView_sea.show()
        # init label_globalmeansea for "globale mean sea level"
        self.label_globalmeansea = QLabel(self.centralwidget)
        self.label_globalmeansea.setObjectName(u"label_globalmeansea")
        self.label_globalmeansea.setStyleSheet("QLabel { color: 'black'; }")
        self.label_globalmeansea.setGeometry(QRect(260, 460, 280, 16))
        self.label_globalmeansea.setFont(font2)
        # init label_valheatmap used as variable for absolute ("a") or difference ("d") heatmap is displayed
        self.label_valheatmap = QLabel(self.centralwidget)
        self.label_valheatmap.setObjectName(u"label_valheatmap")
        self.label_valheatmap.setGeometry(QRect(10260, 480, 280, 16))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi
    # retranslates GUI elements with text
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_scenario.setText(
            QCoreApplication.translate("MainWindow", u"Scenario: CO2: 100 CH4: 100 BC: -1 SO2: -10 OC:  -1", None))
        self.label_globalres.setText(QCoreApplication.translate("MainWindow", u"Global mean results:", None))
        self.button_simotherscen.setText(QCoreApplication.translate("MainWindow", u"simulate another scenario", None))
        self.button_absolutetemp.setText(QCoreApplication.translate("MainWindow", u"absolute temperature map", None))
        self.button_differencetemp.setText(QCoreApplication.translate("MainWindow", u"differencemap to 2014", None))
        self.label_selyear.setText(QCoreApplication.translate("MainWindow", u"2100", None))
        self.label_2014.setText(QCoreApplication.translate("MainWindow", u"2014", None))
        self.label_2114.setText(QCoreApplication.translate("MainWindow", u"2114", None))
        self.label_empty.setText("")
        self.label_tempC.setText(QCoreApplication.translate("MainWindow", u"Temperatures in \u00b0 C", None))
        self.label_globalmeantemp.setText(QCoreApplication.translate("MainWindow", u"globale mean temperature", None))
        self.label_globalmeansea.setText(QCoreApplication.translate("MainWindow", u"globale mean sea level", None))
        self.label_valheatmap.setText("")

    def draw(self, x, y, plt):
        # drawing for plots in graphic views,
        # inputs: x- and y-values, "SeaLevel" (if sea level is supposed to be displayed) or
        # "Temperatur" (if temperature is supposed to be displayed)
        if plt == "SeaLevel":
            try:
                self.graphicsView_sea.plot(x, y, pen=(0, 3))
            except Exception as e:
                print('error while plotting sea level')
                print(e)
        elif plt == "Temperatur":
            try:
                self.graphicsView_temp.plot(x, y, pen=(0, 3))
            except Exception as e:
                print('error while plotting temperature')
                print(e)

    def clear(self):
        # clear plots in graphicsViews
        self.graphicsView_temp.clear()
        self.graphicsView_sea.clear()


stylesheet = """
        QMainWindow {
            border-image: url("KlimaUi/climatesimulationAI/designUIoptimusklimas.png"); 
        }
    """


def startsim():
    # function to start simulation - linked to start simulation button
    # init emissions variables from manual input from gui elements
    # check that only integers can be entered
    global sea
    global ssp19  # if ssp1-1.9 scenario is to be used
    ssp19 = ui.checkBox_ssp19.isChecked()
    global ssp12  # if ssp1-2.6 scenario is to be used
    ssp12 = ui.checkBox_ssp1.isChecked()
    global ssp2  # if ssp2-4.5 scenario is to be used
    ssp2 = ui.checkBox_ssp2.isChecked()
    global ssp3  # if ssp3-7.0 scenario is to be used
    ssp3 = ui.checkBox_ssp3.isChecked()
    global ssp5  # if ssp5-8.5 scenario is to be used
    ssp5 = ui.checkBox_ssp5.isChecked()
    if not (ssp19 or ssp12 or ssp2 or ssp3 or ssp5):
        try:
            global co2
            co2 = int(ui.lineEdit_co2.text())
            global ch4
            ch4 = int(ui.lineEdit_ch4.text())
            global bc
            bc = int(ui.lineEdit_bc.text())
            global so2
            so2 = int(ui.lineEdit_so2.text())
            global oc
            oc = int(ui.lineEdit_oc.text())
            global ghg
            ghg = int(ui.lineEdit_co2.text())
        except Exception as e:
            print('Wrong format of input has been entered, emissions have to be integers or floats!')

    # init values from checkboxes (which aspects are to be considered for the simulation)
    global tippingpoints  # if tipping points are supposed to be considered
    tippingpoints = ui.checkBox_tippingpoints.isChecked()
    global permafrost  # if the tipping point collapse of the boreal permafrost is supposed to be considered
    permafrost = ui.checkBox_permafrost.isChecked()
    global wais  # if the tipping point collapse of the west antarctic ice shield is supposed to be considered
    wais = ui.CheckBox_wais.isChecked()
    global aerobe  # if aerobe conditions are assumed for a possible collapse of the boreal permafrost
    aerobe = ui.checkBox_anaerobe.isChecked()
    global anaerobe  # if anaerobe conditions are assumed for a possible collapse of the boreal permafrost
    anaerobe = ui.checkBox_aeroebe.isChecked()
    global partlyanaerobe  # if partly anaerobe conditions are assumed for a possible collapse of the boreal permafrost
    partlyanaerobe = ui.checkbox_partlyaerobe.isChecked()
    global rainforest  # if the tipping point die-off of the amazonas rainforest is supposed to be considered
    rainforest = ui.checkBox_amazonas.isChecked()
    global partanaerobe  # if values was entered, for which percentage of the area anaerobe conditions are assumed
    try:
        partanaerobe = float(ui.lineEdit_partanerobe.text())
    except:
        partanaerobe = None
    global sea  # if sea level rise is to be simulated
    sea = ui.checkBox_sealevel.isChecked()
    # init ghgs as list
    try:
        ghgs = [co2, co2, ch4, bc, so2, oc]
    except:
        ghgs = [0, 0, 0, 0, 0, 0]
    exceededmodelrange = False
    # check if model emission range was exceeded (manual input)
    # set to false if ssp scenario was used
    for i in range(len(ghgs)):
        if ghgs[i] > 450:
            exceededmodelrange = True
        if ghgs[i] < -90:
            exceededmodelrange = True
        if ssp19:
            exceededmodelrange = False
        if ssp12:
            exceededmodelrange = False
        if ssp2:
            exceededmodelrange = False
        if ssp3:
            exceededmodelrange = False
        if ssp5:
            exceededmodelrange = False
    if not exceededmodelrange:
        MainInput.close()  # close Input Window
        #  appSim = QtWidgets.QApplication(sys.argv)
        # init Output Window
        global MainOutput
        MainOutput = QtWidgets.QMainWindow()
        global uiOutput
        # setup UI for Output Window
        uiOutput = Ui_Output()
        uiOutput.setupUi(MainOutput)
        # init emissions for ssp scenarios
        ghgchangesssp19 = [-100, -100, -60, -80, -85, -58]
        ghgchangesssp126 = [-90, -90, -60, -77, -88, -58]
        ghgchangesssp2 = [-75, -75, -25, -55, -60, -55]
        ghgchangesssp3 = [100, 100, 100, -1, -10, -1]
        ghgchangesssp5 = [200, 200, 50, -60, -60, -43]
        # display emissions in the output window of the concerning ssp scenario if a ssp scenario was chosen
        if ssp19:
            uiOutput.label_scenario.setText(
                'Scenario: CO2:' + str(ghgchangesssp19[1]) + ' CH4: ' + str(ghgchangesssp19[2]) + ' BC: ' + str(
                    ghgchangesssp19[3]) + ' SO2: ' + str(ghgchangesssp19[4]) + ' OC: ' + str(
                    ghgchangesssp19[5]))
        elif ssp12:
            uiOutput.label_scenario.setText(
                'Scenario: CO2:' + str(ghgchangesssp126[1]) + ' CH4: ' + str(ghgchangesssp126[2]) + ' BC: ' + str(
                    ghgchangesssp126[3]) + ' SO2: ' + str(ghgchangesssp126[4]) + ' OC: ' + str(
                    ghgchangesssp126[5]))
        elif ssp2:
            uiOutput.label_scenario.setText(
                'Scenario: CO2:' + str(ghgchangesssp2[1]) + ' CH4: ' + str(ghgchangesssp2[2]) + ' BC: ' + str(
                    ghgchangesssp2[3]) + ' SO2: ' + str(ghgchangesssp2[4]) + ' OC: ' + str(
                    ghgchangesssp2[5]))
        elif ssp3:
            uiOutput.label_scenario.setText(
                'Scenario: CO2:' + str(ghgchangesssp3[1]) + ' CH4: ' + str(ghgchangesssp3[2]) + ' BC: ' + str(
                    ghgchangesssp3[3]) + ' SO2: ' + str(ghgchangesssp3[4]) + ' OC: ' + str(
                    ghgchangesssp3[5]))
        elif ssp5:
            uiOutput.label_scenario.setText(
                'Scenario: CO2:' + str(ghgchangesssp5[1]) + ' CH4: ' + str(ghgchangesssp5[2]) + ' BC: ' + str(
                    ghgchangesssp5[3]) + ' SO2: ' + str(ghgchangesssp5[4]) + ' OC: ' + str(
                    ghgchangesssp5[5]))
        else:
            # display the selected emission scenario (manual input) in the output window
            uiOutput.label_scenario.setText(
                'Scenario: CO2:' + str(co2) + ' CH4: ' + str(ch4) + ' BC: ' + str(bc) + ' SO2: ' + str(
                    so2) + ' OC: ' + str(
                    oc))
        # show Window and execute App
        MainOutput.show()
        # init figure to display results after the simulation
        global fig
        fig = plt.figure(figsize=(16, 35))
        # start thread to handle and start the simulation
        global thread1
        thread1 = myThreadSim(1, "Thread-1", 1)
        thread1.daemon = True
        thread1.start()
        #sys.exit(appSim.exec_())
    else:
        print('The possible range of changes from -90 % to 450 % was exceeded -> No Simulation possible!')


class threadradiogroup(threading.Thread):
    # thread for saving a difference heatmap (which is on other levels not always possible)
    # not a continues loop!
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        while(True):
            if ui.checkBox_permafrost.isChecked() or ui.checkBox_amazonas.isChecked() or ui.checkBox_tippingpoints.isChecked() or ui.CheckBox_wais.isChecked():
                ui.checkBox_ssp2.setDisabled(True)
                ui.checkBox_ssp1.setDisabled(True)
                ui.checkBox_ssp19.setDisabled(True)
            else:
                ui.checkBox_ssp2.setDisabled(False)
                ui.checkBox_ssp1.setDisabled(False)
                ui.checkBox_ssp19.setDisabled(False)
            if ui.checkBox_ssp2.isChecked() or ui.checkBox_ssp1.isChecked() or ui.checkBox_ssp19.isChecked():
                ui.checkBox_aeroebe.setDisabled(True)
                ui.checkBox_anaerobe.setDisabled(True)
                ui.checkbox_partlyaerobe.setDisabled(True)
                ui.checkBox_permafrost.setDisabled(True)
                ui.checkBox_amazonas.setDisabled(True)
                ui.checkBox_tippingpoints.setDisabled(True)
                ui.CheckBox_wais.setDisabled(True)
            if ui.checkBox_ssp3.isChecked() or ui.checkBox_ssp5.isChecked():
                ui.checkBox_aeroebe.setDisabled(False)
                ui.checkBox_anaerobe.setDisabled(False)
                ui.checkbox_partlyaerobe.setDisabled(False)
                ui.checkBox_permafrost.setDisabled(False)
                ui.checkBox_amazonas.setDisabled(False)
                ui.checkBox_tippingpoints.setDisabled(False)
                ui.CheckBox_wais.setDisabled(False)

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

def main():
    app.setStyleSheet(stylesheet)
    ui.setupUi(MainInput)
    # show Window and execute App
    MainInput.show()
    threadradiogroup1 = threadradiogroup(1, "Thread-radiogroup", 1)
    threadradiogroup1.daemon = True
    threadradiogroup1.start()
    sys.exit(app.exec_())

if __name__ == '__main__':
    # init app with UI
    app = QtWidgets.QApplication(sys.argv)
    MainInput = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    main()

