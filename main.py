import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from MainGUI import Ui_MainWindow
from progressBar import Ui_progressDialog
from Nodal_plot import Ui_Nodal_plot
from Pressure_plot import Ui_Pressure_plot
from Temperature_plot import Ui_Temperature_plot
from Holdup_plot import Ui_Holdup_plot
from Velocity_plot import Ui_Velocity_plot

import xml.etree.ElementTree as ET
import threading
from GL import *
from ESP_simple_all_in_one import *
from SPSA import *
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
G = 9.81
pi = np.pi
# E1 = 1e-5
# DENW = 997.                 # water density
# VISW = 1e-3                 # water viscosity
psi_to_pa = 1.013e5 / 14.7
Mpa_to_psi = 145.038
Mpa_to_m = 101.97
psi_to_ft = 2.3066587368787
bbl_to_m3 = 0.15897
bpd_to_m3s = bbl_to_m3/24/3600
bpd_to_m3d = bbl_to_m3
m3s_to_bpd = 543439.65056533
sgl_model='zhang_2016'


class ESPNodalGUI(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None): 
        super(ESPNodalGUI, self).__init__(parent)
        # setup mainWindow location
        self.setupGUI()     

        # Initial value
                    
        self.inputs = {"Surface_line_pressure": 2.8*1e6, "Pipe_diameter":0.088,        
                    "Roughness":0,                   "Liquid_viscosity":0.523,
                   "Liquid_relative_density":0.9521,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
                    "WC": 75, 
                    "Reservoir_C":4.880E-17,     "Reservoir_n":1, 
                    "Reservoir_P":1.25e7,                "GOR":6.7,     "GLR":5, 
                    "Geothermal_gradient":0.03, 
                    "Surface_T":288, "ESP_stage":282, "ESP_RPM":2890, "ESP_length": 8, "ESP_D":0.098,
                    "Surface_tension_GL":0.075,   "Surface_tension_OW":0.035,   ####
                    "ESP_GOR1":6.7,    "ESP_GOR2":6.7, "ESP_GOR3":6.7,
                    "ESP_RPM1":2890, "ESP_RPM2":2890,    "ESP_RPM3":2890, 
                    "ESP_WC1":75,    "ESP_WC2":75,   "ESP_WC3":75,   
                    "ESP_VISL1":1,    "ESP_VISL2":200,    "ESP_VISL3":500,     
                    "SPSA_a":1e-5,   'SPSA_n':2,     "SPSA_iter":50,
                     # missing input
                    "Pump_intake_T": 380, "P_in":3.12, "P_out":14.11, "ESP_Depth": 1320, "Q_ESP": 102.92,
                    "Q_water_ESP":80.36, "H_water_ESP":1651.41*9806.65/1e6
                     }
                    
        self.ESP_GEO=ESP_default['Flex31'].copy()
        self.QBEM = QBEM_default['Flex31']
        self.ESP_test_curve=None
        self.df_well_profile = pd.DataFrame()
        self.UpdateInput()

        # Output Data
        self.dfNodal = pd.DataFrame()
        self.df_pipe = pd.DataFrame()

        # connect signals and slots
        self.actionOpen.triggered.connect(self.actionOpen_triggered)
        self.actionSave.triggered.connect(self.actionSave_triggered)
        self.btnRun.clicked.connect(self.btnRun_clicked)
        self.btnStop.clicked.connect(self.btnStop_clicked)
        self.btnReset.clicked.connect(self.btnReset_clicked)
        self.PB_NodalAnalysis.clicked.connect(self.PB_NodalAnalysis_clicked)
        self.PB_PressureProfile.clicked.connect(self.PB_PressureProfile_clicked)
        self.PB_TemperatureProfile.clicked.connect(self.PB_TemperatureProfile_clicked)
        self.PB_HLProfile.clicked.connect(self.PB_HLProfile_clicked)
        self.PB_VelocityProfile.clicked.connect(self.PB_VelocityProfile_clicked)
        self.PB_read_well_profile.clicked.connect(self.PB_read_well_profile_clicked)
        self.PB_ESP_read_geometry.clicked.connect(self.PB_ESP_read_geometry_clicked)
        self.PB_ESP_read.clicked.connect(self.PB_ESP_read_clicked)
        self.PB_ESP_Cal.clicked.connect(self.PB_ESP_Cal_clicked)
        self.PB_ESP_SPSA.clicked.connect(self.PB_ESP_SPSA_clicked)
        # plot1 initialize
        self.figure1 = plt.figure()
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.toolbar1.setMaximumHeight(20)
        self.ax1 = self.figure1.add_subplot(111)
        plot_layout1 = QtWidgets.QVBoxLayout()
        plot_layout1.addWidget(self.toolbar1)
        plot_layout1.addWidget(self.canvas1)
        self.widgetESP.setLayout(plot_layout1)
    
    def setupGUI(self):
        self.setupUi(self)
        qtRectangle = self.frameGeometry()      # show on the center of screen
        centerPoint = QtWidgets.QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        self.show()
    
    def ReadInput(self):
        self.inputs["Surface_line_pressure"]=float(self.lineEdit_LinePressure.text())
        self.inputs["Surface_T"]=float(self.lineEdit_SurfaceT.text())
        self.inputs["Pipe_diameter"]=float(self.lineEdit_PipeDiameter.text())
        # self.inputs["Pipe_Length"]=float(self.lineEdit_PipeLength.text())
        self.inputs["Roughness"]=float(self.lineEdit_Roughness.text())
        # self.inputs["Inclination_angle"]=float(self.lineEdit_InclinationAngle.text())
        self.inputs["Liquid_viscosity"]=float(self.lineEdit_VisL.text())
        self.inputs["Liquid_relative_density"]=float(self.lineEdit_DenL.text())
        self.inputs["Gas_viscosity"]=float(self.lineEdit_VisG.text())
        self.inputs["Gas_relative_density"]=float(self.lineEdit_DenG.text())
        self.inputs["Surface_tension_GL"]=float(self.lineEdit_STGL.text())
        self.inputs["Surface_tension_OW"]=float(self.lineEdit_STOW.text())
        self.inputs["Reservoir_C"]=float(self.lineEdit_ResC.text())
        self.inputs["Reservoir_n"]=float(self.lineEdit_ResN.text())
        self.inputs["Reservoir_P"]=float(self.lineEdit_ResP.text())
        self.inputs["WC"]=float(self.lineEdit_WC.text())
        self.inputs["GOR"]=float(self.lineEdit_ResGOR.text())
        self.inputs["GLR"]=self.inputs["GOR"] * self.inputs["WC"]/100
        self.inputs["Geothermal_gradient"]=float(self.lineEdit_GeotermalGradient.text())

        self.inputs["ESP_depth"]=float(self.lineEdit_ESP_depth.text())
        self.inputs["ESP_RPM"]=float(self.lineEdit_ESP_RPM.text())
        self.inputs["ESP_stage"]=float(self.lineEdit_ESP_stage.text())
        self.inputs["ESP_D"]=float(self.lineEdit_ESP_D.text())
                
        self.inputs["Pump_intake_T"]=float(self.lineEdit_ESP_T.text())
        self.inputs["P_in"]=float(self.lineEdit_ESP_Pin.text())
        self.inputs["P_out"]=float(self.lineEdit_ESP_Pout.text())
        self.inputs["Q_ESP"]=float(self.lineEdit_ESP_Q.text())
        self.inputs["Q_water_ESP"]=float(self.lineEdit_ESP_Q_water.text())
        self.inputs["H_water_ESP"]=float(self.lineEdit_ESP_H_water.text())
        self.inputs["SPSA_n"]=float(self.lineEdit_SPSA_n.text())
        self.inputs["SPSA_a"]=float(self.lineEdit_SPSA_a.text())
        self.inputs["SPSA_iter"]=float(self.lineEdit_SPSA_iter.text())
        
        
        self.inputs["ESP_GOR1"]=float(self.lineEdit_ESP_GOR1.text())
        self.inputs["ESP_GOR2"]=float(self.lineEdit_ESP_GOR2.text())
        self.inputs["ESP_GOR3"]=float(self.lineEdit_ESP_GOR3.text())
        self.inputs["ESP_RPM1"]=float(self.lineEdit_ESP_RPM1.text())
        self.inputs["ESP_RPM2"]=float(self.lineEdit_ESP_RPM2.text())
        self.inputs["ESP_RPM3"]=float(self.lineEdit_ESP_RPM3.text())
        self.inputs["ESP_VISL1"]=float(self.lineEdit_ESP_VISL1.text())
        self.inputs["ESP_VISL2"]=float(self.lineEdit_ESP_VISL2.text())
        self.inputs["ESP_VISL3"]=float(self.lineEdit_ESP_VISL3.text())
        self.inputs["ESP_WC1"]=float(self.lineEdit_ESP_WC1.text())
        self.inputs["ESP_WC2"]=float(self.lineEdit_ESP_WC2.text())
        self.inputs["ESP_WC3"]=float(self.lineEdit_ESP_WC3.text())
        self.inputs["SPSA_a"]=float(self.lineEdit_SPSA_a.text())
        self.inputs["SPSA_n"]=float(self.lineEdit_SPSA_n.text())
        self.inputs["SPSA_iter"]=float(self.lineEdit_SPSA_iter.text())

    def UpdateInput(self):
        self.lineEdit_LinePressure.setValidator(QtGui.QDoubleValidator())    # set input format to be floating-point numbers
        self.lineEdit_LinePressure.setText(str(self.inputs["Surface_line_pressure"]))
        self.lineEdit_SurfaceT.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_SurfaceT.setText(str(self.inputs["Surface_T"]))
        self.lineEdit_PipeDiameter.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_PipeDiameter.setText(str(self.inputs["Pipe_diameter"]))
        self.lineEdit_Roughness.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_Roughness.setText(str(self.inputs["Roughness"]))
        self.lineEdit_VisL.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_VisL.setText(str(self.inputs["Liquid_viscosity"]))
        self.lineEdit_DenL.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_DenL.setText(str(self.inputs["Liquid_relative_density"]))
        self.lineEdit_VisG.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_VisG.setText(str(self.inputs["Gas_viscosity"]))
        self.lineEdit_DenG.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_DenG.setText(str(self.inputs["Gas_relative_density"]))
        self.lineEdit_STGL.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_STGL.setText(str(self.inputs["Surface_tension_GL"]))
        self.lineEdit_STOW.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_STOW.setText(str(self.inputs["Surface_tension_OW"]))
        self.lineEdit_ResC.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ResC.setText(str(self.inputs["Reservoir_C"]))
        self.lineEdit_ResN.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ResN.setText(str(self.inputs["Reservoir_n"]))
        self.lineEdit_ResP.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ResP.setText(str(self.inputs["Reservoir_P"]))
        self.lineEdit_ResGOR.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ResGOR.setText(str(self.inputs["GOR"]))
        self.lineEdit_GeotermalGradient.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_GeotermalGradient.setText(str(self.inputs["Geothermal_gradient"]))

        self.lineEdit_WC.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_WC.setText(str(self.inputs["WC"]))
        self.lineEdit_ESP_depth.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_depth.setText(str(self.inputs["ESP_Depth"]))
        self.lineEdit_ESP_RPM.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_RPM.setText(str(self.inputs["ESP_RPM"]))
        self.lineEdit_ESP_stage.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_stage.setText(str(self.inputs["ESP_stage"]))
        self.lineEdit_ESP_D.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_D.setText(str(self.inputs["ESP_D"]))
        self.lineEdit_ESP_Pin.setValidator(QtGui.QDoubleValidator())    

        self.lineEdit_ESP_T.setText(str(self.inputs["Pump_intake_T"]))
        self.lineEdit_ESP_T.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_Pin.setText(str(self.inputs["P_in"]))
        self.lineEdit_ESP_Pout.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_Pout.setText(str(self.inputs["P_out"]))
        self.lineEdit_ESP_Q.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_Q.setText(str(self.inputs["Q_ESP"]))
        self.lineEdit_ESP_Q_water.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_Q_water.setText(str(self.inputs["Q_water_ESP"]))
        self.lineEdit_ESP_H_water.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_H_water.setText(str(self.inputs["H_water_ESP"]))
        
        self.lineEdit_SPSA_n.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_SPSA_n.setText(str(self.inputs["SPSA_n"]))
        self.lineEdit_SPSA_a.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_SPSA_a.setText(str(self.inputs["SPSA_a"]))
        self.lineEdit_SPSA_iter.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_SPSA_iter.setText(str(self.inputs["SPSA_iter"]))
        
        self.lineEdit_ESP_GOR1.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_GOR1.setText(str(self.inputs["ESP_GOR1"]))
        self.lineEdit_ESP_GOR2.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_GOR2.setText(str(self.inputs["ESP_GOR2"]))
        self.lineEdit_ESP_GOR3.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_GOR3.setText(str(self.inputs["ESP_GOR3"]))
        self.lineEdit_ESP_RPM1.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_RPM1.setText(str(self.inputs["ESP_RPM1"]))
        self.lineEdit_ESP_RPM2.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_RPM2.setText(str(self.inputs["ESP_RPM2"]))
        self.lineEdit_ESP_RPM3.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_RPM3.setText(str(self.inputs["ESP_RPM3"]))
        self.lineEdit_ESP_VISL1.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_VISL1.setText(str(self.inputs["ESP_VISL1"]))
        self.lineEdit_ESP_VISL2.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_VISL2.setText(str(self.inputs["ESP_VISL2"]))
        self.lineEdit_ESP_VISL3.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_VISL3.setText(str(self.inputs["ESP_VISL3"]))
        self.lineEdit_ESP_WC1.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_WC1.setText(str(self.inputs["ESP_WC1"]))
        self.lineEdit_ESP_WC2.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_WC2.setText(str(self.inputs["ESP_WC2"]))
        self.lineEdit_ESP_WC3.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_ESP_WC3.setText(str(self.inputs["ESP_WC3"]))
        self.lineEdit_SPSA_a.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_SPSA_a.setText(str(self.inputs["SPSA_a"]))
        self.lineEdit_SPSA_n.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_SPSA_n.setText(str(self.inputs["SPSA_n"]))
        self.lineEdit_SPSA_iter.setValidator(QtGui.QDoubleValidator())    
        self.lineEdit_SPSA_iter.setText(str(self.inputs["SPSA_iter"]))

    @QtCore.pyqtSlot()
    def actionOpen_triggered(self):
        try:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Read Input File", "",
                                                                "All Files (*);;txt Files (*.txt);;csv Files (*.csv)",
                                                                options=options)
            if filename:
                if filename.split('.')[-1] == 'xlsx':
                        df=pd.read_excel(filename, sheet_name='Sheet1')
                        dict1 = df.to_dict('records')
                        self.inputs = dict1[0]
                        self.UpdateInput()
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("读取错误!")
            msg.setWindowTitle("错误")
            msg.exec_()

    @QtCore.pyqtSlot()
    def actionSave_triggered(self):
        try:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Read Input File", "",
                                                                "All Files (*);;txt Files (*.txt);;csv Files (*.csv)",
                                                                options=options)
            df = pd.DataFrame(data=self.inputs, index=[0])
            df.to_excel(filename, index=False)
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("存储失败!")
            msg.setWindowTitle("错误")
            msg.exec_()

    @QtCore.pyqtSlot()
    def btnRun_clicked(self):
        self.ReadInput()
        if self.df_well_profile.shape[0]==0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("缺失井眼轨迹")
            msg.setWindowTitle("错误")
            msg.exec_()
        else:
            progbar = NodalProgressBar(self.df_well_profile, self.inputs, self.ESP_GEO, self.QBEM)
            if progbar.exec_():
                self.dfNodal = progbar.dfNodal
                self.dfPipe = progbar.dfPipe


            # output table
            headersNodal = ['Qipr', 'Pipr', 'Qopr', 'Popr']
            columnsNodal = ['Qipr', 'Pipr', 'Qopr', 'Popr']
            self.fill_table(self.tableWidgetNodal, self.dfNodal, headersNodal, columnsNodal)
            
            headersPipe = ['Depth', 'HL', 'FP', 'P', 'T', 'VL', 'VO', 'VW', 'VG']
            columnsPipe = ['Depth', 'HL', 'FP', 'P', 'T', 'VL', 'VO', 'VW', 'VG']
            self.fill_table(self.tableWidgetPipe, self.dfPipe, headersPipe, columnsPipe)

    @QtCore.pyqtSlot()
    def btnStop_clicked(self):
        pass
    
    @QtCore.pyqtSlot()
    def btnReset_clicked(self):
        pass

    @QtCore.pyqtSlot()
    def PB_NodalAnalysis_clicked(self):
        try:
            dlg = Nodal_plot(self.dfNodal['Qipr'],self.dfNodal['Pipr'],self.dfNodal['Qopr'],self.dfNodal['Popr'])
            dlg.exec_()
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Please do calculation first")
            msg.setWindowTitle("Error")
            msg.exec_()
            
    @QtCore.pyqtSlot()
    def PB_PressureProfile_clicked(self):
        try:
            dlg = Pressure_plot(self.dfPipe['P'],self.dfPipe['Depth'])
            dlg.exec_()
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Please do calculation first")
            msg.setWindowTitle("Error")
            msg.exec_()
                
    @QtCore.pyqtSlot()
    def PB_TemperatureProfile_clicked(self):
        try:
            dlg = Temperature_plot(self.dfPipe['T'],self.dfPipe['Depth'])
            dlg.exec_()
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Please do calculation first")
            msg.setWindowTitle("Error")
            msg.exec_()

    @QtCore.pyqtSlot()
    def PB_HLProfile_clicked(self):
        try:
            dlg = HL_plot(self.dfPipe['HL'],self.dfPipe['FP'],self.dfPipe['Depth'])
            dlg.exec_()
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Please do calculation first")
            msg.setWindowTitle("Error")
            msg.exec_()

    @QtCore.pyqtSlot()
    def PB_VelocityProfile_clicked(self):
        try:
            dlg = Velocity_plot(self.dfPipe['VL'],self.dfPipe['VO'],self.dfPipe['VW'],self.dfPipe['VG'],self.dfPipe['Depth'])
            dlg.exec_()
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Please do calculation first")
            msg.setWindowTitle("Error")
            msg.exec_()
            
    @QtCore.pyqtSlot()
    def PB_read_well_profile_clicked(self):
        try:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Read Input File", "",
                                                                "All Files (*);;txt Files (*.txt);;csv Files (*.csv)",
                                                                options=options)
            if filename:
                if filename.split('.')[-1] == 'txt':
                    df_input = pd.read_table(filename).drop(["mark"], 1).set_index(["item"])
                    # for index in df_input.index:
                    #     self.inputValues[index] = df_input.loc[index].value.item()
                elif filename.split('.')[-1] == 'csv':
                    df_input = pd.read_csv(filename).drop(["mark"], 1).set_index(["item"])
                    # for index in df_input.index:
                    #     self.inputValues[index] = df_input.loc[index].value.item()
                elif filename.split('.')[-1] == 'xlsx':
                        self.df_well_profile = pd.read_excel(filename, sheet_name='Sheet1')
                        self.df_well_profile.rename(columns={self.df_well_profile.columns[0]: "ID", self.df_well_profile.columns[1]: "MD", 
                                self.df_well_profile.columns[2]: "DL", self.df_well_profile.columns[3]: "Angle", self.df_well_profile.columns[4]: "TVD", 
                                self.df_well_profile.columns[5]: "HD"}, inplace=True)

                    # for index in df_input.index:
                    #     self.inputValues[index] = df_input.loc[index].value.item()
                else:
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Critical)
                    msg.setText("错误的井眼轨迹!")
                    msg.setWindowTitle("错误")
                    msg.exec_()
            # output table
            headersNodal = list(self.df_well_profile.columns)
            columnsNodal = list(self.df_well_profile.columns)
            self.fill_table(self.tableWidgetWell, self.df_well_profile, headersNodal, columnsNodal)

        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("错误的井眼轨迹")
            msg.setWindowTitle("错误")
            msg.exec_()
    
    @QtCore.pyqtSlot()
    def PB_ESP_read_geometry_clicked(self):
        try:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Read Input File", "",
                                                                "All Files (*);;txt Files (*.txt);;csv Files (*.csv)",
                                                                options=options)
            if filename:
                if filename.split('.')[-1] == 'xlsx':
                        self.ESP_GEO = pd.read_excel(filename, sheet_name='Sheet1')
                        self.QBEM = self.ESP_GEO['QBEM']
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("错误的电潜泵尺寸!")
            msg.setWindowTitle("错误")
            msg.exec_()

    @QtCore.pyqtSlot()
    def PB_ESP_read_clicked(self):
        try:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Read Input File", "",
                                                                "All Files (*);;txt Files (*.txt);;csv Files (*.csv)",
                                                                options=options)
            if filename:
                if filename.split('.')[-1] == 'xlsx':
                        self.ESP_test_curve = pd.read_excel(filename, sheet_name='Sheet1')

                    # for index in df_input.index:
                    #     self.inputValues[index] = df_input.loc[index].value.item()
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("错误的电潜泵实验数据!")
            msg.setWindowTitle("错误")
            msg.exec_()

    @QtCore.pyqtSlot()
    def PB_ESP_Cal_clicked (self):
        self.ReadInput()
        try:
            QL = np.arange(0.01, 1.1, 0.02) * 10000 * (self.ESP_GEO.iloc[0].R2/0.04)

            self.ESP_GEO.N.iloc[0] = self.inputs['ESP_RPM1']


            ESP_GEO = self.ESP_GEO.iloc[0].to_dict()
            HP1 = ESP_head (self.ESP_GEO.QBEM.iloc[0], ESP_GEO, QL, 0.001, 1000, 
                    self.inputs['ESP_VISL1']/1000, self.inputs['Liquid_relative_density']*1000, 
                    0.000018, self.inputs['ESP_WC1'], self.inputs['ESP_GOR1']*self.inputs['ESP_WC1']/100, 
                    (self.inputs['P_in']+self.inputs['P_out'])/2*1e6, self.inputs['Pump_intake_T'], 0.035, None)

            self.ESP_GEO.N.iloc[0] = self.inputs['ESP_RPM2']
            HP2 = ESP_head (self.ESP_GEO.QBEM.iloc[0], ESP_GEO, QL, 0.001, 1000, 
                    self.inputs['ESP_VISL2']/1000, self.inputs['Liquid_relative_density']*1000, 
                    0.000018, self.inputs['ESP_WC2'], self.inputs['ESP_GOR2']*self.inputs['ESP_WC2']/100, 
                    (self.inputs['P_in']+self.inputs['P_out'])/2*1e6, self.inputs['Pump_intake_T'], 0.035, None)

            self.ESP_GEO.N.iloc[0] = self.inputs['ESP_RPM3']
            HP3 = ESP_head (self.ESP_GEO.QBEM.iloc[0], ESP_GEO, QL, 0.001, 1000, 
                    self.inputs['ESP_VISL3']/1000, self.inputs['Liquid_relative_density']*1000, 
                    0.000018, self.inputs['ESP_WC3'], self.inputs['ESP_GOR3']*self.inputs['ESP_WC3']/100, 
                    (self.inputs['P_in']+self.inputs['P_out'])/2*1e6, self.inputs['Pump_intake_T'], 0.035, None)

            
            df1 = pd.DataFrame({'QL':QL*bpd_to_m3d, 'HP1':HP1*psi_to_pa/1e6})
            df1 = df1[df1.HP1>0]
            df2 = pd.DataFrame({'QL':QL*bpd_to_m3d, 'HP2':HP2*psi_to_pa/1e6})
            df2 = df2[df2.HP2>0]
            df3 = pd.DataFrame({'QL':QL*bpd_to_m3d, 'HP3':HP3*psi_to_pa/1e6})
            df3 = df3[df3.HP3>0]
            self.figure1.clear()
            self.ax1 = self.figure1.add_subplot(111)
            self.ax1.plot(df1.QL, df1.HP1, 'k-', label='曲线1')
            self.ax1.plot(df2.QL, df2.HP2, 'r-', label='曲线2')
            self.ax1.plot(df3.QL, df3.HP3, 'b-', label='曲线3')
            self.ax1.tick_params(direction='in', labelsize=9)
            self.ax1.set_title(r'电潜泵扬程', fontsize=12)
            self.ax1.set_xlabel('排量（m3/d）', fontsize=12)
            self.ax1.set_ylabel('扬程（Mpa）', fontsize=12)
            self.ax1.legend(frameon=False, fontsize=9)
            self.ax1.set_ylim(0)
            self.ax1.set_xlim(0)
            self.figure1.tight_layout()
            self.canvas1.draw()
            
            self.ESP_GEO = ESP_GEO
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("错误的电潜泵几何尺寸!")
            msg.setWindowTitle("错误")
            msg.exec_()

    @QtCore.pyqtSlot()
    def PB_ESP_SPSA_clicked (self):
        try:
            self.ReadInput()
            if self.ESP_test_curve == None:
                Q1 = self.inputs['Q_water_ESP']/bbl_to_m3
                Q2 = self.inputs['Q_ESP']/bbl_to_m3
                H1 = self.inputs['H_water_ESP']*1e6/psi_to_pa/self.inputs['ESP_stage']
                H2 = (self.inputs['P_out']-self.inputs['P_in'])*1e6/psi_to_pa/self.inputs['ESP_stage']
                Input = pd.DataFrame({'QL':[Q1, Q2], 
                            'HP': [H1, H2], 
                            'VISO':[1,self.inputs['Liquid_viscosity']*1000], 
                            'RPM':[self.inputs['ESP_RPM'],self.inputs['ESP_RPM']],
                            'DENO':[1000,self.inputs['Liquid_relative_density']*1000],
                             'VISG':[0.000018,0.000018], 
                             'WC':[100,self.inputs['WC']], 
                             'GLR':[0,self.inputs['GLR']], 
                             'P':[0.1e6,(self.inputs['P_in']+self.inputs['P_out'])/2*1e6], 
                             'TT': [288,self.inputs['Pump_intake_T']], 
                             'STOW': [0.035,self.inputs['Surface_tension_OW']]
                             })
            else:
                Input = self.ESP_test_curve
            Input.VISO = Input.VISO/1000
            noise_var, a_par, max_iter, report=0.1, self.inputs['SPSA_a'], int(self.inputs['SPSA_iter']), 10
            Train_parameter = np.ones(int(self.inputs['SPSA_n']))
            min_vals=np.ones(int(self.inputs['SPSA_n']))*0.1
            max_vals = np.ones(int(self.inputs['SPSA_n']))*10

            _, Train_parameter1, _, _ = SPSA_match(Train_parameter, Input, Input.HP, noise_var, a_par, min_vals, max_vals, max_iter, report)


            ESP_input, QBEM = ESP_geometry(base_pump = 'Flex31', Train_parameter=Train_parameter1)

            ESP_GEO = ESP_input
            QL = np.arange(0.01, 1.1, 0.02) * 10000 * (ESP_GEO['R2']/0.04)
            HP1 = ESP_head (QBEM, ESP_GEO, QL, 0.001, 1000, 
                    self.inputs['ESP_VISL1']/1000, self.inputs['Liquid_relative_density']*1000, 
                    0.000018, self.inputs['ESP_WC1'], self.inputs['ESP_GOR1']*self.inputs['ESP_WC1']/100, 
                    (self.inputs['P_in']+self.inputs['P_out'])/2*1e6, self.inputs['Pump_intake_T'], 0.035, None)

            ESP_GEO['N'] = self.inputs['ESP_RPM2']
            HP2 = ESP_head (QBEM, ESP_GEO, QL, 0.001, 1000, 
                    self.inputs['ESP_VISL2']/1000, self.inputs['Liquid_relative_density']*1000, 
                    0.000018, self.inputs['ESP_WC2'], self.inputs['ESP_GOR2']*self.inputs['ESP_WC2']/100, 
                    (self.inputs['P_in']+self.inputs['P_out'])/2*1e6, self.inputs['Pump_intake_T'], 0.035, None)

            ESP_GEO['N'] = self.inputs['ESP_RPM3']
            HP3 = ESP_head (QBEM, ESP_GEO, QL, 0.001, 1000, 
                    self.inputs['ESP_VISL3']/1000, self.inputs['Liquid_relative_density']*1000, 
                    0.000018, self.inputs['ESP_WC3'], self.inputs['ESP_GOR3']*self.inputs['ESP_WC3']/100, 
                    (self.inputs['P_in']+self.inputs['P_out'])/2*1e6, self.inputs['Pump_intake_T'], 0.035, None)

            
            
            df1 = pd.DataFrame({'QL':QL*bpd_to_m3d, 'HP1':HP1*psi_to_pa/1e6})
            df1 = df1[df1.HP1>0]
            df2 = pd.DataFrame({'QL':QL*bpd_to_m3d, 'HP2':HP2*psi_to_pa/1e6})
            df2 = df2[df2.HP2>0]
            df3 = pd.DataFrame({'QL':QL*bpd_to_m3d, 'HP3':HP3*psi_to_pa/1e6})
            df3 = df3[df3.HP3>0]
            self.figure1.clear()
            self.ax1 = self.figure1.add_subplot(111)
            self.ax1.plot(df1.QL, df1.HP1, 'k-', label='曲线1')
            self.ax1.plot(df2.QL, df2.HP2, 'r-', label='曲线2')
            self.ax1.plot(df3.QL, df3.HP3, 'b-', label='曲线3')
            self.ax1.scatter((Input.QL*bpd_to_m3d).to_numpy(), (Input.HP*psi_to_pa/1e6).to_numpy(), label='实验/现场')
            self.ax1.tick_params(direction='in', labelsize=9)
            self.ax1.set_title(r'电潜泵扬程', fontsize=12)
            self.ax1.set_xlabel('排量（m3/d）', fontsize=12)
            self.ax1.set_ylabel('扬程（Mpa）', fontsize=12)
            self.ax1.legend(frameon=False, fontsize=9)
            self.ax1.set_ylim(0)
            self.ax1.set_xlim(0)
            self.figure1.tight_layout()
            self.canvas1.draw()

            self.ESP_GEO = ESP_GEO
            self.QBEM = QBEM
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("SPSA计算错误，查看输入参数!")
            msg.setWindowTitle("错误")
            msg.exec_()
    
    @staticmethod
    def fill_table(qtable, df_data, headers, columns):
        """
        :param qtable:  Qtable to fill
        :param df_data: dataframe
        :param headers: headers to change table headers
        :param columns: columns for selecting data
        :return:        None
        """
        try:
            qtable.clear()
            df = df_data[columns]
            qtable.setColumnCount(df.shape[1])
            qtable.setRowCount(df.shape[0])
            qtable.setHorizontalHeaderLabels(headers)
            qtable.resizeColumnsToContents()
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    try:
                        item = str(np.around(df.iloc[i, j], decimals=6))
                    except TypeError:
                        item = str(df.iloc[i, j])
                    qtable.setItem(i, j, QtWidgets.QTableWidgetItem(item))
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("输出数据错误!")
            msg.setWindowTitle("错误")
            msg.exec_()
        
class NodalProgressBar(QtWidgets.QDialog, Ui_progressDialog):
    def __init__(self, df_well_profile, inputValues, ESP_GEO, QBEM, parent=None):
        super(NodalProgressBar, self).__init__(parent)
        self.setupUi(self)              
        self.dfNodal = pd.DataFrame()
        self.dfPipe = pd.DataFrame()
        self.df_well_profile = df_well_profile
        self.df_well_profile = df_well_profile
        self.ESP_GEO = ESP_GEO
        self.QBEM = QBEM
        self.progressBar.setMaximum(4)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.NodalThread = NodalThread(df_well_profile, inputValues, ESP_GEO, QBEM)
        self.NodalThread.update.connect(self.update)
        self.NodalThread.finished.connect(self.finished)
        self.btn_OK.clicked.connect(self.accept)
        self.icon = 0
        self.NodalThread.start()

    @QtCore.pyqtSlot()
    def finished(self):
        self.dfNodal = pd.DataFrame({'Qipr': self.NodalThread.Qipr, 'Pipr': self.NodalThread.Pipr,
                                     'Qopr': self.NodalThread.Qopr, 'Popr': self.NodalThread.Popr})
        self.dfPipe = pd.DataFrame({'HL': self.NodalThread.HL, 'P': self.NodalThread.Ppipe,
                                     'T': self.NodalThread.Tpipe, 'VO': self.NodalThread.VO, 
                                     'VL': self.NodalThread.VL, 'VG': self.NodalThread.VG,
                                     'VW': self.NodalThread.VW, 'Depth': self.df_well_profile.MD,
                                     'FP': self.NodalThread.fp})
        self.dfPipe.to_excel("result/dfPipe.xlsx")
        self.dfNodal.to_excel("result/dfNodal.xlsx")

        self.progressBar.setValue(4)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("The calculation is finished!")
        msg.setWindowTitle("Finish")
        msg.exec_()

    @QtCore.pyqtSlot()
    def update(self):
        self.icon += 1
        self.progressBar.setValue(self.icon)

class NodalThread(QtCore.QThread):
    # class variables for signals
    update = QtCore.pyqtSignal()

    def __init__(self, df_well_profile, inputValues, ESP_GEO, QBEM):
        super(NodalThread, self).__init__()
        self.df_well_profile = df_well_profile                          # maximum run times
        self.inputValues = inputValues      # input dictionary
        self.ESP_GEO = ESP_GEO
        self.QBEM = QBEM
        self.HL=[]
        self.Ppipe=[]
        self.Tpipe=[]
        self.VO=[]
        self.VW=[]
        self.VL=[]
        self.VG=[]
        self.fp=[]      # flow pattern
        

        self.Qipr=[]
        self.Pipr=[]
        self.Qopr=[]
        self.Popr=[]
   
    def run(self):
        """
        Qg_res: gas flow rate from reservoir
        Ql_res: liquid flow rate from reservoir
        pwf: Near well bore pressure (flow pressure) used to calculate flow rate from reservoir
        pbh: bottom hole pressure calculated from surface by using the flow rate from reservoir and unified model 
        
        """
        well_name = '22b'
        Q_oil = self.inputValues['Q_ESP']
        WC=self.inputValues['WC']
        GLR = self.inputValues['GLR']
        ESP_input=self.ESP_GEO
        QBEM=self.QBEM
        ESP_input['SN']=self.inputValues['ESP_stage']
        ESP_input['N']=self.inputValues['ESP_RPM']
        self.update.emit()
        '''Pressure profile'''
        ql, P = Nodal_solve(self.inputValues, self.df_well_profile, GLR, WC, QBEM, ESP_input)
        print('Nodal: ', ql,' m3d', P,' Mpa')
        Qg_res = ql*GLR/24/3600
        Qo_res = ql*(1-WC/100)/24/3600
        Qw_res = ql*WC/100/24/3600
        P, df = P_bottom_Cal (self.inputValues, self.df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input)
        self.HL=df.List_HL
        self.Ppipe=df.List_P
        self.Tpipe=df.List_T
        self.VO=df.List_VO
        self.VW=df.List_VW
        self.VL=df.List_VL
        self.VG=df.List_VG
        self.fp=df.List_ID
        '''Nodel'''
        QL = np.array([ql/32,ql/16,ql/8,ql/4,ql/2,ql,ql*1.5])    # m3/d
        QL = QL/24/3600
        Pr = self.inputValues['Reservoir_P']
        Pwf = np.array([0, Pr/6, Pr/5,Pr/4,Pr/3,Pr/2,Pr/1])
        self.Qopr, self.Popr, self.Qipr, self.Pipr = Nodal(self.inputValues, self.df_well_profile, QL, GLR, Pwf, WC, QBEM, ESP_input, Q_oil, df.List_P.iloc[-1], well_name)
        self.update.emit()
        self.update.emit()

class Nodal_plot(QtWidgets.QDialog, Ui_Nodal_plot):
    def __init__(self, Qipr, Pipr, Qopr, Popr, parent=None):
        super(Nodal_plot, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pushButton_clicked)
        
        # plot1 initialize
        self.figure1 = plt.figure(tight_layout=True)
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.toolbar1.setMaximumHeight(20)
        self.ax1 = self.figure1.add_subplot(111)
        plot_layout1 = QtWidgets.QVBoxLayout()
        plot_layout1.addWidget(self.toolbar1)
        plot_layout1.addWidget(self.canvas1)
        self.widget.setLayout(plot_layout1)
        
        self.figure1.clear()
        self.ax1 = self.figure1.add_subplot(111)
        self.ax1.plot(Qipr, Pipr, 'k-', label='节点流入IPR曲线')
        self.ax1.plot(Qopr, Popr, 'r-', label='节点流出OPR曲线')
        self.ax1.tick_params(direction='in', labelsize=9)
        self.ax1.set_title(r'节点分析法', fontsize=12)
        self.ax1.set_xlabel('产液量（m3/d）', fontsize=12)
        self.ax1.set_ylabel('井底压力（MPa）', fontsize=12)
        self.ax1.legend(frameon=False, fontsize=9)
        self.ax1.set_ylim(0)
        self.ax1.set_xlim(0)
        self.figure1.tight_layout()
        self.canvas1.draw()

        # ax.scatter(Q_oil, H_oil, label='现场（%.2fm$^3$/d，%.2fMPA）' % (Q_oil,H_oil), marker='*', linewidth=0.75)

    def pushButton_clicked(self):
        try:
            # self.table_input.clearContents()
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data File", "",
                                                                "Images (*.jpg)",
                                                                options=options)
            # Write XML file
            self.figure1.savefig(filename)
              
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error in save plot!")
            msg.setWindowTitle("Save File Error")
            msg.exec_()

class Pressure_plot(QtWidgets.QDialog, Ui_Nodal_plot):
    def __init__(self, P, L, parent=None):
        super(Pressure_plot, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pushButton_clicked)
        
        # plot1 initialize
        self.figure1 = plt.figure(tight_layout=True)
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.toolbar1.setMaximumHeight(20)
        self.ax1 = self.figure1.add_subplot(111)
        plot_layout1 = QtWidgets.QVBoxLayout()
        plot_layout1.addWidget(self.toolbar1)
        plot_layout1.addWidget(self.canvas1)
        self.widget.setLayout(plot_layout1)
        
        self.figure1.clear()
        self.ax1 = self.figure1.add_subplot(111)
        self.ax1.plot(L, P, 'k-', label='压力')
        self.ax1.tick_params(direction='in', labelsize=9)
        self.ax1.set_title(r'井的压力剖面', fontsize=12)
        self.ax1.set_xlabel('测深 (m)', fontsize=12)
        self.ax1.set_ylabel('压力（MPa）', fontsize=12)
        self.ax1.legend(frameon=False, fontsize=9)
        self.figure1.tight_layout()
        self.canvas1.draw()


    @QtCore.pyqtSlot()

    def pushButton_clicked(self):
        try:
            # self.table_input.clearContents()
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data File", "",
                                                                "Images (*.jpg)",
                                                                options=options)
            # Write XML file
            self.figure1.savefig(filename)
              
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error in save plot!")
            msg.setWindowTitle("Save File Error")
            msg.exec_()

class Temperature_plot(QtWidgets.QDialog, Ui_Nodal_plot):
    def __init__(self, T, L, parent=None):
        super(Temperature_plot, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pushButton_clicked)
        
        # plot1 initialize
        self.figure1 = plt.figure(tight_layout=True)
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.toolbar1.setMaximumHeight(20)
        self.ax1 = self.figure1.add_subplot(111)
        plot_layout1 = QtWidgets.QVBoxLayout()
        plot_layout1.addWidget(self.toolbar1)
        plot_layout1.addWidget(self.canvas1)
        self.widget.setLayout(plot_layout1)
        
        self.figure1.clear()
        self.ax1 = self.figure1.add_subplot(111)
        self.ax1.plot(L, T, 'k-', label='温度')
        self.ax1.tick_params(direction='in', labelsize=9)
        self.ax1.set_title(r'井的压力剖面', fontsize=12)
        self.ax1.set_xlabel('测深 (m)', fontsize=12)
        self.ax1.set_ylabel('温度 (K)', fontsize=12)
        self.ax1.legend(frameon=False, fontsize=9)
        self.figure1.tight_layout()
        self.canvas1.draw()
    @QtCore.pyqtSlot()

    def pushButton_clicked(self):
        try:
            # self.table_input.clearContents()
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data File", "",
                                                                "Images (*.jpg)",
                                                                options=options)
            # Write XML file
            self.figure1.savefig(filename)
              
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error in save plot!")
            msg.setWindowTitle("Save File Error")
            msg.exec_()

class Velocity_plot(QtWidgets.QDialog, Ui_Nodal_plot):
    def __init__(self, VL, VO, VW, VG, L, parent=None):
        super(Velocity_plot, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pushButton_clicked)
        
        # plot1 initialize
        self.figure1 = plt.figure(tight_layout=True)
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.toolbar1.setMaximumHeight(20)
        self.ax1 = self.figure1.add_subplot(111)
        plot_layout1 = QtWidgets.QVBoxLayout()
        plot_layout1.addWidget(self.toolbar1)
        plot_layout1.addWidget(self.canvas1)
        self.widget.setLayout(plot_layout1)

        self.figure1.clear()
        self.ax1 = self.figure1.add_subplot(111)
        self.ax1.plot(L, VL, 'k-', label='液相流速')
        self.ax1.plot(L, VO, 'r-', label='油相流速')
        self.ax1.plot(L, VW, 'b-', label='水相流速')
        self.ax1.legend(frameon=False, loc='upper left', fontsize=9)
        self.ax1.tick_params(direction='in', labelsize=9)
        self.ax1.set_title(r'流速剖面', fontsize=12)
        self.ax1.set_xlabel('测深 (m)', fontsize=12)
        self.ax1.set_ylabel('液相流速 (m/s)', fontsize=12)
        
        ax44 = self.ax1.twinx()
        ax44.plot(L, VG, color = 'g', label='气相流速', linewidth=0.75)
        ax44.set_ylabel('气相流速 (m/s)', fontsize=12)
        ax44.legend(frameon=False, loc='lower right', fontsize=9)
        ax44.tick_params(direction='in', labelsize=9)
        self.figure1.tight_layout()
        self.canvas1.draw()
    @QtCore.pyqtSlot()

    def pushButton_clicked(self):
        try:
            # self.table_input.clearContents()
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data File", "",
                                                                "Images (*.jpg)",
                                                                options=options)
            # Write XML file
            self.figure1.savefig(filename)
              
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error in save plot!")
            msg.setWindowTitle("Save File Error")
            msg.exec_()

class HL_plot(QtWidgets.QDialog, Ui_Nodal_plot):
    def __init__(self, HL, ID, L, parent=None):
        super(HL_plot, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pushButton_clicked)
        
        # plot1 initialize
        self.figure1 = plt.figure(tight_layout=True)
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.toolbar1.setMaximumHeight(20)
        self.ax1 = self.figure1.add_subplot(111)
        plot_layout1 = QtWidgets.QVBoxLayout()
        plot_layout1.addWidget(self.toolbar1)
        plot_layout1.addWidget(self.canvas1)
        self.widget.setLayout(plot_layout1)
        
        self.figure1.clear()
        self.ax1 = self.figure1.add_subplot(111)
        self.ax1.plot(L, HL*100, 'k-', label='持液率')
        self.ax1.tick_params(direction='in', labelsize=9)
        self.ax1.set_title(r'持液率和流态剖面', fontsize=12)
        self.ax1.set_xlabel('测深 (m)', fontsize=12)
        self.ax1.set_ylabel('持液率 (%)', fontsize=12)
        self.ax1.legend(frameon=False, fontsize=9)

        ax44 = self.ax1.twinx()
        ax44.plot(L, ID, color = 'g', label='流态', linewidth=0.75)
        ax44.set_ylabel('1:层流 2:气泡流 3:段塞流 4:环空流, 5:均质', fontsize=12)
        ax44.legend(frameon=False, loc='lower right', fontsize=9)
        ax44.tick_params(direction='in', labelsize=9)
        self.figure1.tight_layout()

        self.canvas1.draw()
        
    def pushButton_clicked(self):
        try:
            # self.table_input.clearContents()
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data File", "",
                                                                "Images (*.jpg)",
                                                                options=options)
            # Write XML file
            self.figure1.savefig(filename)
              
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error in save plot!")
            msg.setWindowTitle("Save File Error")
            msg.exec_()

# functions
def Temperature_C_K(T_C):
    T_K = T_C + 273.15
    return T_K

def Temperature_C_F(T_C):
    T_F = (T_C*9/5)+32
    return T_F

def P_bottom_Cal (inputValues, df_well_profile, Qg_res = 10, Qo_res = 0.1, Qw_res = 0.1, QBEM=6000, ESP_input=ESP['Flex31']):
    P=inputValues['Surface_line_pressure']
    T=inputValues['Surface_T']
    
    
    A_pipe = 3.14/4*inputValues['Pipe_diameter']**2.0
    D = inputValues['Pipe_diameter']
    ED = inputValues['Roughness']
    DENO = inputValues['Liquid_relative_density'] * 1000
    WC = inputValues['WC'] 
    DENL = DENO*(100-WC)/100+1000*WC/100
    GLR = inputValues['GLR'] 
    ESP_bottom = inputValues['ESP_Depth']
    ESP_top = inputValues['ESP_Depth'] - inputValues['ESP_length']
    for i in range(0, df_well_profile.shape[0], 1):
        if i>1 and df_well_profile.MD[i-1] < ESP_top and df_well_profile.MD[i] > ESP_top:
            i_ESP_top = i
        if i>1 and df_well_profile.MD[i-1] < ESP_bottom and df_well_profile.MD[i] > ESP_bottom:
            i_ESP_bottom = i
  
    list_P = []
    list_ID = []
    List_VL = []
    List_VO = []
    List_VW = []
    List_VG = []
    List_HL = []
    List_T = []
    List_DENG = []
    List_VISG = []
    List_ST = []

    ESP_complete = 'No'
    HP = 0
    for i in range(0, df_well_profile.shape[0], 1):
        VSO = Qo_res/A_pipe
        VSW = Qw_res/A_pipe
        VSL = VSO+VSW
        DENG = gas_density(inputValues['Gas_relative_density'], T, P)
        DENG_stg = gas_density(inputValues['Gas_relative_density'], 288, 101325)
        VSG = Qg_res/A_pipe/DENG*DENG_stg
        # VISO = liquid_viscosity(inputValues['Liquid_relative_density'], T)
        VISO = inputValues['Liquid_viscosity']
        VISL = (VISO*(100-WC)/100+0.001*WC/100)
        VISG = gas_viscosity(inputValues['Gas_relative_density'], T, P/psi_to_pa)
        STGL = surface_tension(inputValues['Liquid_relative_density'], inputValues['Gas_relative_density'], T)
        
        # CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, \
        #         ICON = GAL(D, ED, ANG, VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, [1])
        CU, CF, FE, FI, FQN, HLF, HLS, HL, PGT, PGA, PGF, PGG, RSU, SL, SI, TH, TH0, VF, VC, VT, FGL, IFGL, ICON= GAL(D, ED, 90-df_well_profile.Angle[i], VSL, VSG, DENL, DENG, VISL, VISG, STGL, P, [1])
        P=P-PGT*df_well_profile.DL[i]
        T=T+inputValues['Geothermal_gradient']*df_well_profile.DL[i]
    
        if i>1 and i >= i_ESP_bottom and ESP_complete == 'No':
            # di = i_ESP_bottom - i_ESP_top
            P_ave = P/1.5
            GLR = GLR*DENG_stg/DENG
            while True:
                QL = np.array((Qw_res+Qo_res)*m3s_to_bpd)
                # DENG = gas_density(inputValues['Gas_relative_density'], T, P_ave)
                # DENG_stg = gas_density(inputValues['Gas_relative_density'], 288, 101325)
                # QG = Qg_res/DENG*DENG_stg
                # GVF = QG/(QG+QL)
                HP= ESP_head (QBEM, ESP_input, QL, 0.001, 1000, VISO, DENO, VISG, WC, GLR, P_ave, T, 0.035, None)
                ESP_complete = 'Yes'
                HP = HP * psi_to_pa
                P_new = P-HP*ESP_input['SN']
                P_ave2 = (P_new + P)/2
                if abs(P_ave - P_ave2) / abs(P_ave) > 1e-3:
                    P_ave = P_ave2*0.5+P_ave*0.5

                    if i > 1000:
                        break
                else:
                    P = P_new
                    break
            
        list_P.append(P/1e6)
        if HL == 0:
            List_VL.append(0)
            List_VO.append(0)
            List_VW.append(0)
            List_VG.append(VSG/(1-HL))
            List_HL.append(HL)
        elif HL == 1:
            List_VL.append(VSL/HL)
            List_VO.append(VSO/HL)
            List_VW.append(VSW/HL)
            List_VG.append(0)
            List_HL.append(HL)
        else:
            List_VL.append(VSL/HL)
            List_VO.append(VSO/HL)
            List_VW.append(VSW/HL)
            List_VG.append(VSG/(1-HL))
            List_HL.append(HL)
        List_T.append(T)
        List_DENG.append(DENG)
        List_VISG.append(VISG)
        List_ST.append(STGL)
        if FGL == 'STR':
            list_ID.append(1)
        elif FGL == 'BUB':
            list_ID.append(2)
        elif FGL == 'INT':
            list_ID.append(3)
        elif FGL == 'D-B':
            list_ID.append(5)
        else:
            list_ID.append(4)

    df = pd.DataFrame({'List_P': list_P, 'List_ID': list_ID, 'List_VL': List_VL, 'List_VO':List_VO, 
                    'List_VW': List_VW, 'List_VG': List_VG, 'List_HL': List_HL, 'List_T':List_T, 'List_DENG': List_DENG, 
                    'List_VISG': List_VISG, 'List_ST': List_ST})
    return P, df

def Nodal (inputValues, df_well_profile, QL_list, GLR, Pwf_list, WC, QBEM, ESP_input, Q_oil, H_oil, well_name):
    Q_bottom = []
    P_bottom = []
    
    for ql in QL_list:
        Qw_res = ql*WC/100
        Qo_res = ql*(100-WC)/100/inputValues['Liquid_relative_density']
        Qg_res = ql*GLR
        P, _= P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input)
        Q_bottom.append(ql*24*3600)
        P_bottom.append(P/1e6)

    Q_bottom_IPR = []
    P_bottom_IPR = []
    for pwf in Pwf_list:
        Qg_res, Ql_res = fgreservoir(inputValues['Reservoir_P'], pwf, inputValues['Reservoir_C'], inputValues['Reservoir_n'], inputValues['GLR'])
        Q_bottom_IPR.append(Ql_res*24*3600)
        P_bottom_IPR.append(pwf/1e6)

    return Q_bottom,P_bottom,Q_bottom_IPR,P_bottom_IPR

def Nodal_solve (inputValues, df_well_profile, GLR, WC, QBEM, ESP_input):
    ql = 0.001
    for i in range (1000):
        Qw_res = ql*WC/100
        Qo_res = ql*(100-WC)/100/inputValues['Liquid_relative_density']
        Qg_res = ql*GLR
        P, _= P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input)
        # Q_bottom.append(ql*24*3600)
        # P_bottom.append(P/1e6)
        Qg_res_IPR, Ql_res_IPR = fgreservoir(inputValues['Reservoir_P'], P, inputValues['Reservoir_C'], inputValues['Reservoir_n'], inputValues['GLR'])

        if abs(Ql_res_IPR-ql)/ql<0.001:
            return ql*24*3600, P/1e6
        else:
            ql = (Ql_res_IPR+ql)/2
    return ql*24*3600, P/1e6

def ESP_head (QBEM=6000, ESP_input=ESP, QL=np.arange(0.001, 1.1, 0.002) *5000, VISW_in=0.001, DENW_in=1000, VISO_in = 0.5, DENO_in = 950, VISG_in = 0.000018, WC=0.8, GLR=5, P = 350, T=288, O_W_ST = 0.035, GVF = None):
    '''
    QL (in bpd): list of flow rate to predict corresponding pump head
    '''
    VISL = VISO_in
    VISW = VISW_in
    VISG = VISG_in
    DENL = DENO_in
    DENW = DENW_in
    DENG_std = gas_density(0.7, 288, 101325)
    DENG = gas_density(0.7, T, P)

    P = P * np.ones(QL.shape)
    T = T * np.ones(QL.shape)
    try:
        N = ESP_input['N'].iloc[0]
        NS = ESP_input['NS'].iloc[0]
        SGM = ESP_input['SGM'].iloc[0]
        SN = ESP_input['SN'].iloc[0]
        ST = ESP_input['ST'].iloc[0]
    except:
        N = ESP_input['N']
        NS = ESP_input['NS'] 
        SGM = ESP_input['SGM'] 
        SN = ESP_input['SN'] 
        ST = ESP_input['ST'] 
        WC = WC 
    
    QL =  QL * bbl_to_m3 / 24.0 / 3600.0
    QBEM = QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(QL.shape)# * np.ones(QL.shape)

    if GVF == None:
        QG = QL * GLR * DENG_std/DENG
    else: 
        QG = QL * GVF

    gl = GasLiquidModel(ESP_input,QBEM)
    gl_cal = np.vectorize(gl.gl_calculate_new)
    flgz = 'Z'   # Jianjun Zhu simple version model
    HP, HE, HF, HT, HD, HRE, HLKloss, QLK, GV= gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgz)

    return HP

def ESP_compare (ESP_input, inputValues, QBEM,well_name,Q_water=80,H_water=1650,Q_Oil=103,H_Oil=1215, VisL_target=523):
    QL= np.arange(0.005, 1.1, 0.025) * 5000.0

    VISO_in=inputValues['Liquid_viscosity']
    WC=inputValues['WC']
    # VISO_in=0.001
    VISW_in=0.001
    VISG_in=inputValues['Gas_viscosity']
    DENO_in=inputValues['Liquid_relative_density']*1000
    DENW_in=1000
    DENG_std = gas_density(inputValues['Gas_relative_density'], 288, 101325)
    # DENG = gas_density(inputValues['Gas_relative_density'], inputValues['Pump_intake_T'], 
    #             (inputValues['P_in']+inputValues['P_out'])/2*1e6)
    GLR = inputValues['GLR']
    GOR = GLR/(1-WC/100)
    # GLR=GLR*DENG_std/DENG
    # 22b well
    P = (inputValues['P_in']+inputValues['P_out'])/2*1e6
    T=inputValues['Pump_intake_T']
    O_W_ST = 0.035
    fig, ax = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)

    for VISO_test in [1,10,50,100,300,500,700,1000]:
    # for VISO_test in [1,200,300,500,800,1000]:
        if VISO_test>VISO_in*1000:
            break
        if VISO_test == 1: 
            VISO_test = 1
            VISO_test = VISO_test/1000
            HP = ESP_head (QBEM, ESP_input, QL, VISW_in, DENW_in, VISO_test, DENO_in, VISG_in, WC, 0, P, T, O_W_ST, None)
        else:
            VISO_test = VISO_test/1000
            # HP, _, _, _, _, _, _, _ = Oil_curve(QBEM=QBEM, ESP_input=ESP_input, QL=QL, VISL_in=VISL_in, DENL_in=DENL_in)
            HP = ESP_head (QBEM, ESP_input, QL, VISW_in, DENW_in, VISO_test, DENO_in, VISG_in, WC, GLR, P, T, O_W_ST, None)
            if VISO_test>0.5:
                # check point
                QL1= np.arange(1,2,1) * 86 * m3s_to_bpd/24/3600
                HP1 = ESP_head (QBEM, ESP_input, QL1, VISW_in, DENW_in, VISO_test, DENO_in, VISG_in, WC, GLR, P, T, O_W_ST, None)

        ax.plot(QL/m3s_to_bpd*24*3600, HP*0.3048/0.433*ESP_input['SN'], label='原油粘度='+str(int(VISO_test*1000))+'cp', linewidth=0.75)
        ax.set_ylim(0)

        ax.set_xlabel('排量（m3/d）', fontsize=8)
        ax.set_ylabel('扬程（m）', fontsize=8)
        ax.set_title(well_name+'井电潜泵扬程拟合验证(GOR=%d,WC=%d'%(GOR,WC)+'%)', fontsize=8)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
    
    ax.scatter([Q_water], [H_water], color = 'black', label=('实验 水1cp (%dm$^3$/d,%dm)'%(Q_water,H_water)),  marker='*', linewidth=0.3)
    ax.scatter([Q_Oil], [H_Oil], color = 'red', label=('现场 油%dcp (%dm$^3$/d,%dm)'%(VisL_target,Q_Oil,H_Oil)),  marker='*', linewidth=0.3)
    ax.plot([0, Q_Oil], [H_Oil, H_Oil], color = 'red', linewidth=0.75)
    ax.plot([Q_Oil, Q_Oil], [0, H_Oil], color = 'red', linewidth=0.75)
    ax.legend(frameon=False, fontsize=6)
    QL= np.array(Q_Oil*m3s_to_bpd/24/3600)
    VISG = gas_viscosity(inputValues['Gas_relative_density'], T, P)
    HP = ESP_head (QBEM, ESP_input, QL, VISW_in, DENW_in, VISO_in, DENO_in, VISG, WC, GLR, P, T, O_W_ST, None)
    print (QL/(m3s_to_bpd/24/3600), HP*0.3048/0.433*ESP_input['SN'], GLR)
    fig.tight_layout()
    fig.savefig('result/'+well_name+'Viscosity'+'.jpg')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = ESPNodalGUI()
    sys.exit(app.exec_())

       