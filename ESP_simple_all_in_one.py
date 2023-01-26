# -*- coding: utf-8 -*-

"""
The ESP calculation module contains two classes for computing inputs performance under single-phase water/viscous
fluid flow or gas-liquid two-phase flow conditions.

The two-phase model was originally proposed by Dr. Zhang, TUALP ABM (2013) and later revised by Zhu (2017). Only the 
simplified version is programmed.

Version:    1st, Aug, 2017
Developer:  Jianjun Zhu
"""
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib as mpl
from Pipe_model import *

G = 9.81
pi = np.pi
E1 = 10e-3   # for gas
E2 = 1e-7   # for single phase
DENW = 997.                 # water density
VISW = 1e-3                 # water viscosity
psi_to_pa = 1.013e5 / 14.7
psi_to_ft = 2.3066587368787
bbl_to_m3 = 0.15897
bpd_to_m3d = 0.159
ft_to_m = 0.3048
hp_to_kW = 0.7457

mpl.use('Qt5Agg')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# global variables

sgl_model = 'zhang_2016'
factor = 0.25    
DB_GVF_EFF = 1     
DB_NS_EFF = 2      
alphaG_crit_coef = 0.1
alphaG_crit_critical = 0.45   


CD_Gas_EFF = 0.62       
CD_Liquid_EFF = 1       
CD_INT_EFF = 1.5         
CD_GV_EFF = 1         
transition_zone = 0.01 
FTI = 3       # Original turning factor
FTD = 3       # Original turning factor

error_control_high = 100  # reduce data noise for error > error_control_high
error_control_low = -50  # reduce data noise for error < error_control_low
ABSerror_control_high = 2  # reduce data noise for error > error_control_high
ABSerror_control_low = 2  # reduce data noise for error > error_control_high
transition_zone = 0.3   # transition zone between slug and bubble, transition_zone*QBEM

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
            '#acc2d9', '#56ae57','#b2996e','#a8ff04','#69d84f','#894585','#65ab7c','#952e8f','#fcfc81','#a5a391',
            '#001146', '#a5a502', '#ceb301']
# more color please check # https://xkcd.com/color/rgb, or matplotlib code

symbols = ['X', 'o', '^', '*', 'd', 'p', 'v', 'D', '<', 's',
            '>', '*', 'h', '1', 'p', '2', '8', 'd', '_', 'X',
            '4','8', '|']
sgl_model = 'zhang_2016'
# sgl_model = 'zhu_2018'

# customize matplotlib
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['lines.linewidth'] = 0.75

plt.style.use('seaborn-ticks')
mpl.rcParams['figure.figsize'] = (4, 3)
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['markers.fillstyle'] = 'none'
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["axes.titleweight"] = "bold"
# mpl.rcParams["axes.titlecolor"] = "black"    
mpl.rcParams["axes.linewidth"] = 0.75

#xtick.labelsize:     medium  # fontsize of the tick labels

"""
Typical inputs: Q in bpd, N in rpm, angle in degree, all others in SI units
"""
#SN: stage number
#   AB		=	impeller blade surface area on both side (m2)				
#   AV		=	diffuser vane surface area on both side (m2)			
#   ASF	    =	shroud front surface area (m2)				
#   ASB	    =	shroud back surface area (m2)				
#   ADF	    =	diffuser front surface area (m2)					
#   ADB	    =	diffuser back surface area (m2)	
#   RD1     =   Diffuser inlet radius, input for sgl_calculate_jc by Jiecheng Zhang
#   RD2     =   Diffuser outlet radius, input for sgl_calculate_jc by Jiecheng Zhang
#   NS      =   specific speed based on field units
#   SL      =   Leakage gap width
#   LG      =   Leakage gap length
xxx = 0.039403/0.050013
bbb = 2800/3220
QBEM_default = {'TE2700': 9000, 'DN1750': 4000, 'GC6100': 7600, 'P100': 11000, 'Flex31': 6500}    
ESP = {'TE2700':
            {
                "R1": 0.017496,     "R2": 0.056054,     "TB": 0.00272,      "TV": 0.00448,      "RD1": 0.056,
                "RD2": 0.017496,    "YI1": 0.012194,    "YI2": 0.007835,    "VOI": 0.000016119, "VOD": 0.000011153,
                "AIW":0.00465816,   "ADW":0.003933,
                "ASF": 0.00176464,  "ASB": 0.00157452,  "AB": 0.001319,     "AV": 0.001516,     "ADF": 0.001482,
                "ADB": 0.000935,    # ASF to ADB not necessary
                "LI": 0.076,        "LD": 0.08708,      "RLK": 0.056209,    "LG": 0.00806,
                "SL": 0.00005,      "EA": 0.000254,     "ZI": 5,            "ZD": 9,            "B1": 19.5,
                "B2": 24.7,         "NS": 1600,         "DENL": 1000,       "DENG": 11.2,       "DENW": 1000,       "VISL": 0.001,
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3500,          "SGM": 0.3,
                "QL": 2700,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 1
            },
       'GC6100':
           {
                "R1": 0.027746,     "R2": 0.048013,     "TB": 0.0019862,    "TV": 0.002894,     "RD1": 0.0547,
                "RD2": 0.017517,    "YI1": 0.017399,    "YI2": 0.013716,    "VOI": 1.512E-5,    "VOD": 1.9818E-5,
                "AIW":0.00287127,   "ADW":0.0048477,
                "ASF": 8.9654E-4,   "ASB": 9.4143E-4,   "AB": 1.0333E-3,    "AV": 1.769E-3,     "ADF": 2.0486E-3,
                "ADB": 1.0301E-3,   # ASF to ADB not necessary
                "LI": 0.0529,       "LD": 0.0839,       "RLK": 4.35E-2,   "LG": 0.0015475,
                "SL": 3.81E-4,      "EA": 0.0003,       "ZI": 7,            "ZD": 8,            "B1": 33.375,
                "B2": 41.387,       "NS": 3220,         "DENL": 1000,       "DENG": 11.2,       "DENW": 1000,       "VISL": 0.001,
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3600,          "SGM": 0.3,
                "QL": 6100,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 1
           },
        # original R1 = 0.014351   TB = 0.0025896   "RLK": 6.1237E-2, 
        # new "RLK": 4.35E-2


        'Flex31':
            {
                "R1": 0.018,        "R2": 0.039403,     "TB": 0.0018875,    "TV": 0.0030065,    "RD1": 0.042062,
                "RD2": 0.018841,    "YI1": 0.015341,    "YI2": 0.01046,     "VOI": 0.000010506, "VOD": 0.000010108,
                "AIW":0.0020663,   "ADW":0.0025879,
                "ASF": 0,           "ASB": 0,           "AB": 0.0020663,    "AV": 0.0025879,    "ADF": 0, 
                "ADB": 0,           "LI": 0.04252,      "LD": 0.060315,     "RLK": 0.033179,    "LG": 0.005, 
                "SL": 0.000254,     "EA": 0.0003,       "ZI": 6,            "ZD": 8,            "B1": 21.99,        
                "B2": 57.03,        "NS": 2975,         "DENL": 1000,       "DENG": 11.2,       "DENW": 1000,       "VISL": 0.001,      
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3600,          "SGM": 0.3,
                "QL": 4000,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 1
            },
        

        'DN1750':
           {
                "R1": 1.9875E-2,    "R2": 3.5599E-2,    "TB": 1.7E-3,       "TV": 3.12E-3,      "RD1": 0.04,
                "RD2": 0.01674,     "YI1": 1.3536E-2,   "YI2": 7.13E-3,     "VOI": 6.283E-6,    "VOD": 7.063E-6,
                "AIW":0.00203005,   "ADW":0.00243763,
                "ASF": 6.8159E-04,  "ASB": 6.549E-04,   "AB": 6.9356E-04,   "AV": 7.1277E-04,   "ADF": 1.0605E-03,
                "ADB": 6.6436E-04,  # ASF to ADB not necessary
                "LI": 0.039,        "LD": 5.185E-02,    "RLK": 0.04,        "LG": 0.01,
                "SL": 0.00005,      "EA": 0.000254,     "ZI": 6,            "ZD": 8,            "B1": 20.3,
                "B2": 36.2,         "NS": 2815,         "DENL": 1000,       "DENG": 11.2,       "VISL": 0.001,
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3500,          "SGM": 0.3,
                "QL": 1750,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 3
           },

       'P100':
           {
                "R1": 0.023793966,  "R2": 0.05097,      "TB": 0.0023,       "TV": 0.00284,      "RD1": 0.052424,
                "RD2": 0.025349,    "YI1": 0.02315,     "YI2": 0.01644,     "VOI": 2.9E-5,      "VOD": 2.61E-5,
                "AIW":0.003967143,   "ADW":0.006166,
                "ASF": 0.00083,     "ASB": 0.001277143, "AB": 0.00186,      "AV": 0.00224,      "ADF": 0.002506,
                "ADB": 0.00142,     # ASF to ADB not necessary
                "LI": 0.04336,      "LD": 0.0810175,    "RLK": 0.045465,    "LG": 0.009605,
                "SL": 0.0001,       "EA": 0.000254,     "ZI": 7,            "ZD": 7,            "B1": 35.315,
                "B2": 38.17,        "NS": 3448,         "DENL": 1000,       "DENG": 11.2,       "DENW": 1000,       "VISL": 0.001,
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3600,          "SGM": 0.3,
                "QL": 9000,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 1
           }
           
       }

class SinglePhaseModel(object):
    def __init__(self, inputs, QBEM):
        self.R1 = inputs['R1']
        self.R2 = inputs['R2']
        self.RD1 = inputs['RD1']
        self.RD2 = inputs['RD2']
        self.TB = inputs['TB']
        self.TV = inputs['TV']
        self.YI1 = inputs['YI1']
        self.YI2 = inputs['YI2']
        self.VOI = inputs['VOI']
        self.VOD = inputs['VOD']
        self.AIW = inputs['AIW']
        self.ADW = inputs['ADW']
        self.LI = inputs['LI']
        self.LD = inputs['LD']
        self.RLK = inputs['RLK']
        self.LG = inputs['LG']
        self.SL = inputs['SL']
        self.EA = inputs['EA']
        self.ZI = inputs['ZI']
        self.ZD = inputs['ZD']
        self.B1 = inputs['B1'] * (pi / 180.0)
        self.B2 = inputs['B2'] * (pi / 180.0)
        self.DENL = inputs['DENL']
        self.VISL = inputs['VISL']
        self.VISW = inputs['VISW']
        self.N = inputs['N']
        self.SGM = inputs['SGM']
        self.ST = inputs['ST']
        self.NS = inputs['NS']
        self.WC = inputs['WC']
        self.SN = inputs['SN']
        self.QBEM = QBEM * bbl_to_m3 / 24.0 / 3600.0
        self.OMEGA = 2.0 * pi * self.N / 60.0
    @staticmethod
    def get_fff(re, ed):
        # friction factor used in unified model
        lo = 1000.
        hi = 3000.
        REM = re
        ED = ed
        Fl = 16.0 / REM
        Fh = 0.07716 / (np.log(6.9 / REM + (ED / 3.7)**1.11))**2.0

        if REM < lo:
            return Fl
        elif REM > hi:
            return Fh
        else:
            return (Fh * (REM - lo) + Fl * (hi - REM)) / (hi - lo)
    @staticmethod
    def get_fff_leakage(re, ed, N, RLK, VLK, LG, SL):
        '''
        N: rotational speed RPM
        RLK: leakage radius (rotational effect, assume equal to RI=R1+R2)
        VLK: axial velocity in the leakage area
        LG: leakage length
        SL: leakage width
        '''
        # friction factor based on Childs 1983 and Zhu et al. 2019 10.4043/29480-MS
        REM = re
        OMEGA = 2.0 * pi * N / 60.0     # rotational speed in rads/s
        fff = LG/SL*0.066*REM**-0.25*(1+OMEGA**2*RLK**2/4/VLK**2)**0.375
        return fff
    @staticmethod
    def emulsion(VOI, R2, VISO, VISW, DENO, DENW, WC, ST, N, Q, SN, mod='zhu'):
        """
        The model is based on Brinkman (1952) correlation and Zhang (2017, Fall)
        :param VOI: volume of impeller (m3)
        :param R2:  impeller outer radius (m)
        :param VISO:viscosity of oil (kg/m-s)
        :param VISW:viscosity of water (kg/m-s)
        :param DENO:density of oil (kg/m3)
        :param DENW:density of water (kg/m3)
        :param WC:  water cut (%)
        :param ST:  surface tension (N/m)
        :param N:   rotational speed (rpm)
        :param Q:   flow rate (m3/s)
        :param SN:  stage number (-)
        :param mod: select different model type: tualp, banjar, or zhu
        :return: miu in Pas
        """
        # E = 3.  # exponential index
        E = 4  # exponential index
        # E = 2  # exponential index
        f = N / 60.
        WC = WC / 100.
        miu_tilda = VISO / VISW
        phi_OI = miu_tilda ** (1. / E) / (1. + miu_tilda ** (1. / E))
        phi_WI = 1. - phi_OI
        phi_OE = 1. - (VISW / VISO) ** (1. / E)
        i = 0.

        get_C = lambda SN, WE, RE, ST: SN ** 0.01 * WE ** 0.1 * RE ** 0.1 / (2.5 * ST ** 0.2)

        if mod == "tualp":
            get_C = lambda SN, WE, RE, ST: (SN * WE * RE) ** 0.15 / (10 * ST ** 0.5)
        elif mod == "banjar":
            get_C = lambda SN, WE, RE, ST: (SN * WE * RE) ** 0.1 / (10 * ST ** 0.2)
        elif mod == "zhu":
            # get_C = lambda SN, WE, RE, ST: (SN * WE) ** 0.1 * RE ** 0.1 / (2.5 * ST ** 0.2)
            get_C = lambda SN, WE, RE, ST: SN ** 0.01 * WE ** 0.1 * RE ** 0.1 / (2.5 * ST ** 0.2)  # ori
            get_C = lambda SN, WE, RE, ST: SN ** 0.01 * WE ** 0.1 * RE ** 0.1 / (2.5 * ST ** 0.2)

        # find the inversion point
        St = f * VOI / Q
        for i in np.arange(10000) / 10000.:
            if i == 0:
                continue
            rouA = i * DENW + (1 - i) * DENO
            We = rouA * Q ** 2 / (ST * VOI)
            miu_M = VISW / (1 - (1 - i) * phi_OE) ** E

            # assume oil in water
            Re = rouA * Q / (VISW * 2 * R2)
            C = get_C(SN, We, Re, St)
            miu_E = VISW / (1 - (1 - i)) ** E
            miu_A_OW = C * (miu_E - miu_M) + miu_M

            # assume water in oil
            Re = rouA * Q / (VISO * 2 * R2)
            C = get_C(SN, We, Re, St)
            miu_E = VISO / (1 - i) ** E
            miu_A_WO = C * (miu_E - miu_M) + miu_M

            if np.abs(miu_A_OW - miu_A_WO) / miu_A_OW < 0.01:
                break

        if WC > i:
            # oil in water
            rouA = WC * DENW + (1 - WC) * DENO
            We = rouA * Q ** 2 / (ST * VOI)
            miu_M = VISW / (1 - (1 - WC) * phi_OE) ** E

            Re = rouA * Q / (VISW * 2 * R2)
            C = get_C(SN, We, Re, St)
            miu_E = VISW / (1 - (1 - WC)) ** E
            miu_A = C * (miu_E - miu_M) + miu_M
            miu_A_2 = WC * VISW + (1 - WC) * VISO
            if miu_A_2>miu_A:
                miu_A=miu_A_2
        else:
            # water in oil
            rouA = WC * DENW + (1 - WC) * DENO
            We = rouA * Q ** 2 / (ST * VOI)
            miu_M = VISW / (1 - (1 - WC) * phi_OE) ** E

            Re = rouA * Q / (VISO * 2 * R2)
            C = get_C(SN, We, Re, St)
            miu_E = VISO / (1 - WC) ** E
            miu_A = C * (miu_E - miu_M) + miu_M
            miu_A_2 = WC * VISW + (1 - WC) * VISO
            if miu_A_2>miu_A:
                miu_A=miu_A_2
        return miu_A
    def sgl_calculate_new(self, Q, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC):
        """
        Dr Zhang new update on previous single-phase model with new consideration on recirculation flow loss
        :param Q:   flow rate in m3/s
        :param QEM: best match flow rate in m3/s
        :param DENL: liquid density in kg/m3
        :param DENW: water density in kg/m3
        :param N:   rotational speed in rpm
        :param NS:  specific speed based on field units
        :param SGM: tuning factor
        :param SN:  stage number
        :param ST:  surface tension Nm
        :param VISL: liquid viscosity in Pas
        :param VISW: water viscosity in Pas
        :param WC:  water cut in %
        :return: HP, HE, HF, HT, HD, HRE, HLK, QLK in filed units
        """

        QLK = 0.02 * Q
        HP = 10.
        HE, HEE = 0., 0
        HFI, HFD = 0., 0
        HTI, HTD = 0., 0
        HLK = 0.
        HLKloss = 0.
        icon = 0
        HP_new, QLK_new = 1., 1.

        ABH = np.abs((HP - HP_new) / HP_new)
        ABQ = np.abs((QLK - QLK_new) / QLK_new)
        AIW = self.AIW
        ADW = self.ADW

        # needs change
        SGM = self.SGM
        OMEGA = 2.0 * pi * N / 60.0
        SGMU = 1 - 0.01 * (VISL / VISW ) **0.35 # HAIWEN  2022 for shengli project
        # SGMU = 1
        # new QBEM due to rotational speed
        QBEM = QBEM * (N / 3500)

        # check if emulsion occurs
        if WC==100:
            VISL=VISW
        elif WC > 0.:
            VISL = self.emulsion(self.VOI, self.R2, VISL, VISW, DENL, DENW, WC, ST, N, Q, SN)

        while (ABH > E2) and (ABQ > E2):

            C1M = (Q + QLK) / ((2.0 * pi * self.R1 - self.ZI * self.TB) * self.YI1)
            C2M = (Q + QLK) / ((2.0 * pi * self.R2 - self.ZI * self.TB) * self.YI2)
            U1 = self.R1 * OMEGA
            U2 = self.R2 * OMEGA
            W1 = C1M / np.sin(self.B1)
            W2 = C2M / np.sin(self.B2)
            C1 = np.sqrt(C1M ** 2 + (U1 - C1M / np.tan(self.B1)) ** 2)
            C2 = np.sqrt(C2M ** 2 + (U2 - C2M / np.tan(self.B2)) ** 2)
            CMB = QBEM / ((2.0 * pi * self.R2 - self.ZI * self.TB) * self.YI2)
            C2B = np.sqrt(CMB ** 2 + (U2 - CMB / np.tan(self.B2)) ** 2)

            # Euler head
            # HE=(U2**2-U1**2+W1**2-W2**2+C2**2-C1**2)/(2.0*G)					        # with pre-rotation
            HE = (U2 ** 2 * SGMU - U2 * C2M / np.tan(self.B2)) / G                             # without pre-rotation

            # head loss due to recirculation
            if (Q + QLK) < QBEM:
                '''zhang 2016'''
                VSH = U2 * (QBEM - (Q + QLK)) / QBEM
                C2F = C2B * (Q + QLK) / QBEM
                DC = 2.0 * pi * self.R2 * np.sin(self.B2) / self.ZI
                REC = DENL * VSH * DC / VISL
                C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)
                C2E = C2F
                HEE = HE + (C2E ** 2 - C2 ** 2) / (2.0 * G)
            
            else:
                '''zhang 2016'''
                VSH = U2 * (Q + QLK - QBEM) / QBEM
                C2F = C2B * (Q + QLK) / QBEM
                C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)
                C2E = C2F + SGM * (C2P - C2F) * (Q + QLK - QBEM) / QBEM     # new development by Dr Zhang
                HEE = HE + (C2E ** 2 - C2 ** 2) / (2.0 * G)
                '''ori recirculation model'''
                # VSH = U2 * (Q + QLK - QBEM) / QBEM
                # C2F = C2B * (Q + QLK) / QBEM
                # DC = 2.0 * pi * self.R2 * np.sin(self.B2) / self.ZI
                # REC = DENL * VSH * DC / VISL
                # SGM = (VISW / VISL) ** 0.1 / (10.0 + 0.02 * REC ** 0.2)
                # C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)
                # # C2E = C2F + SGM * (C2P - C2F) * (Q + QLK - QBEM) / QBEM
                # C2E = (C2**2+C2F**2-VSH**2)/2/C2F
                # HEE = HE + (C2E ** 2 - C2 ** 2) / (2.0 * G)
            # friction loss
            AI = self.VOI / self.LI
            AD = self.VOD / self.LD
            VI = (Q + QLK) / self.ZI / AI
            VD = Q / self.ZD / AD
            DI = 4.0 * self.VOI / AIW
            DD = 4.0 * self.VOD / ADW
            REI = DENL * (W1 + W2) * DI / VISL / 2.0
            RED = DENL * VD * DD / VISL
            EDI = self.EA / DI
            EDD = self.EA / DD
            FFI = self.get_fff(REI, EDI)
            FFD = self.get_fff(RED, EDD)
            #HFI = 2.5 * 4.0 * FFI * (W1 + W2) ** 2 * self.LI / (8.0 * G * DI)
            HFI = 2.5 * 4.0 * FFI * (VI) ** 2 * self.LI / (2.0 * G * DI)    # modified by Haiwen zhu (it is used in gl model)
            HFD = 2.5 * 4.0 * FFD * VD ** 2 * self.LD / (2.0 * G * DD)

            # turn loss
            # FTI = 3.0
            # FTD = 3.0
            HTI = FTI * VI ** 2 / (2.0 * G)          # ori
            HTD = FTD * VD ** 2 / (2.0 * G)          # ori
            # HTI = FTI  * ( (VISL / VISW) ** 0.05 ) * VI ** 2 / (2.0 * G)                #zimo HTI = FTI * VI ** 2 / (2.0 * G)   SGM = (VISW / VISL) ** 0.1 / (10.0 + 0.02 * REC ** 0.25)
            # HTD = FTD  * ( (VISL / VISW) ** 0.05 ) * VD ** 2 / (2.0 * G)              #zimo HTD = FTD * VD ** 2 / (2.0 * G)

            # new pump head
            HP_new = HEE - HFI - HFD - HTI - HTD

            # calculate leakage
            UL = self.RLK * OMEGA
            HIO = HEE - HFI - HTI
            HLK = HIO - (U2 ** 2 - UL ** 2) / (8.0 * G)
            if HLK >= 0:
                VL = np.abs(QLK) / (2.0 * pi * self.RLK * self.SL)
                REL = DENL * VL * self.SL / VISL
                EDL = 0.0
                FFL = self.get_fff(REL, EDL)
                FFL = self.get_fff_leakage(REL,EDL,N, (self.R1+self.R2)/2, VL, self.LG, self.SL)        # by Haiwen Zhu
                VL = np.sqrt(2.0 * G * HLK / (1.5 + 4.0 * FFL * self.LG / self.SL))
                QLK_new = 2.0 * pi * self.RLK * self.SL * VL
            else:
                VL = np.abs(QLK / (2.0 * pi * self.RLK * self.SL))
                REL = DENL * VL * self.SL / VISL
                EDL = 0.
                FFL = self.get_fff(REL, EDL)
                FFL = self.get_fff_leakage(REL,EDL,N, (self.R1+self.R2)/2, VL, self.LG, self.SL)        # by Haiwen Zhu
                VL = np.sqrt(2.0 * G * np.abs(HLK) / (1.5 + 4.0 * FFL * self.LG / self.SL))
                QLK_new = -2.0 * pi * self.RLK * self.SL * VL

            ABQ = np.abs((QLK_new - QLK) / QLK_new)
            QLK = QLK_new
            # HLKloss = 20/2/G*(QLK/self.ZD/AI)**2        # by Haiwen Zhu
            # HLKloss = 0.25/2/G*(VL)**2        # by Haiwen Zhu
            HLKloss = 0.25/2/G*(QLK/AI)**2        # by Haiwen Zhu
            HP_new -= HLKloss
            ABH = np.abs((HP_new - HP) / HP_new)
            HP = HP_new

            if icon > 500:
                break
            else:
                icon += 1

        # return pressure in psi, flow rate in bpd
        HP = HP * G * DENL / psi_to_pa
        HE = HE * G * DENL / psi_to_pa
        HEE = HEE * G * DENL / psi_to_pa
        HF = (HFI + HFD) * G * DENL / psi_to_pa
        HT = (HTI + HTD) * G * DENL / psi_to_pa
        HD = (HFD + HTD) * G * DENL / psi_to_pa
        HRE = HE - HEE
        HLKloss = HLKloss * G * DENL / psi_to_pa
        QLK = QLK * 24.0 * 3600.0 / bbl_to_m3
        return HP, HE, HF, HT, HD, HRE, HLKloss, QLK
# two-phase mechanistic model class
class GasLiquidModel(SinglePhaseModel):
    def __init__(self, inputs, QBEM):
        super(GasLiquidModel, self).__init__(inputs, QBEM)
        self.DENG = inputs['DENG']
        self.VISG = inputs['VISG']
        self.QL = inputs['QL']
        self.QG = inputs['QG']
        self.RI = (self.R1 + self.R2) / 2.0
        self.YI = (self.YI1 + self.YI2) / 2.0
    @staticmethod
    def CD_Cal(VSR, DENL, VISL, N, DB):
        # CD-Legendre & Magnaudet (1998), Clift et al. (1978), Rastello et al. (2011)
        REB = DENL * VSR * DB / VISL
        if VSR == 0: VSR = 1e-6    # eliminate error
        SR = DB * (2.0 * np.pi * N / 60.0) / VSR
        CD = 24.0 / REB * (1.0 + 0.15 * REB**0.687) * (1.0 + 0.55 * SR**2.0)

        return CD
    def CD_Cal_Combine(self, VSR, DENL, VISL, N, DB, Q, QLK, QG, GV, FGL, alphaG2):
        if FGL == 'BUB':
            '''bubble flow'''
            REB = DENL * VSR * DB / VISL
            if VSR == 0: VSR = 1e-6    # eliminate error
            SR = DB * (2.0 * np.pi * N / 60.0) / VSR
            CD = 24.0 / REB * (1.0 + 0.15 * REB**0.687) * (1.0 + 0.55 * SR**2.0)
        else:
            REB = DENL * VSR * DB / VISL
            if VSR == 0: VSR = 1e-6    # eliminate error
            SR = DB * (2.0 * np.pi * N / 60.0) / VSR
            CD1 = 24.0 / REB * (1.0 + 0.15 * REB**0.687) * (1.0 + 0.55 * SR**2.0)

            C1M_L = (Q + QLK) / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI)
            W1_L = C1M_L / np.sin((self.B1+self.B2/2))
            C1M_G = QG / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI)
            W1_G = C1M_G / np.sin((self.B1+self.B2/2))
            W1L = W1_L / (1 - GV)
            W1G = W1_G / GV
            mium1 = (1 - GV) * VISL + GV * self.VISG
            DENM1 = (1 - GV) * DENL + GV * self.DENG
            CD2 = (12. * mium1 * (CD_INT_EFF*9.13e7 * GV ** CD_GV_EFF / W1_G ** CD_Gas_EFF * W1_L ** CD_Liquid_EFF ) / (np.abs(W1G - W1L) * DENM1) * (N/3600)**2)*DB
            CD1 = CD1*((1-GV)*1)
            CD2 = CD2*((GV-alphaG2)*80)
            CD = (CD1+CD2)/(((1-GV)*1)+((GV-alphaG2)*80))
        return CD
    def get_DB(self, Q, HP, N, GV):
        if HP*Q<0.000001: 
            HP=1
            Q=0.000001    # eliminate convergence and error problem
        DB = factor*6.034 * GV**DB_GVF_EFF * (self.ST / self.DENL)**0.6 * (HP * Q *(N/3500* 0.05/self.R2)**DB_NS_EFF/ (self.DENL * self.ZI * self.VOI))**(-0.4) * (self.DENL / self.DENG)**0.2  #original 2021-10
        DBMAX = DB/0.43
        return DB, DBMAX
    def get_lambda_c1(self, Q, HP, N):
        # critical bubble size in turbulent flow (Barnea 1982)
        DBCrit = 2.0 * (0.4 * 0.073 / (self.DENL - self.DENG) / ((2.0 * pi * N / 60.0)**2 * self.RI))**0.5
        if HP*Q<0.000001: 
            HP=1
            Q=0.000001    # eliminate convergence and error problem
        LambdaC1 = DBCrit / (6.034 / 0.6 * (self.ST / self.DENL)**0.6 * (HP * Q / (self.DENL * self.ZI * self.VOI)) **  \
                             (-0.4) * (self.DENL / self.DENG)**0.2)
        return LambdaC1
    def get_lambda_c2(self, Q, HP, N):
        alphaG = 0.5
        VSR = 0.1
        LambdaC2 = 0.1
        ABV = 1.0
        icon = 0
        relax = 0.1
        AIW = self.AIW
        DI = 4.0 * self.VOI / AIW
        EDI = self.EA / DI
        AI = self.VOI / self.LI
        while ABV > E1:
            alphaG_crit = 0.5 - alphaG_crit_critical*(np.exp(-(N/3500.0)))**alphaG_crit_coef   #Fotran code selection
            '''bubble shape effect'''    
            if icon > 1000:
                break
            else:
                icon += 1
            DB, DBMAX = self.get_DB(Q,HP,N,alphaG)
            REB = self.DENL * VSR * DB / self.VISL
            if VSR < 0.001: VSR = 0.001     # eliminate error
            SR = DB * (2.0 * pi * N / 60.0) / VSR
            # original CD
            if REB <= 50:
                if REB < 0.001:
                    REB = 0.001
                if SR < 0.001:
                    SR = 0.001
                CD = 24.0 / REB * (1.0 + 0.15 * REB**0.687) * (1.0 + 0.3 * SR**2.5)
            else:
                CD = 24.0 / REB * (1.0 + 0.15 * REB**0.687) * (1.0 + 0.55 * SR**2.0)
            CD = self.CD_Cal_Combine(VSR, self.DENL, self.VISL, N, DB, 0, 0, 0, 0, 'BUB', 0)
            VSR = np.sqrt(4.0 * DB * (self.DENL - self.DENG) * self.RI / (3.0 * CD * self.DENL)) * (2.0 * pi * N / 60.0)
            RS = 2*VSR * (2.0 * pi * self.RI - self.ZI * self.TB) * self.YI / Q
            if RS >= 1: RS = 0.9999
            elif RS <=0: RS = 0.0001        # eliminate error and non convergence problem
            alphaG = (RS - 1.0 + np.sqrt((1.0 - RS)**2 + 4.0 * RS * LambdaC2)) / (2.0 * RS)
            ABV = np.abs((alphaG - alphaG_crit) / alphaG_crit)
            if alphaG > alphaG_crit:
                LambdaC2 *=0.9
            else:
                LambdaC2 *=1.1
            if LambdaC2 < 0:
                LambdaC2 = -LambdaC2
                return LambdaC2
            if icon > 1e5:
                return LambdaC2
            else:
                icon += 1
        return LambdaC2
    def get_lambda_c3(self, Q, N):
        LambdaC3 = 0.1
        ANG = -90.0 * pi / 180.0
        FIC = 0.0142
        AI = self.VOI / self.LI
        BI = (self.B1 + self.B2) / 2.0
        AIW = self.AIW
        DI = 4.0 * self.VOI / AIW
        EDI = self.EA / DI
        GC = (2.0 * pi * N / 60.0)**2 * self.RI * np.sin(BI)
        VSL = Q / self.ZI / AI
        CS = (32.0 * np.cos(ANG)**2 + 16.0 * np.sin(ANG)**2) * DI
        CC = 1.25 - 0.5 * np.abs(np.sin(ANG))
        if CS > self.LI:
            CS = self.LI

        V24 = 19.0 * VSL / 6.0
        # guess a VSG
        VSG = VSL
        VM = VSL + VSG
        HLS = 1.0 / (1.0 + np.abs(VM / 8.66)**1.39)
        if HLS < 0.24:
            HLS = 0.24
        FE = 0.0
        HLF = VSL / VM
        VC = VSG / (1.0 - HLF)
        VCN = 0
        VF = VSL / HLF 
        FI = FIC
        REMX = 5000.0
        RESL = self.DENL * VSL * DI / self.VISL
        ABCD = V24**2
        ABU = np.abs((VCN - VC) / VC)
        icon = 0
        while ABU > E1:
            if icon > 1000:
                break
            else:
                icon += 1
            # Entrainement fraction based on Oliemans et al's (1986)
            WEB = self.DENG * VSG * VSG * DI / self.ST
            FRO = np.sqrt(G * DI) / VSG
            RESG = self.DENG * VSG * DI / self.VISG
            CCC = 0.003 * WEB**1.8 * FRO**0.92 * RESL**0.7 * (self.DENL / self.DENG)**0.38 * (self.VISL / self.VISG)**0.97 / RESG**1.24
            FEN = CCC / (1.0 + CCC)
            if FEN > 0.9:
                FEN = 0.9
            FE = (FEN + 9.0 * FE) / 10.0
            # Translational velocity based on Nicklin (1962), Bendiksen (1984) and Zhang et al. (2000)
            if REMX < 2000.0:
                VAV = 2.0 * VM
            elif REMX > 2000.0:
            #elif REMX > 4000.0: #Fotran code
            #    VAV = 1.2*VM    #Fotran code
                VAV = 1.3 * VM
            else:
                VAV = (2.0 - 0.7 * (REMX - 2000.0) / 2000.0) * VM
            VT = VAV + (0.54 * np.cos(ANG) + 0.35 * np.sin(ANG)) * np.sqrt(GC * DI * np.abs(self.DENL - self.DENG) / self.DENL)
            if VT < 0:
                VT = np.abs(VT)
            HLFN = ((HLS * (VT - VM) + VSL) * (VSG + VSL * FE) - VT * VSL * FE) / (VT * VSG)
            if HLFN < 0.0:
                HLFN = np.abs(HLFN)

            if HLFN > 1.0:
                HLFN = 1.0 / HLFN
            ABHLF = np.abs(HLF-HLFN)/HLFN
            HLF = HLFN
            HLC = (1.0 - HLF) * VSL * FE / (VM - VSL * (1.0 - FE))
            if HLC < 0.0:
                HLC = 0.0
            # Taylor bubble geometries
            DeltaL = DI / 2.0 * (1. - np.sqrt(1.0 - HLF))
            AC = pi * (DI - 2.0 * DeltaL)**2 / 4.0
            AF = pi * DeltaL * (DI - 1.0 * DeltaL)
            SI = pi * (DI - 2.0 * DeltaL)
            SF = pi * DI
            DF = 4.0 * DeltaL * (DI - DeltaL) / DI
            DC = DI - 2.0 * DeltaL
            THF = DeltaL
            VFN = VSL * (1.0 - FE) / HLF
            VF = (VFN + 9.0 * VF) / 10.0
            # Reynolds number to get friction factor
            DENC = (self.DENL * HLC + self.DENG * (1.0 - HLF - HLC)) / (1.0 - HLF)
            REF = np.abs(self.DENL * VF * DF / self.VISL)
            REC1 = np.abs(self.DENG * VC * DC / self.VISG)
            FF = self.get_fff(REF, EDI)
            FIM = self.get_fff(REC1, THF / DI)
            # Interfacial friction factor based on Ambrosini et al. (1991)
            REG = np.abs(VC * self.DENG * DI / self.VISG)
            WED = self.DENG * VC * VC * DI / self.ST
            FS = 0.046 / REG**0.2
            SHI = np.abs(FI * self.DENG * (VC - VF)**2 / 2.0)
            THFO = THF * np.sqrt(np.abs(SHI / self.DENG)) / self.VISG
            FRA = FS * (1.0 + 13.8 * (THFO - 200.0 * np.sqrt(self.DENG / self.DENL)) * WED**0.2 / REG**0.6)
            FRA1 = self.get_fff(REC1, THF / DI)
            if FRA > FRA1:
                FRA = FRA1
            FIN = FRA
            if FIN > 0.5:       #Commentted in Fotran code
                FIN = 0.5
            FI = (FIN + 9.0 * FI) / 10.0
            ABCDN = (-(self.DENL - DENC) * GC + SF * FF * self.DENL * VF * VF / (2.0 * AF)) * 2 / (SI * FI * self.DENG * (1.0 / AF + 1.0 / AC))   # Original
            ABCD = (ABCDN + 9.0 * ABCD) / 10.0
            if ABCD < 0.:
                VCN = VC * 0.9
            else:
                VCN = np.sqrt(ABCD) + VF
            if VCN < V24:
                VCN = V24
            ABU = np.abs((VCN - VC) / VC)
            VC = 0.5*VCN+0.5*VC
            VSG = VC * (1.0 - HLF) - VSL * FE
            VSG = VC * (1.0 - HLF) * (1 - FE)
            if VSG < 0.0:
                VSG = -VSG
            VM = VSL + VSG
            DPEX = (self.DENL * (VM - VF) * (VT - VF) * HLF + DENC * (VM - VC) * (VT - VC) * (1.0 - HLF)) * DI / CS / 4.0
            REMX = np.abs(DI * VM * self.DENL / self.VISL)
            FM = self.get_fff(REMX, EDI)
            DPSL = FM * self.DENL * VM * VM / 2.0
            DPAL = DPSL + DPEX
            if REMX < 5000.0:
                DPAL = DPAL * np.sqrt(REMX / 5000.0)
            AD = DPAL / (3.16 * CC * np.sqrt(self.ST * np.abs(self.DENL - self.DENG) * GC))
            HLSN = 1.0 / (1.0 + AD)
            if HLSN < 0.24:
                HLSN = 0.24
            HLS = HLSN
            LambdaC3 = VSG / (VSG + VSL)
        return LambdaC3
    def gv_Barrios(self, DENL, DENG, GF, N, Q, QLK, R1, RI, ST, TB, VISL, YI, ZI):
        """
        Use Barrios (2007) drag force coefficient, which is the original form Dr Zhang's development
        :param DENL: liquid density in m3/s
        :param DENG: gas density in m3/s
        :param GF: gas volumetric fraction at pump inlet
        :param N: rotatonal speed in rpm
        :param Q: liquid flow rate in m3/s
        :param QLK: leakage flow rate in m3/s
        :param R1: radius of impeller at inlet in m
        :param RI: radius of impeller in m
        :param ST: interfacial tension in N/m
        :param TB: impeller blade thickness in m
        :param VISL: liquid viscosity in Pas
        :param YI: impeller outlet height
        :param ZI: impeller blade number
        :return: GV (in-situ gas void fraction), CD_to_rb(CD over rb)
        """
        GV, DB= GF, 0.3
        CD = 0.
        ABV = 1.
        VSR = 0.1
        counter = 0
        OMEGA = 2.0 * pi * N / 60.0
        AIW = self.AIW
        DI = 4.0 * self.VOI / AIW
        EDI = self.EA / DI
        AI = self.VOI / self.LI
        while ABV > E1:
            counter += 1
            if counter > 10000:
                return GV, CD/DB
            DB = 3.0 * 0.0348 * N ** 0.8809 * GV ** 0.25 * (ST / DENL) ** 0.6 / (N ** 3 * R1 ** 2) ** 0.4
            REB = DENL * VSR * DB / VISL
            if REB <= 0.1:
                REB = 0.1
            # Drag coefficient of gas bubble in radial direction
            YY = 0.00983 + 389.9 * REB / N ** 2
            CD = (24.0 + 5.48 * (REB * YY) ** 0.427 + 0.36 * REB * YY) / (REB * YY)
            VSRN = np.sqrt(4.0 * DB * (DENL - DENG) * RI / (3.0 * CD * DENL)) * OMEGA
            ABV = np.abs((VSRN - VSR) / VSR)
            VSR = VSRN
            RS = VSR * (2.0 * pi * RI - ZI * TB) * YI / (Q + QLK)
            if GV < 0.0:
                GV = 0.0
            else:
                GV = (RS - 1.0 + np.sqrt((1.0 - RS) ** 2 + 4.0 * RS * GF)) / (2.0 * RS)
        return GV, CD, DB, VSR
    def gv_zhu(self, DENL, DENG, GF, HP, N, Q, QLK, RI, ST, TB, VISL, VOI, YI, ZI):
        """
        Bubble flow model based on Zhu (2017)
        :param DENL: liquid density in m3/s
        :param DENG: gas density in m3/s
        :param GF: gas volumetric fraction at pump inlet
        :param HP: single-phase pump pressure increment in pa
        :param N: rotatonal speed in rpm
        :param Q: liquid flow rate in m3/s
        :param QLK: leakage flow rate in m3/s
        :param RI: radius of impeller in m
        :param ST: interfacial tension in N/m
        :param TB: impeller blade thickness in m
        :param VISL: liquid viscosity in Pas
        :param VOI: impeller volume in m3
        :param YI: impeller outlet height
        :param ZI: impeller blade number
        :return: GV (in-situ gas void fraction), CD_to_rb(CD over rb)
        """
        GV, DBMAX = GF, 0.3
        CD = 0.
        ABV = 1.
        VSR = 0.1
        counter = 0
        OMEGA = 2.0 * pi * N / 60.0
        while ABV > E1:
            counter += 1
            if counter > 10000:
                return GV, CD/DBMAX
            DB, DBMAX = self.get_DB(Q+QLK, HP,N, GF)
            REB = DENL * VSR * DB / VISL
            SR = DB * OMEGA / VSR
            if REB <= 1:
                REB = 1
            if REB < 50.0:
                CD = 24.0 / REB * (1.0 + 0.15 * REB ** 0.687) * (1.0 + 0.3 * SR ** 2.5)
            else:
                CD = 24.0 / REB * (1.0 + 0.15 * REB ** 0.687) * (1.0 + 0.55 * SR ** 2.0)
            VSRN = np.sqrt(4.0 * DB * (DENL - DENG) * RI / (3.0 * CD * DENL)) * OMEGA
            ABV = np.abs((VSRN - VSR) / VSR)
            VSR = VSRN
            RS = VSR * (2.0 * pi * RI - ZI * TB) * YI / (Q + QLK)
            if GV < 0.0:
                GV = 0.0
            else:
                GV = (RS - 1.0 + np.sqrt((1.0 - RS) ** 2 + 4.0 * RS * GF)) / (2.0 * RS)
            return GV, DB
    @staticmethod # with this statement, we can use self variable and function
    def DBFLOW(GF):
        GV = GF
        return GV
    def BUFLOW(self,DENL,DENG,GF,HP,N,Q,RI,ST,VISL,VISG,VOI,YI,ZI,QLK,TB):
        """
        # Bubble flow model based on Zhu (2017)
        :param DENL: liquid density in m3/s
        :param DENG: gas density in m3/s
        :param GF: gas volumetric fraction at pump inlet
        :param HP: single-phase pump pressure increment in pa
        :param N: rotatonal speed in rpm
        :param Q: liquid flow rate in m3/s
        :param QLK: leakage flow rate in m3/s
        :param RI: radius of impeller in m
        :param ST: interfacial tension in N/m
        :param TB: impeller blade thickness in m
        :param VISL: liquid viscosity in Pas
        :param VOI: impeller volume in m3
        :param YI: impeller outlet height
        :param ZI: impeller blade number
        :return: GV (in-situ gas void fraction), CD_to_rb(CD over rb)
        """
        VSR = 0.1
        ABV = 1.0
        GV  = 0.005
        RS = 0.0
        VSRN = 0.0
        O = 2.0 * np.pi * N / 60.0
        icon=0
        while(ABV > E1):
            if icon > 1000:
                GV = GF
                break
            else:
                icon += 1
            DB, DBMAX = self.get_DB(Q, HP,N, GV)
            CD = self.CD_Cal_Combine(VSR, self.DENL, self.VISL, N, DB, 0, 0, 0, 0, 'BUB', 0)
            VSRN    = np.sqrt(4.0*DB*(DENL-DENG)*RI/(3.0*CD*DENL))*O
            ABV     = np.abs((VSRN-VSR)/VSR)
            VSR     = (VSRN + 9.0*VSR)/10.0
            RS      = VSR*(2.0*np.pi*RI-ZI*TB)*YI/(Q+QLK)
            if(GF < 0.0): 
                GV  = 0.0001
            else:
                GV  = (RS-1.0+np.sqrt((1.0-RS)**2+4.0*RS*GF))/(2.0*RS)
        return GV, CD, DB, VSR
    def ITFLOW_HWZ(self,AI,B1,B2,DI,EDI,DENL,DENG,LI,N,RI,VISL,VISG,ST,ZI,Q,QG,QLK,YI,TB,HP):
        # Based on Sun drag coefficient
        VSR = 0.1
        ABV = 1.0
        GV  = 0.5
        REB = 0.0
        RS = 0.0
        SR = 0.0
        VSRN = 0.0
        O = 2.0 * np.pi * N / 60.0
        icon=0
        AIW = self.AIW
        DI = 4.0 * self.VOI / AIW
        EDI = self.EA / DI
        AI = self.VOI / self.LI
        GF = QG / (Q + QLK + QG)
        CD = 0.1
        counter = 0
        OMEGA = 2.0 * pi * N / 60.0
        C1M_L = (Q + QLK) / ((2.0 * pi * RI - ZI * TB) * YI)
        W1_L = C1M_L / np.sin((B1+B2/2))
        C1M_G = QG / ((2.0 * pi * RI - ZI * TB) * YI)
        W1_G = C1M_G / np.sin((B1+B2/2))
        DB = 0.00001
        while(ABV > E1):
            if icon > 1000:
                GV = GF
                break
            else:
                icon += 1
            DB, DBMAX = self.get_DB(Q, HP,N, GV)    # before 2021, which one is better?
            REB = DENL*VSR*DB/VISL
            W1L = W1_L / (1 - GV)
            W1G = W1_G / GV
            mium1 = (1 - GV) * VISL + GV * VISG
            DENM1 = (1 - GV) * DENL + GV * DENG            
            CD = (12. * mium1 * (CD_INT_EFF*9.13e7 * GV ** CD_GV_EFF / W1_G ** CD_Gas_EFF * W1_L ** CD_Liquid_EFF ) / (np.abs(W1G - W1L) * DENM1) * (N/3600)**2)*DB
            SR  = DB*O/VSR
            VSRN    = np.sqrt(4.0 * DB * (DENL - DENG) * RI / (3.0 * CD * DENL)) * OMEGA
            ABV     = np.abs((VSRN-VSR)/VSR)
            VSR     = (VSRN + 9.0*VSR)/10.0
            RS      = VSR*(2.0*np.pi*RI-ZI*TB)*YI/(Q+QLK)
            if(GF < 0.0): 
                GV  = 0.0001
            else:
                GV  = (RS-1.0+np.sqrt((1.0-RS)**2+4.0*RS*GF))/(2.0*RS)
        return GV, CD, DB, VSR
    def Combine_flow_cal(self,DI,DENL,DENG,N,RI,VISL,ZI,Q,QG,QLK,YI,TB,HP,FGL):
        GF = QG / (Q + QLK + QG)
        DB = 0.00001
        OMEGA = 2.0 * pi * N / 60.0
        icon=0
        AIW = self.AIW
        DI = 4.0 * self.VOI / AIW
        EDI = self.EA / DI
        AI = self.VOI / self.LI
        REB = 0.0
        RS = 0.0
        SR = 0.0
        VSRN = 0.0
        VSRN1 = 0.0
        ABV = 1.0
        VSR = 0.1
        VSR1 = 0.1
        '''bubble shape effect'''
        if FGL == 'BUB':
            GV  = 0.005
        else:
            GV  = 0.2
        while(ABV > E1/1):
            if icon > 1000:
                GV = GF
                break
            else:
                icon += 1
            '''bubble shape effect'''
            alphaG_crit = 0.5 - alphaG_crit_critical*(np.exp(-(N/3500.0)))**alphaG_crit_coef   #Fotran code selectio
            alphaG_crit = (1-transition_zone)* alphaG_crit
            if GV < alphaG_crit:
                FGL = 'BUB'
            else:
                FGL = 'INT'
            DB, DBMAX = self.get_DB(Q, HP,N, GV)    # before 2021, which one is better?
            REB = DENL*VSR*DB/VISL
            CD = self.CD_Cal_Combine(VSR, DENL, VISL, N, DB, Q, QLK, QG, GV, FGL, alphaG_crit)
            VSRN    = np.sqrt(4.0*DB * (DENL - DENG) * RI / (3.0 * CD * DENL)) * OMEGA
            ABV     = np.abs((VSRN-VSR)/VSR)
            VSR     = (VSRN + 9.0*VSR)/10.0
            RS      = VSR*(2.0*np.pi*RI-ZI*TB)*YI/(Q+QLK)
            
            if(GF < 0.0): 
                GV  = 0.0001
            else:
                GV  = (RS-1.0+np.sqrt((1.0-RS)**2+4.0*RS*GF))/(2.0*RS)
        return GV, CD, DB, VSR, FGL
    def gv_zhu_flowpattern(self, GF, DENL, DENG, VISL, VISG, HP, N, QL, QG, QLK, ST, AI, DI, EDI, DENW):
        '''new'''
        GVC2 = self.get_lambda_c2(QL+QLK,HP,N)
        if GF < GVC2:
            FGL = 'BUB'
        else:
            FGL = 'INT'
        GV, CD, DB, VSR, FGL= self.Combine_flow_cal(DI,DENL,DENG,N,self.RI,VISL,self.ZI,QL,QG,QLK,self.YI,self.TB,HP,FGL)
        return GV, CD, DB, VSR
    def gl_calculate_new(self, QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW, WC, flg):
        """
        Calcualte gas-liquid performance of ESP
        :param QL:  liquid flow rate in m3/s
        :param QG:  gas flow rate m3/s
        :param QEM: best match flow rate in m3/s
        :param flg: 'Z': Zhu model; 'S': Sun model; 'B': Barrios model
        :param DENG: gas density in kg/m3
        :param DENL: liquid density in kg/m3
        :param DENW: water density in kg/m3
        :param N:   rotational speed in rpm
        :param NS:  specific speed based on field units
        :param SGM: tuning factor
        :param SN:  stage number
        :param ST:  surface tension Nm
        :param VISG: gas viscosity in Pas
        :param VISL: liquid viscosity in Pas
        :param VISM: water viscosity in Pas
        :param WC:  water cut in %
        :return: PP, PE, PF, PT, PD, PRE, PLK, QLK, GV in field units
        """
        FGL = 'Other'   # flow pattern
        # run single-phase calculation to initialize
        HP, HE, HF, HT, HD, HRE, HLK, QLK = self.sgl_calculate_new(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW,
                                                                   WC)  
        if HP < 0:
            HP = HP
        # convert filed units
        PP = HP * psi_to_pa
        # QG = 
        GF = QG / (QL + QG)
        icon = 0
        ABP = 1.
        PE, PEE = 0., 0
        PFI, PFD = 0., 0
        PTI, PTD = 0, 0
        GV = 0.
        PLK = 0.
        HLKloss = 0.
        QLK = 0.02 * QL
        OMEGA = 2.0 * pi * N / 60.0
        AIW = self.AIW
        ADW = self.ADW
        AI = self.VOI / self.LI
        AD = self.VOD / self.LD
        DI = 4.0 * self.VOI / AIW
        DD = 4.0 * self.VOD / ADW
        EDI = self.EA / DI
        EDD = self.EA / DD
        DEND = DENL * (1.0 - GF) + DENG * GF
        VISD = VISL * (1.0 - GF) + VISG * GF
        self.DENG = DENG
        CF = 0
        # new QBEM due to liquid viscosity
        # SGMU = 1 - np.sqrt(np.sin(self.B2)) / self.ZI ** (1.5* (3448 / NS) ** 0.4)* (VISL / VISW ) **0.01 # BASED ON ZIMO
        SGMU = 1 - 0.01 * (VISL / VISW ) **0.35 # HAIWEN  2022 for shengli project
        QBEM = QBEM * (N / 3600) * (VISL / VISW) ** (0.01 * (3448 / NS) ** 4)   #original
        # check if emulsion occurs
        if WC > 0.:
            VISL = self.emulsion(self.VOI, self.R2, VISL, VISW, DENL, DENW, WC, ST, N, QL, SN)
        while ABP > E1 or ABQ >E1:
            VI = (QL + QG + QLK) / self.ZI / AI     # add QG by Haiwen Zhu
            VD = (QL + QG) / self.ZD / AD       # add QG by Haiwen Zhu
            C1M = (QL + QG + QLK) / ((2.0 * pi * self.R1 - self.ZI * self.TB) * self.YI1)        # add QG by Haiwen Zhu
            C2M = (QL + QG + QLK) / ((2.0 * pi * self.R2 - self.ZI * self.TB) * self.YI2)        # add QG by Haiwen Zhu
            U1 = self.R1 * OMEGA
            U2 = self.R2 * OMEGA
            W1 = C1M / np.sin(self.B1)
            W2 = C2M / np.sin(self.B2)
            C1 = np.sqrt(C1M ** 2 + (U1 - C1M / np.tan(self.B1)) ** 2)
            C2 = np.sqrt(C2M ** 2 + (U2 - C2M / np.tan(self.B2)) ** 2)
            CMB = QBEM / ((2.0 * pi * self.R2 - self.ZI * self.TB) * self.YI2)
            C2B = np.sqrt(CMB ** 2 + (U2 - CMB / np.tan(self.B2)) ** 2)
            HE = (U2 ** 2 * SGMU - U2 * C2M / np.tan(self.B2)) / G
            if GF <= E1/1000000:
                PP = HP * psi_to_pa * (DENG * GF + DENL * (1 - GF)) / DENL
                GV = GF
                CD = 1
                VSR = 1
                DB = 1e-10
                FGL = 'D-B'
                break
            elif GF >= (1-E1/1000000):
                PP = HP * psi_to_pa * (DENG * GF + DENL * (1 - GF)) / DENL
                GV = GF
                CD = 1
                VSR = 1
                DB = 1
                FGL = 'ANN'
                break
            if flg == 'Z':
                GV, DB = self.gv_zhu(DENL, DENG, GF, HP*psi_to_pa, N, QL, QLK, self.RI, self.ST, self.TB, VISL, self.VOI, self.YI, self.ZI)
            elif flg == 'B':
                GV, _, DB, _= self.gv_Barrios(self.DENL, DENG, GF, N, QL, QLK, self.R1, self.RI, ST, self.TB, VISL, self.YI, self.ZI)
            elif flg == 'F':
                GV, CD, DB, VSR = self.gv_zhu_flowpattern(GF, DENL, DENG, VISL, VISG, HP*psi_to_pa, N, QL, QG, QLK, ST, AI, DI, EDI, DENW)
            elif flg == 'H':
                GV = self.DBFLOW(GF)
            DENI = DENL * (1.0 - GV) + DENG * GV
            PE = HE * DENI * G
            VLR = (QL + QLK) / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI * (1.0 - GV))
            VGR = QG / ((2.0 * pi * self.RI - self.ZI * self.TB) * self.YI * GV)
            if (QL + QLK) <= QBEM:
                VSH = U2 * (QBEM - (QL + QLK)) / QBEM
                C2F = C2B * (QL + QLK) / QBEM
                DC = 2.0 * pi * self.R2 * np.sin(self.B2) / self.ZI
                REC1 = DENI * VSH * DC / VISL
                C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)
                C2E = C2F
                PEE = PE + DENI * (C2E ** 2 - C2 ** 2) / 2.0
            else:
                VSH = U2 * (QL + QLK - QBEM) / QBEM
                C2F = C2B * (QL + QLK) / QBEM
                C2P = (C2 ** 2 + C2F ** 2 - VSH ** 2) / (2.0 * C2F)
                C2E = C2F + SGM * (C2P - C2F) * (QL + QLK - QBEM) / QBEM
                PEE = PE + DENI * (C2E ** 2 - C2 ** 2) / 2.0
            REI = DENI * (W1 + W2) * DI / VISL / 2.0 / 2.0
            RED = DEND * VD * DD / VISL / 2.0
            FFI = self.get_fff(REI, EDI)
            FFD = self.get_fff(RED, EDD)
            PFI = 2.5 * 4.0 * FFI * DENI * VI ** 2 * self.LI / (2.0 * DI)
            PFD = 2.5 * 4.0 * FFD * DEND * VD ** 2 * self.LD / (2.0 * DD)
            PTI = FTI * DENI * VI ** 2 / 2.0
            PTD = FTD * DEND * VD ** 2 / 2.0
            PPN = PEE - PFI - PFD - PTI - PTD
            UL = self.RLK * OMEGA
            PIO = PEE - PFI - PTI
            PLK = PIO - DENI * (U2 ** 2 - UL ** 2) / 8.0
            if PLK >= 0.:
                VL = np.abs(QLK) / (2.0 * pi * self.RLK * self.SL)
                REL = np.abs(DEND * VL * self.SL / VISD)    # changed VISL to VISD by Haiwen Zhu
                EDL = 0.0
                FFL = self.get_fff(REL, EDL)
                FFL = self.get_fff_leakage(REL,EDL,N, (self.R1+self.R2)/2, VL, self.LG, self.SL)        # by Haiwen Zhu
                VL = np.sqrt(2.0 * PLK / (1.5 + 4.0 * FFL * self.LG / self.SL) / DEND)
                QLKN = 2.0 * pi * self.RLK * self.SL * VL
            else:
                VL = np.abs(QLK / (2.0 * pi * self.RLK * self.SL))
                REL = DENL * VL * self.SL / VISL
                EDL = 0.0
                FFL = self.get_fff(REL, EDL)
                FFL = self.get_fff_leakage(REL,EDL,N, (self.R1+self.R2)/2, VL, self.LG, self.SL)        # by Haiwen Zhu
                VL = np.sqrt(2.0 * np.abs(PLK) / (1.5 + 4.0 * FFL * self.LG / self.SL) / DEND)
                QLKN = -2.0 * pi * self.RLK * self.SL * VL
            HLKloss = 0.25* DENI*(QLK/AI)**2        # by Haiwen Zhu
            PPN = PEE - PFI - PFD - PTI - PTD - HLKloss                              # by Haiwen Zhu
            ABQ = np.abs((QLKN - QLK) / QLK)
            QLK = QLKN
            ABP = np.abs((PPN - PP) / PPN)
            if icon > 200:
                break
            else:
                PP = 0.5*PPN+0.5*PP
                icon += 1
        # return pressure in psi, flow rate in bpd
        PP = PP / psi_to_pa
        PE = PE / psi_to_pa
        PEE = PEE / psi_to_pa
        PF = (PFI + PFD) / psi_to_pa
        PT = (PTI + PTD) / psi_to_pa
        PD = (PFD + PTD) / psi_to_pa
        PRE = np.abs(PE - PEE)
        HLKloss = HLKloss / psi_to_pa
        QLK = QLK * 24.0 * 3600.0 / bbl_to_m3
        if FGL == 'BUB':
            FGL == 'BUB'
        return PP, PE, PF, PT, DB, PRE, HLKloss, QLK, GV

class SinglePhaseCompare(object):
    def __init__(self, pump, conn):
        # QBEM = {'TE2700': 5600, 'DN1750': 3300, 'GC6100': 8800, 'P100': 12000}      # sgl_new
        # QBEM = {'TE2700': 4500, 'DN1750': 3000, 'GC6100': 7800, 'P100': 11000, 'Flex31': 5000}    # sgl_2018
        QBEM = QBEM_default
        self.pump = pump
        self.ESP = ESP[pump]
        self.conn = conn
        self.df_catalog = pd.read_sql_query("SELECT * FROM Catalog_All;", self.conn)
        self.df_pump = self.df_catalog[self.df_catalog['Pump'] == pump]
        # print(self.df_pump)
        self.QBEM = QBEM[pump]

    def single_phase_water(self, sgl_model):
        sgl = SinglePhaseModel(self.ESP, self.QBEM)
        QL, hpsgl, _, _, _, _, _, _, _ = sgl.performance_curve(sgl_model)       #four model options
        # fig = plt.figure(dpi=300)
        # ax = fig.add_subplot(111)
        # ax.plot(QL, hpsgl, 'b-', label='model')
        # ax.plot(self.df_pump['Flow_bpd'], self.df_pump['DP_psi'], 'ro', label='catalog')
        # ax.set_xlabel(r'$Q_L$ (bpd)')
        # ax.set_ylabel(r'$P$ (psi)')
        # ax.set_title(r'{} ESP, $Q_B$={} bpd'.format(self.pump, self.QBEM))
        # ax.legend(frameon=False)
        # 
        df = pd.DataFrame({'ql': QL, 'hp': hpsgl})
        return df

    def single_phase_viscous(self, QL, N, DENL, VISL, WC):
        """
        :param QL: the liquid flow rate in bpd
        :param DENL: liquid density in kg/m3
        :param VISL: liquid viscosity in cP
        :return: HP, a single-point calculation based on the input flow conditions
        """
        sgl = SinglePhaseModel(self.ESP, self.QBEM)
        NS = self.ESP['NS']
        SGM = self.ESP['SGM']
        SN = self.ESP['SN']
        ST = self.ESP['ST']
        QBEM = self.QBEM * bbl_to_m3 / 24.0 / 3600.0
        QL = QL * bbl_to_m3 / 24.0 / 3600.0
        VISL = VISL / 1000.
        # HP, _, _, _, _, _, _, _ = sgl.sgl_calculate_new(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)
        HP, _, _, _, _, _, _, _ = sgl.sgl_calculate_2018(QL, QBEM, DENL, DENW, N, NS, SGM, SN, 0.03, VISL, VISW, WC)
        return HP

    def viscous_fluid_performance(self, N, DENL, VISL, WC, sgl_model=sgl_model):
        """
        calculate single-phase viscous fluid flow H-Q performance with plotting
        :param vis: viscosity in Pas
        :param den: density in kg/m3
        :param WC: water cut
        :return: df_model
        """
        sgl = SinglePhaseModel(self.ESP, self.QBEM)
        NS = self.ESP['NS']
        SGM = self.ESP['SGM']
        SN = self.ESP['SN']
        ST = self.ESP['ST']
        QBEM = self.QBEM * bbl_to_m3 / 24.0 / 3600.0

        QL = np.arange(5000)[1:] * 50.0
        hpsgl = []
        VISL = VISL /1000.

        # q in bpd
        for ql in QL:
            # q in bpd
            ql = ql * bbl_to_m3 / 24.0 / 3600.0
            HP, _, _, _, _, _, _, _ = sgl.sgl_calculate_new(ql, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)
            #HP, _, _, _, _, _, _, _ = sgl.sgl_calculate_2018(ql, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)
            if HP > 0:
                hpsgl.append(HP)
            else:
                break
        df_model = pd.DataFrame({'QL': QL[:len(hpsgl)], 'DP': hpsgl})
        return df_model

#################################
class TwoPhaseCompare(object):
    def __init__(self, pump, conn):
        # QBEM = {'TE2700': 6000, 'DN1750': 3300, 'GC6100': 8800, 'P100': 12000, 'Flex31': 6200}  # gl_new
        QBEM = QBEM_default
        QBEP = {'TE2700': 2700, 'DN1750': 1750, 'GC6100': 6100, 'P100': 9000, 'Flex31': 3100}  # bep flow rate
        self.pump = pump
        self.ESP = ESP[pump]
        self.conn = conn
        self.QBEM = QBEM[pump]
        self.QBEP = QBEP[pump]

    def surging_cd_to_db(self):
        """
        :return: two dataframes for GV and CD_over_dB, the column names: zhu, Barrios, sun
        """
        sgl = SinglePhaseModel(self.ESP, self.QBEM)
        gl = GasLiquidModel(self.ESP, self.QBEM)
        sgl_cal = np.vectorize(sgl.sgl_calculate_new)      #four options: old (zhu and zhang), new (Dr. Zhang update), jc (jiecheng), 2018 (jiecheng-jianjun)
        gl_cal = np.vectorize(gl.gl_calculate_new)  #old, new
        zhu_cal = np.vectorize(gl.gv_zhu)
        sun_cal = np.vectorize(gl.gv_sun)
        Barrios_cal = np.vectorize(gl.gv_Barrios)

        # use the best efficiency point to compare
        GF =  np.arange(0.01, 1, 0.05)
        QL = gl.QL * bbl_to_m3 / 24.0 / 3600.0 * np.ones(GF.shape)
        QG = GF / (1 - GF) * QL
        QBEM = self.QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(GF.shape)
        QLK = 0 * np.ones(GF.shape)
        DENL = gl.DENL * np.ones(GF.shape)
        DENG = gl.DENG * np.ones(GF.shape)
        N = gl.N * np.ones(GF.shape)
        NS = gl.NS * np.ones(GF.shape)
        SGM = gl.SGM * np.ones(GF.shape)
        SN = gl.SN * np.ones(GF.shape)
        ST = gl.ST * np.ones(GF.shape)
        VISL = gl.VISL * np.ones(GF.shape)
        VISG = gl.VISG * np.ones(GF.shape)
        WC = gl.WC * np.ones(GF.shape)

        R1 = gl.R1 * np.ones(GF.shape)
        R2 = gl.R2 * np.ones(GF.shape)
        YI1 = gl.YI1 * np.ones(GF.shape)
        YI2 = gl.YI2 * np.ones(GF.shape)
        RI = gl.RI * np.ones(GF.shape)
        VOI = gl.VOI * np.ones(GF.shape)
        YI = gl.YI * np.ones(GF.shape)
        ZI = gl.ZI * np.ones(GF.shape)
        TB = gl.TB * np.ones(GF.shape)
        B1 = gl.B1 * np.ones(GF.shape)
        B2 = gl.B2 * np.ones(GF.shape)

        HP, _, _, _, _, _, _, _ = sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)
        gv_zhu, cd_over_db_zhu = zhu_cal(DENL, DENG, GF, HP * psi_to_pa, N, QL, QLK, RI, ST, TB, VISL, VOI, YI, ZI)
        gv_sun, cd_over_db_sun = sun_cal(B1, B2, DENL, DENG, N, QL, QG, QLK, R1, R2, TB, VISL, VISG, YI1, YI2, YI, ZI)
        gv_Barrios, cd_over_db_Barrios = Barrios_cal(DENL, DENG, GF, N, QL, QLK, R1, RI, ST, TB, VISL, YI, ZI)

        df_gv = pd.DataFrame({'zhu': gv_zhu, 'sun': gv_sun, 'Barrios': gv_Barrios, 'gf': GF})
        df_cd_over_db = pd.DataFrame({'zhu': cd_over_db_zhu, 'sun': cd_over_db_sun,
                                      'Barrios': cd_over_db_Barrios, 'gf': GF})
        
        # flg for different drag coefficient model
        flgz = np.empty(GF.shape, dtype='str')
        flgs = np.empty(GF.shape, dtype='str')
        flgb = np.empty(GF.shape, dtype='str')
        flgz[:] = 'F'
        flgb[:] = 'B'
        flgs[:] = 'S'

        HP, _, _, _, _, _, _, _ = sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)

        PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                             WC, flgz)
        # df = pd.DataFrame({'gf': GF, 'zhu': PPZ/HP})

        PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                             WC, flgs)
        #df = pd.DataFrame({'gf': GF, 'sun': PPS/HP})
        
        PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                             SN, ST, VISG, VISL, VISW, WC, flgb)
        # df = pd.DataFrame({'gf': GF, 'Barrios': PPB/HP})
        

        df = pd.DataFrame({'gf': GF, 'zhu': PPZ/HP, 'sun': PPS/HP, 'Barrios': PPB/HP})
        
        return df_gv, df_cd_over_db, df

    def surging_performance(self, QL, maxGF, N, p, t,flg='F'):
        """
        :param QL: liquid flow rate in bpd
        :param maxGF: maximum GF for calculation
        :param N: array for rotational speed rpm
        :param p: array for gas pressure psi
        :param t: array for temperature F
        :return: dataframe of predicted pump heads under surging flow, the column names: zhu, Barrios, sun
        """
        sgl = SinglePhaseModel(self.ESP, self.QBEM)
        gl = GasLiquidModel(self.ESP, self.QBEM)
        sgl_cal = np.vectorize(sgl.sgl_calculate_new) #four options: old (zhu and zhang), new (Dr. Zhang update), jc (jiecheng), 2018 (jiecheng-jianjun)
        gl_cal = np.vectorize(gl.gl_calculate_new)  #old, new

        GF = np.arange(0.0, maxGF + 0.02, 0.001)
        QL = QL * bbl_to_m3 / 24.0 / 3600.0 * np.ones(GF.shape)
        QBEM = self.QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(GF.shape)
        QG = GF / (1 - GF) * QL
        DENL = gl.DENL * np.ones(GF.shape)
        DENG = gasdensity(p, t, 0)
        DENG = DENG * np.ones(GF.shape)
        N = N * np.ones(GF.shape)
        NS = gl.NS * np.ones(GF.shape)
        SGM = gl.SGM * np.ones(GF.shape)
        SN = gl.SN * np.ones(GF.shape)
        ST = gl.ST * np.ones(GF.shape)
        VISL = gl.VISL * np.ones(GF.shape)
        VISG = gl.VISG * np.ones(GF.shape)
        WC = gl.WC * np.ones(GF.shape)

        # flg for different drag coefficient model
        
        flgzz = np.empty(GF.shape, dtype='str')
        flgh = np.empty(GF.shape, dtype='str')
        flgz = np.empty(GF.shape, dtype='str')
        flgs = np.empty(GF.shape, dtype='str')
        flgb = np.empty(GF.shape, dtype='str')
        flgzz[:] = 'Z'
        flgz[:] = 'F'
        flgb[:] = 'B'
        flgs[:] = 'S'
        flgh[:] = 'H'

        HP, _, _, _, _, _, _, _ = sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)

        # PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
        #                                      WC, flgz)
        # df = pd.DataFrame({'gf': GF, 'zhu': PPZ/HP})

        # PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
        #                                      WC, flgs)
        #df = pd.DataFrame({'gf': GF, 'sun': PPS/HP})
        
        # PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
        #                                      SN, ST, VISG, VISL, VISW, WC, flgb)
        # df = pd.DataFrame({'gf': GF, 'Barrios': PPB/HP})
        

        # df = pd.DataFrame({'gf': GF, 'zhu': PPZ/HP, 'sun': PPS/HP, 'Barrios': PPB/HP})
        # df = pd.DataFrame({'gf': GF, 'zhu': PPZ, 'sun': PPS, 'Barrios': PPB})

        if flg == 'all':
            PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgz)
            PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgs)
            PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                                SN, ST, VISG, VISL, VISW, WC, flgb)
            PPZZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgzz)
            PPH, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgh)
        if flg == 'F':
            PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgz)
            PPS=PPZ
            PPB=PPZ
            PPZZ=PPZ
            PPH=PPZ
        if flg == 'B':
            PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                                SN, ST, VISG, VISL, VISW, WC, flgb)
            PPZ=PPB
            PPS=PPB
            PPZZ=PPB
            PPH=PPB
        if flg == 'H':
            PPH, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgh)
            PPZ=PPB
            PPS=PPB
            PPB=PPB
            PPZZ=PPB
        if flg == 'S':
            PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgs)
            PPZ=PPS
            PPB=PPS
            PPZZ=PPS
            PPH=PPS
        if flg == 'Z':
            PPZZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgzz)
            PPZ=PPZZ
            PPS=PPZZ
            PPB=PPZZ
            PPH=PPZZ


        df = pd.DataFrame({'gf': GF, 'zhu': PPZ, 'sun': PPS, 'Barrios': PPB, 'Old_zhu': PPZZ, 'Homo': PPH})

        return df

    def error_analysis(self, QL, GVF, N, p, t):
        """
        :param QL: array for liquid flow rate bpd
        :param QG: array for gas friction factor %
        :param N: array for rotational speed rpm
        :param p: array for gas pressure psi
        :param t: array for temperature F
        :return: a data frame for pressure increment
        """
        gl = GasLiquidModel(self.ESP, self.QBEM)
        gl_cal = np.vectorize(gl.gl_calculate_new)

        QL = QL * bbl_to_m3 / 24.0 / 3600.0
        QBEM = self.QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(QL.shape)
        QG = GVF / (100 - GVF) * QL
        DENL = gl.DENL * np.ones(QL.shape)
        h = np.zeros(QL.shape)
        DENG = gasdensity(p, t, h)
        NS = gl.NS * np.ones(QL.shape)
        SGM = gl.SGM * np.ones(QL.shape)
        SN = gl.SN * np.ones(QL.shape)
        ST = gl.ST * np.ones(QL.shape)
        VISL = gl.VISL * np.ones(QL.shape)
        VISG = gl.VISG * np.ones(QL.shape)
        WC = gl.WC * np.ones(QL.shape)

        # flg for different drag coefficient model
        flgz = np.empty(QL.shape, dtype='str')
        flgs = np.empty(QL.shape, dtype='str')
        flgb = np.empty(QL.shape, dtype='str')
        flgz[:] = 'F'
        flgb[:] = 'B'
        flgs[:] = 'S'

        PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                             WC, flgz)
        # PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
        #                                      WC, flgs)
        # PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
        #                                      SN, ST, VISG, VISL, VISW, WC, flgb)

        # df = pd.DataFrame({'zhu': PPZ, 'sun': PPS, 'Barrios': PPB})
        df = pd.DataFrame({'zhu': PPZ})
        return df

    def mapping_performance(self, QG, maxQL, N, p, t,flg='F',sgl_model=sgl_model):
        """
        :param QG: constant gas flow rate bpd
        :param maxQL: maximum liquid flow rate bpd
        :param N: rotational speed rpm
        :param p: array for gas pressure psi
        :param t: array for temperature F
        :return: dataframe of predicted pump heads under mapping flow, the column names: zhu, Barrios, sun
        """
        sgl_model = sgl_model
        gl = GasLiquidModel(self.ESP, self.QBEM)
        gl_cal = np.vectorize(gl.gl_calculate_new)

        # if QG/maxQL<0.02:
        #     # one point wrong, no idea, avoid the point
        #     QL1 =  np.arange(0.01, 0.34, 0.02) * maxQL * bbl_to_m3 / 24.0 / 3600.0
        #     QL2 =  np.arange(0.34, 1.2, 0.08) * maxQL * bbl_to_m3 / 24.0 / 3600.0
        #     QL = np.hstack((QL1,QL2))   # horizontal combine
        #     # QL = np.vstack((QL1,QL2))   # vertical combine
        # else:
        #     QL =  np.arange(0.01, 1.1, 0.02) * maxQL * bbl_to_m3 / 24.0 / 3600.0
        QL =  np.arange(0.01, 1.1, 0.02) * maxQL * bbl_to_m3 / 24.0 / 3600.0
        QG = QG * bbl_to_m3 / 24.0 / 3600.0 * np.ones(QL.shape)

        QBEM = self.QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(QL.shape)
        DENL = DENW * np.ones(QL.shape)
        DENG = gasdensity(p, t, 0)
        DENG = DENG * np.ones(QL.shape)
        NS = gl.NS * np.ones(QL.shape)
        N = N * np.ones(QL.shape)
        SGM = gl.SGM * np.ones(QL.shape)
        SN = gl.SN * np.ones(QL.shape)
        ST = gl.ST * np.ones(QL.shape)
        VISL = gl.VISL * np.ones(QL.shape)
        VISG = gl.VISG * np.ones(QL.shape)
        WC = gl.WC * np.ones(QL.shape)

        # flg for different drag coefficient model
        flgz = np.empty(QL.shape, dtype='str')
        flgs = np.empty(QL.shape, dtype='str')
        flgb = np.empty(QL.shape, dtype='str')
        flgzz = np.empty(QL.shape, dtype='str')
        flgh = np.empty(QL.shape, dtype='str')
        flgz[:] = 'F'   # new Zhu model
        flgb[:] = 'B'   # Barrios
        flgs[:] = 'S'   # Sun
        flgzz[:] = 'Z'  # old Zhu model
        flgh[:] = 'H'   # homogenous model
        if flg == 'all':
            PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgz)
            PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgs)
            PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                                SN, ST, VISG, VISL, VISW, WC, flgb)
            PPZZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgzz)
            PPH, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgh)
        if flg == 'F':
            PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgz)
            PPS=PPZ
            PPB=PPZ
            PPZZ=PPZ
            PPH=PPZ
        if flg == 'B':
            PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                                SN, ST, VISG, VISL, VISW, WC, flgb)
            PPZ=PPB
            PPS=PPB
            PPZZ=PPB
            PPH=PPB
        if flg == 'H':
            PPH, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgh)
            PPZ=PPB
            PPS=PPB
            PPB=PPB
            PPZZ=PPB
        if flg == 'S':
            PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgs)
            PPZ=PPS
            PPB=PPS
            PPZZ=PPS
            PPH=PPS
        if flg == 'Z':
            PPZZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgzz)
            PPZ=PPZZ
            PPS=PPZZ
            PPB=PPZZ
            PPH=PPZZ
        # df = pd.DataFrame({'ql': QL/bbl_to_m3 * 3600 * 24, 'zhu': PPZ, 'sun': PPS, 'Barrios': PPB})
        # df = pd.DataFrame({'ql': QL/bbl_to_m3 * 3600 * 24, 'zhu': PPZ})
        df = pd.DataFrame({'ql': QL/bbl_to_m3 * 3600 * 24, 'zhu': PPZ, 'sun': PPS, 'Barrios': PPB, 'Old_zhu': PPZZ, 'Homo': PPH})

        return df

    def GVF_performance(self, GF, maxQL, minQL, N, p, t,flg='F'):
        """
        :param QG: constant gas flow rate bpd
        :param maxQL: maximum liquid flow rate bpd
        :param N: rotational speed rpm
        :param p: array for gas pressure psi
        :param t: array for temperature F
        :return: dataframe of predicted pump heads under mapping flow, the column names: zhu, Barrios, sun
        """
        sgl = SinglePhaseModel(self.ESP, self.QBEM)
        gl = GasLiquidModel(self.ESP, self.QBEM)
        sgl_cal = np.vectorize(sgl.sgl_calculate_new)
        gl_cal = np.vectorize(gl.gl_calculate_new)
        QLmin = minQL/maxQL
        if QLmin == 0:
            QLmin = 0.01
        QL = np.arange(0.3*QLmin, 1.2, 0.02) * maxQL * bbl_to_m3 / 24.0 / 3600.0
        QG = GF / (1 - GF) * QL

        QBEM = self.QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(QL.shape)
        DENL = DENW * np.ones(QL.shape)
        DENG = gasdensity(p, t, 0)
        DENG = DENG * np.ones(QL.shape)
        NS = gl.NS * np.ones(QL.shape)
        N = N * np.ones(QL.shape)
        SGM = gl.SGM * np.ones(QL.shape)
        SN = gl.SN * np.ones(QL.shape)
        ST = gl.ST * np.ones(QL.shape)
        VISL = gl.VISL * np.ones(QL.shape)
        VISG = gl.VISG * np.ones(QL.shape)
        WC = gl.WC * np.ones(QL.shape)

        # flg for different drag coefficient model
        flgz = np.empty(QL.shape, dtype='str')
        flgs = np.empty(QL.shape, dtype='str')
        flgb = np.empty(QL.shape, dtype='str')
        flgz[:] = 'F'
        flgb[:] = 'B'    
        flgs[:] = 'S'
        
        HP, _, _, _, _, _, _, _ = sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)

        PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                             WC, flgz)
        # df = pd.DataFrame({'ql': QL/bbl_to_m3 * 3600 * 24, 'zhu': PPZ/HP})

        # PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
        #                                      WC, flgs)
        # #df = pd.DataFrame({'ql': QL/bbl_to_m3 * 3600 * 24, 'sun': PPS, 'water': HP})
        
        # PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
        #                                      SN, ST, VISG, VISL, VISW, WC, flgb)
        # df = pd.DataFrame({'ql': QL/bbl_to_m3 * 3600 * 24, 'Barrios': PPB/HP})
        

        df = pd.DataFrame({'ql': QL/bbl_to_m3 * 3600 * 24, 'zhu': PPZ})
        # df = pd.DataFrame({'ql': QL/bbl_to_m3 * 3600 * 24, 'zhu': PPZ, 'sun': PPS, 'Barrios': PPB, 'water': HP})
        #df = pd.DataFrame({'ql': QL / bbl_to_m3 * 3600 * 24, 'sun': PPS/HP})

        return df

def connect_db(db):
    """
    :param db: the data base name in text
    :return: a connection with database and a cursor
    """
    conn = sqlite3.connect(db)
    c = conn.cursor()
    return conn, c

def disconnect_db(conn):
    """
    :param conn: a sqlite database connection
    :return: None
    """
    conn.commit()   #apply changes to the database
    conn.close()

def gasdensity(p, t, h):
    """
    gas density based on CIPM-81 (Davis, 1972) correlations
    :param p: pressure in psig
    :param t: temperature in Fahrenheit
    :param h: humidity in %
    :return: gas density in kg/m3
    """
    A = 1.2811805e-5
    B = -1.9509874e-2
    C = 34.04926034
    D = -6.3536311e3
    alpha = 1.00062
    beta = 3.14e-8
    gamma = 5.6e-7
    a0 = 1.62419e-6
    a1 = -2.8969e-8
    a2 = 1.0880e-10
    b0 = 5.757e-6
    b1 = -2.589e-8
    c0 = 1.9297e-4
    c1 = -2.285e-6
    d = 1.73e-11
    e = -1.034e-8
    R = 8.31441
    Ma = 28.9635  # air molecular weight, g/mol
    Mv = 18  # water molecular weight, g/mol

    Pabs = (p + 14.7) * 6894.76
    Tt = (t - 32) / 1.8
    Tabs = Tt + 273.15
    psv = 1.0 * np.exp(A * (Tabs) ** 2.0 + B * Tabs + C + D / Tabs)
    f = alpha + beta * Pabs + gamma * (Tt) ** 2
    xv = h / 100.0 * f * psv / Pabs
    Z = 1.0 - Pabs / Tabs * (a0 + a1 * Tt + a2 * (Tt) ** 2.0 + (b0 + b1 * Tt) * xv + (c0 + c1 * Tt) * (xv) ** 2) + \
        (Pabs / Tabs) ** 2.0 * (d + e * (xv) ** 2.0)
    return Pabs * Ma / 1000.0 / (Z * R * Tabs) * (1.0 - xv * (1.0 - Mv / Ma))

# fitting water curve
def fitting_water(pump_name, QBEM, sgl_model):
    conn, c = connect_db('ESP.db')
    df_data = pd.read_sql_query("SELECT * FROM Catalog_All;", conn)
    df_data = df_data[df_data.Pump == pump_name]
    df_data = df_data[df_data.Flow_bpd != 0]
    df_data=df_data.reset_index(drop=True)
    
    # fig=plt.figure(dpi=128, figsize=(10,6))
    # ax = fig.add_subplot(111)
    # ax.scatter(df_data.Flow_bpd, df_data.DP_psi,label='test', color = 'red')

    sgl = SinglePhaseModel(ESP[pump_name],QBEM)

    if sgl_model=='zhang_2015':
        sgl_cal = np.vectorize(sgl.sgl_calculate_old)           
    elif sgl_model=='zhang_2016':
        sgl_cal = np.vectorize(sgl.sgl_calculate_new)    
    elif sgl_model=='jiecheng_2017':
        sgl_cal = np.vectorize(sgl.sgl_calculate_jc) 
    elif sgl_model=='zhu_2018':
        sgl_cal = np.vectorize(sgl.sgl_calculate_2018)

    ABV = 1
    error = 1
    icon = 0
    QL = df_data.Flow_bpd * bbl_to_m3 / 24.0 / 3600.0
    QBEM = QBEM * bbl_to_m3 / 24.0 / 3600.0 # * np.ones(QL.shape)
    DENL = DENW # * np.ones(QL.shape)
    N = ESP[pump_name]['N'] # * np.ones(QL.shape)
    NS = ESP[pump_name]['NS'] # * np.ones(QL.shape)
    SGM = ESP[pump_name]['SGM'] # * np.ones(QL.shape)
    SN = ESP[pump_name]['SN']
    ST = ESP[pump_name]['ST']
    VISL = ESP[pump_name]['VISW']
    VISW = ESP[pump_name]['VISW']
    WC = ESP[pump_name]['WC']
    # HP = df_data.DP_psi
    HP = np.ones(df_data.Flow_bpd.shape)
    for i in range(HP.shape[0]):
        HP[i] = df_data.DP_psi[i]


    while np.abs(ABV) > E1 or np.abs(error) >E1:
        error = 0
        ABV = 0
        icon += 1
        if icon > 1000:
            break
        HPN, _, _, _, _, _, _, _ = sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)

        for i in range(HP.shape[0]):
            error += (HPN[i]-df_data.DP_psi[i])/(df_data.DP_psi[i]) 
            ABV += (HP[i]-HPN[i])/HP[i]
            HP[i] = HPN[i]
        if error >0:
            QBEM = QBEM * (1+error*0.1)
            # if QBEM*0.1 < error* bbl_to_m3 / 24.0 / 3600.0:
            #     QBEM = QBEM * 1.1
            # else:
            #     QBEM = QBEM + error* bbl_to_m3 / 24.0 / 3600.0
        else:
            QBEM = QBEM * (1+error*0.1)
            # if QBEM*0.1 > np.abs(error* bbl_to_m3 / 24.0 / 3600.0):
            #     QBEM = QBEM * 0.9
            # else:
            #     QBEM = QBEM + error* bbl_to_m3 / 24.0 / 3600.0

    QL = QL / (bbl_to_m3 / 24.0 / 3600.0)
    QBEM = QBEM / (bbl_to_m3 / 24.0 / 3600.0)
    
    # df_model = ESP_case.single_phase_water(sgl_model)
    # ax.plot(QL, HP, label='model', color = 'blue')
    # ax.set_title(pump_name + ' catalog at ''QBEM= '+ str(QBEM))
    # ax.legend(frameon=False, fontsize=5)
    # 
    return QBEM, QL, HP

# plot TE2700 ESP flow case
def vis_te2700_plot(SQL_path):
    conn, c = connect_db(SQL_path)
    te2700_case = SinglePhaseCompare('TE2700', conn)

    df_jc = pd.read_sql_query("SELECT * FROM TE2700_JiechengZhang;", conn)
    df_3500 = df_jc[(df_jc.RPM == 3500)]
    df_2400 = df_jc[(df_jc.RPM == 2400)]
    df_all = [df_3500, df_2400]


    for index, df in enumerate(df_all):
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        id = 0

        for visl in df.viscosity_cP.unique():
            denl = df[df.viscosity_cP == visl]['Density_kg/m3'].mean()  # average density
            df_model = te2700_case.viscous_fluid_performance(3500 - 1100 * index, denl, visl, 0)
            df_turzo = turzo_2000('TE2700', visl, denl, 3500 - 1100 * index)
            ax.scatter(df[df.viscosity_cP == visl]['flow_bpd'], df[df.viscosity_cP == visl]['DP_psi'],
                        marker=symbols[id], label=r'{} cp_exp'.format(visl), facecolors='C{}'.format(id + 1), linewidths=0.75)
            ax.plot(df_model.QL, df_model.DP, c='C{}'.format(id + 1), label=r'{} cp_model'.format(visl), linewidth=0.75)
            ax.plot(df_turzo.Qvis, df_turzo.DPvis, c='C{}'.format(id + 1), label=r'{} cp_Turzo (2000)'.format(visl),
                     linestyle='--', linewidth=0.75)
            id += 1

        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[10], handles[11], handles[12], handles[13], handles[14],
                   handles[0], handles[2], handles[4], handles[6], handles[8],
                   handles[1], handles[3], handles[5], handles[7], handles[9]]

        labels = [labels[10], labels[11], labels[12], labels[13], labels[14],
                  labels[0], labels[2], labels[4], labels[6], labels[8],
                  labels[1], labels[3], labels[5], labels[7], labels[9]]

        ax.set_xlabel(r'$Q_L$ (bpd)')
        ax.set_ylabel(r'$P$ (psi)')
        ax.legend(handles, labels, frameon=False, fontsize=5)
        #fig.show()
    fig.savefig('VisTe2700.jpg')

    # plot best match line curve
    fig3 = plt.figure(dpi=300)
    ax3 = fig3.add_subplot(111)
    model_predict = []

    for i in range(df_jc.shape[0]):
        ql = df_jc.iloc[i].flow_bpd
        N = df_jc.iloc[i].RPM
        DENL = df_jc.iloc[i]['Density_kg/m3']
        VISL = df_jc.iloc[i].viscosity_cP
        model_predict.append(te2700_case.single_phase_viscous(ql, N, DENL, VISL, 0))

    df_jc['model'] = model_predict
    x_max = np.array(range(int(max(df_jc.model) + 1)))

    df_2400 = df_jc[df_jc.RPM == 2400]
    df_3500 = df_jc[df_jc.RPM == 3500]
    ax3.scatter(df_2400.DP_psi, df_2400.model, facecolor='b',
                marker=symbols[0], label=r'$N=2400$ rpm', linewidth=0.75)
    ax3.scatter(df_3500.DP_psi, df_3500.model, facecolor='r',
                marker=symbols[1], label=r'$N=3500$ rpm', linewidth=0.75)
    ax3.plot(x_max, x_max, 'k--', label='perfect match')
    ax3.plot(x_max, 1.25 * x_max, ls=':', c='gray')
    ax3.plot(x_max, 0.85 * x_max, ls=':', c='gray')
    ax3.set_xlabel(r'$P_{exp}$ (psi)')
    ax3.set_ylabel(r'$P_{sim}$ (psi)')
    ax3.legend(frameon=False, fontsize=8)
    fig3.show()
    fig3.savefig('VisTe2700Error.jpg')
    
    # epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6 = stats_analysis(df_jc.model, df_jc.DP_psi)
    # print('##########',epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6,'##########')

    disconnect_db(conn)

# define Turzo et al. (2000) method for viscous fluid flow
def turzo_2000(pump, vis, den, N):
    """
    :param pump: pump name in string
    :param vis: viscosity in cP
    :param den: density in kg/m3
    :param N: rotational speed in rpm
    :return: boosting pressure at four different flow rates, 0.6, 0.8, 1.0. 1.2 Qbep
    """
    # QBEM = {'TE2700': 4500, 'DN1750': 3000, 'GC6100': 7800, 'P100': 11000}
    QBEM = QBEM_default
    QBEP = {'TE2700': 2700, 'DN1750': 1750, 'GC6100': 6100, 'P100': 9000}

    qbep = QBEP[pump] * bbl_to_m3 / 3600 / 24
    qbem = QBEM[pump] * bbl_to_m3 / 3600 / 24
    esp = ESP[pump]
    ns = esp['NS']
    sgm = esp['SGM']
    sn = esp['SN']
    st = esp['ST']
    denl = esp['DENL']
    visl = esp['VISL']
    visw = esp['VISW']
    wc = esp['WC']

    sgl = SinglePhaseModel(esp, qbem)
    DPbep, _, _, _, _, _, _, _ = sgl.sgl_calculate_2018(qbep, qbem, denl, DENW, N, ns, sgm, sn, st, visl, visw, wc)
    DPbep06, _, _, _, _, _, _, _ = sgl.sgl_calculate_2018(0.6 * qbep, qbem, denl, DENW, N, ns, sgm, sn, st, visl,
                                                          visw, wc)
    DPbep08, _, _, _, _, _, _, _ = sgl.sgl_calculate_2018(0.8 * qbep, qbem, denl, DENW, N, ns, sgm, sn, st, visl,
                                                          visw, wc)
    DPbep12, _, _, _, _, _, _, _ = sgl.sgl_calculate_2018(1.2 * qbep, qbem, denl, DENW, N, ns, sgm, sn, st, visl,
                                                          visw, wc)
    # convert units
    qbep = QBEP[pump] * 42 / 1440               # to 100 gpm
    vis = vis / (den / DENW)                    # to cSt
    Hbep = DPbep * psi_to_ft

    y = -7.5946 + 6.6504 * np.log(Hbep) + 12.8429 * np.log(qbep)
    Qstar = np.exp((39.5276 + 26.5605 * np.log(vis) - y)/51.6565)

    CQ = 1.0 - 4.0327e-3 * Qstar - 1.724e-4 * Qstar**2

    CH06 = 1.0 - 3.68e-3 * Qstar - 4.36e-5 * Qstar**2
    CH08 = 1.0 - 4.4723e-3 * Qstar - 4.18e-5 * Qstar**2
    CH10 = 1.0 - 7.00763e-3 * Qstar - 1.41e-5 * Qstar**2
    CH12 = 1.0 - 9.01e-3 * Qstar + 1.31e-5 * Qstar**2

    Qvis = CQ * np.array([0.6 * qbep, 0.8 * qbep, qbep, 1.2 * qbep]) * 1440 / 42        # to bpd
    DPvis = np.array([CH06, CH08, CH10, CH12]) * np.array([DPbep06, DPbep08, DPbep, DPbep12])

    df = pd.DataFrame({'Qvis': Qvis, 'DPvis': DPvis, 'N': [N] * 4})

    return df

# define a statistical function
def stats_analysis(df_pre, df_exp):
    df_relative = (df_pre - df_exp) / df_exp * 100
    df_actual = df_pre - df_exp

    epsilon1 = df_relative.mean()
    epsilon2 = df_relative.abs().mean()
    epsilon3 = df_relative.std()
    epsilon4 = df_actual.mean()
    epsilon5 = df_actual.abs().mean()
    epsilon6 = df_actual.std()
    return epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6

def All_pump_water(SQLname,tablename,casename, pump_name,time, QBEM, testtype, Npumplist):
    try:
        conn, c = connect_db(SQLname)
        df_data = pd.read_sql_query("SELECT * FROM " + tablename + ";", conn)
        df_data = df_data[df_data.Pump == pump_name]
        df_data = df_data[df_data.QL_bpd > 0]
        df_data = df_data[df_data.Time == time]
        df_data = df_data[df_data.TargetVISL_cp == 1]
    except:
        print('错误的SQL路径，不能进行验证')
        return

    if testtype != 'none':
        df_data = df_data[df_data.Test == testtype]
    df_data=df_data.reset_index(drop=True)
    if df_data.shape[0]<1:
        print('错误的SQL筛选条件，没有测试数据')
        return
    
    sgl = SinglePhaseModel(ESP[pump_name],QBEM)

    if sgl_model=='zhang_2015':
        sgl_cal = np.vectorize(sgl.sgl_calculate_old)           
    elif sgl_model=='zhang_2016':
        sgl_cal = np.vectorize(sgl.sgl_calculate_new)    
    elif sgl_model=='jiecheng_2017':
        sgl_cal = np.vectorize(sgl.sgl_calculate_jc) 
    elif sgl_model=='zhu_2018':
        sgl_cal = np.vectorize(sgl.sgl_calculate_2018)

    QBEM_default[pump_name]=QBEM
    ABV = 1
    error = 1
    icon = 0
    QL =  np.arange(0.01, 1.1, 0.02) * df_data.QL_bpd.max() * bbl_to_m3 / 24.0 / 3600.0
    QBEM = QBEM * bbl_to_m3 / 24.0 / 3600.0 # * np.ones(QL.shape)
    DENL = DENW # * np.ones(QL.shape)
    NS = ESP[pump_name]['NS'] # * np.ones(QL.shape)
    SGM = ESP[pump_name]['SGM'] # * np.ones(QL.shape)
    SN = ESP[pump_name]['SN']
    ST = ESP[pump_name]['ST']
    VISL = ESP[pump_name]['VISW']
    VISW = ESP[pump_name]['VISW']
    WC = ESP[pump_name]['WC']
    # HP = df_data['DP_psi']
    HP = np.ones(df_data.QL_bpd.shape)
    for i in range(HP.shape[0]):
        HP[i] = df_data['DP_psi'][i]


    fig, ax = plt.subplots (dpi =128, figsize = (3.33,2.5))
    fig2, ax2 = plt.subplots (dpi =128, figsize = (3.33,2.5))
    icon = 0

    df_water = df_data.copy(deep=False)
    if df_water.shape[0]>1000:
        # df_water = df_water.sample(frac = 0.20)
        df_water = df_data.sample(500)
    HP, _, _, _, _, _, _, _ = sgl_cal(df_water.QL_bpd* bbl_to_m3 / 24.0 / 3600.0, QBEM, DENL, DENW, df_water.TargetRPM, NS, SGM, SN, ST, VISL, VISW, WC)
    df_stats = pd.DataFrame({'pre': HP.tolist(), 'exp': df_water['DP_psi'].values.tolist()})
    epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6 = stats_analysis(df_stats.pre, df_stats.exp)
    print('##########',epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6,'#############################',)

    for N in Npumplist:
        df_water = df_data[df_data.TargetRPM == N]
        if df_water.shape[0]>1000:
            # df_water = df_water.sample(frac = 0.20)
            df_water = df_water.sample(500)
        HP, _, _, _, _, _, _, _ = sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)
        ax.plot(QL / (bbl_to_m3 / 24.0 / 3600.0), HP, linestyle='-', label='Sim ' + str(round(N)) + ' RPM', c='C{}'.format(icon), linewidth=0.75)       # field
        ax.scatter(df_water['QL_bpd'],df_water['DP_psi'],label='Test ' + str(round(N)) + ' RPM', 
                            facecolor='C{}'.format(icon),marker=symbols[icon],linewidths=0.75, s=8)       # field
        HP, _, _, _, _, _, _, _ = sgl_cal(df_water.QL_bpd* bbl_to_m3 / 24.0 / 3600.0, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)
        ax2.scatter(df_water['DP_psi'],HP,label='Test ' + str(round(N)) + ' RPM', 
                            facecolor='C{}'.format(icon),marker=symbols[icon],linewidths=0.75, s=8)       # field
        icon+=1
    
    ax.set_xlabel('Qw bpd', fontsize=8)
    ax.set_ylabel('Head psi', fontsize=8)
    ax.legend(frameon=False, fontsize=5)
    if pump_name == 'Flex31':
        pump_name = 'MTESP'
    title='Water performance: '+pump_name
    ax.set_title(title, fontsize=8)
    # ax.set_xlim(0,df_data['QL_bpd'].max()/150.96*1.2)       # SI
    # ax.set_ylim(0,df_data['dp12_psi'].max()/1.4223*1.2)       # SI
    dx = round((df_data['QL_bpd'].max()*1.2-0)/400)*100
    dy = round((df_data['DP_psi'].max()*1.2-0)/4)*1
    ax.xaxis.set_ticks(np.arange(round(0), round(df_data['QL_bpd'].max()*1.2+200), dx))
    ax.yaxis.set_ticks(np.arange(round(0), round(df_data['DP_psi'].max()*1.2+1), dy))

    ax.set_xlim(0,df_data['QL_bpd'].max()*1.25)
    ax.set_ylim(0,df_data['DP_psi'].max()*1.2)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    fig.tight_layout()
    title2=title.replace(":","")+'.jpg'
    fig.savefig(title2)

    x_max = np.arange(0, df_data['DP_psi'].max()*1.2, 0.2)
    ax2.plot(x_max, x_max, color='black',linestyle='-', label='perfect match', linewidth=0.75)
    ax2.plot(x_max, x_max*0.8, 'r--', label='-20%', linewidth=0.75)
    ax2.plot(x_max, x_max*1.2, 'r-.', label='+20%',  linewidth=0.75)
    # ax2.set_xlabel('Exp head (m of water)', fontsize=8)       # SI
    # ax2.set_ylabel("Sim head (m of water)", fontsize=8)       # SI
    ax2.set_xlabel('Exp head (psi)', fontsize=8)       # field
    ax2.set_ylabel("Sim head (psi)", fontsize=8)       # field
    ax2.legend(frameon=False, fontsize=5)
    # ax2.set_xlim(0,x_max.max()/1.4223)       # SI
    # ax2.set_ylim(0,x_max.max()/1.4223)       # SI
    dx = round((x_max.max()-0)/4)
    dy = round((x_max.max()-0)/4)
    ax2.xaxis.set_ticks(np.arange(round(0), round(x_max.max()+1), dx))
    ax2.yaxis.set_ticks(np.arange(round(0), round(x_max.max()+1), dy))
    ax2.set_xlim(0,x_max.max())       # field
    ax2.set_ylim(0,x_max.max())       # field
    title='Water curve error analysis: '+pump_name
    ax2.set_title(title, fontsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    fig2.tight_layout()
    title2=title.replace(":","")+'.jpg'
    fig2.savefig(title2)

    HP_sim, _, _, _, _, _, _, _ = sgl_cal(QL, QBEM, DENL, DENW, 3600, NS, SGM, SN, ST, VISL, VISW, WC)
    
    return QBEM/(bbl_to_m3 / 24.0 / 3600.0), QL/(bbl_to_m3 / 24.0 / 3600.0), HP_sim

def All_pump_mapping(SQLname,tablename,casename, TargetQg
                        , xlabel, ylabel, pump_name, test_type, Npump, Pin_psi, erroron):
    conn, c = connect_db(SQLname)
    df_data = pd.read_sql_query("SELECT * FROM " + tablename + ";", conn)
    df_data=df_data[(df_data.Case == casename) & (df_data.Test == test_type)]
    if Npump != 'none':
        df_data=df_data[(df_data.TargetRPM == Npump)]
    if Pin_psi != 'none':
        df_data=df_data[(df_data.TargetP_psi == Pin_psi)]
    if df_data.shape[0] < 1:
        print('错误的SQL路径，不能进行验证')
        return
    df_data=df_data.reset_index(drop=True)

    fig, ax = plt.subplots (dpi =128, figsize = (3.33,2.5))
    fig2, ax2 = plt.subplots (dpi =128, figsize = (3.33,2.5))

    ESP_case = TwoPhaseCompare(pump_name, conn)
    icon=0
    # 1st mapping
    for TargetQg1 in TargetQg:
        df_mapping1=df_data[df_data.TargetQG_bpd == TargetQg1]
        if df_mapping1.shape[0] > 1:
            df_model1 = ESP_case.mapping_performance(df_mapping1['QG_bpd'].mean(), df_data.QL_bpd.max() + 100, df_mapping1.RPM.mean(),
                                                        df_mapping1.Ptank_psi.mean(), df_mapping1.Tin_F.mean())
            ax.plot(df_model1.ql, df_model1.zhu, linestyle='-', label='Sim ' + str(round(df_mapping1['QG_bpd'].mean())) + ' bpd',
                                                            c='C{}'.format(icon), linewidth=0.75)       # field
            # ax.plot(df_model1.ql, df_model1.sun, linestyle='--', label='Sim ' + str(round(df_mapping1['QG_bpd'].mean())) + ' bpd',
            #                                                 c='C{}'.format(icon), linewidth=0.75)       # field
            # ax.plot(df_model1.ql, df_model1.Barrios, linestyle='-.', label='Sim ' + str(round(df_mapping1['QG_bpd'].mean())) + ' bpd',
            #                                                 c='C{}'.format(icon), linewidth=0.75)       # field
            # ax.plot(df_model1.ql, df_model1.Old_zhu, linestyle=':', label='Sim ' + str(round(df_mapping1['QG_bpd'].mean())) + ' bpd',
            #                                                 c='C{}'.format(icon), linewidth=0.75)       # field
            # ax.plot(df_model1.ql, df_model1.Homo, linestyle=' ', label='Sim ' + str(round(df_mapping1['QG_bpd'].mean())) + ' bpd',
                                                            # c='C{}'.format(icon), linewidth=0.75)       # field
            if erroron == False:
                ax.scatter(df_mapping1['QL_bpd'],df_mapping1['dp12_psi'],label='Test ' + str(round(df_mapping1['QG_bpd'].mean())) + ' bpd', 
                            facecolor='C{}'.format(icon),marker=symbols[icon],linewidths=0.75, s=8)       # field
            else:
                df_model = ESP_case.error_analysis(df_mapping1.QL_bpd, df_mapping1['HG_%'], df_mapping1.RPM, df_mapping1.Ptank_psi, df_mapping1.Tin_F)
                df_model=df_model.reset_index(drop=True)
                df_mapping1=df_mapping1.reset_index(drop=True)
                df = pd.concat( [df_model, df_mapping1], axis=1 )       # 1 combine column
                df.insert(0,'error', 0)       # avoid warning
                # df['error']=0.0
                for i in range(df.shape[0]):    # return column number
                    # df['error'][i]=(df.zhu[i]-df['dp12_psi'][i])/df['dp12_psi'][i]*100.0
                    if df['dp12_psi'][i] == 0: df.iloc [i, df.columns.get_loc("dp12_psi")] = 1e-6     # avoid error
                    df.iloc [i, df.columns.get_loc("error")]=(df.zhu[i]-df['dp12_psi'][i])/df['dp12_psi'][i]*100.0

                df=df[(df.error < error_control_high) & (df.error > error_control_low)]
                df_reduced = df.sort_values(by=['QL_bpd'])
                ax.scatter(df_reduced['QL_bpd'],df_reduced['dp12_psi'],label='Test ' + str(round(df_mapping1['QG_bpd'].mean())) + ' bpd', 
                            facecolor='C{}'.format(icon),marker=symbols[icon],linewidths=0.75, s=8)       # field
                ax2.scatter(df_reduced['dp12_psi'], df_reduced.zhu, marker=symbols[icon], facecolor='C{}'.format(icon),linewidths=0.75, s=8,
                                                        label='Qg = ' + str(round(df_mapping1['QG_bpd'].mean())) + ' bpd')       # field
                df_stats = pd.DataFrame({'pre': df_reduced.zhu.values.tolist(), 'exp': df_reduced['dp12_psi'].values.tolist()})
                epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6 = stats_analysis(df_stats.pre, df_stats.exp)
                print('##########',epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6,'#############################',)
                icon+=1

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(frameon=False, fontsize=5)
    if pump_name == 'Flex31':
        pump_name = 'MTESP'
    title='Performance: '+pump_name+' '+test_type
    if Npump != 'none' or Pin_psi != 'none':
        title=title+' '+str(Npump)+' RPM'
    if Pin_psi != 'none':
        title=title+' '+str(Pin_psi)+' psi'
    ax.set_title(title, fontsize=8)
    
    dx = round((df_data['QL_bpd'].max()*1.2-0)/400)*100
    dy = round((df_data['dp12_psi'].max()*1.2-0)/4)
    ax.xaxis.set_ticks(np.arange(round(0), round(df_data['QL_bpd'].max()*1.2+200), dx))
    ax.yaxis.set_ticks(np.arange(round(0), round(df_data['dp12_psi'].max()*1.2+1), dy))

    ax.set_xlim(0,df_data['QL_bpd'].max()*1.25)
    ax.set_ylim(0,df_data['dp12_psi'].max()*1.2)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    fig.tight_layout()
    title2=title.replace(":","")+'.jpg'
    fig.savefig(title2)

    x_max = np.arange(0, df_data['dp12_psi'].max()*1.2, 0.2)
    ax2.plot(x_max, x_max, color='black',linestyle='-', label='perfect match', linewidth=0.75)
    ax2.plot(x_max, x_max*0.8, 'r--', label='-20%', linewidth=0.75)
    ax2.plot(x_max, x_max*1.2, 'r-.', label='+20%',  linewidth=0.75)
    ax2.set_xlabel('Exp head (psi)', fontsize=8)       # field
    ax2.set_ylabel("Sim head (psi)", fontsize=8)       # field
    ax2.legend(frameon=False, fontsize=5)
    
    dx = round((x_max.max()-0)/4)*1
    dy = round((x_max.max()-0)/4)
    ax2.xaxis.set_ticks(np.arange(round(0), round(x_max.max()+1), dx))
    ax2.yaxis.set_ticks(np.arange(round(0), round(x_max.max()+1), dy))

    ax2.set_xlim(0,x_max.max())       # field
    ax2.set_ylim(0,x_max.max())       # field
    title='Error analysis: '+pump_name+' '+test_type+' '
    if Npump != 'none':
        title=title+str(Npump)+' RPM'
    if Pin_psi != 'none':
        title=title+' '+str(Pin_psi)+' psi'
    ax2.set_title(title, fontsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    fig2.tight_layout()
    title2=title.replace(":","")+'.jpg'
    fig2.savefig(title2)

    disconnect_db(conn)

def All_pump_surging(SQLname,tablename,casename, TargetQl
                        , xlabel, ylabel, pump_name, test_type, Npump, Pin_psi, erroron):
    conn, c = connect_db(SQLname)
    df_data = pd.read_sql_query("SELECT * FROM " + tablename + ";", conn)
    df_data=df_data[(df_data.Case == casename) & (df_data.Test == test_type)]
    if Npump != 'none':
        df_data=df_data[(df_data.TargetRPM == Npump)]
    if Pin_psi != 'none':
        df_data=df_data[(df_data.TargetP_psi == Pin_psi)]
    if df_data.shape[0] < 1:
        print('No data selected')
        return
    df_data=df_data.reset_index(drop=True)

    fig, ax = plt.subplots (dpi =128, figsize = (3.33,2.5))
    fig2, ax2 = plt.subplots (dpi =128, figsize = (3.33,2.5))

    ESP_case = TwoPhaseCompare(pump_name, conn)
    icon=0
    # 1st mapping
    for TargetQl1 in TargetQl:
        df_mapping1=df_data[df_data.TargetQL_bpd == TargetQl1]
        if df_mapping1.shape[0] > 1:
            df_model1 = ESP_case.surging_performance(df_mapping1['QL_bpd'].mean(), df_data['HG_%'].max() / 100., df_mapping1.RPM.mean(),
                                                        df_mapping1.Ptank_psi.mean(), df_mapping1.Tin_F.mean())
            # ax.plot(df_model1.gf * 100, df_model1.zhu/1.4223, linestyle='-', label='Sim ' + str(round(df_mapping1['QL_bpd'].mean())) + ' m$^{3}$/h', c='C{}'.format(icon), linewidth=0.75)       # SI
            ax.plot(df_model1.gf * 100, df_model1.zhu, linestyle='-', label='Sim ' + str(round(df_mapping1['QL_bpd'].mean())) + ' bpd',c='C{}'.format(icon), linewidth=0.75)       # field

            if erroron == False:
                # ax.scatter(df_mapping1['HG_%']/150.96,df_mapping1['DP_psi']/1.4223,label='Test ' + str(round(df_mapping1['QL_bpd'].mean())) + ' m$^{3}$/h', 
                #             facecolor='C{}'.format(icon),marker=symbols[icon],linewidths=0.75, s=8)       # SI
                ax.scatter(df_mapping1['HG_%'],df_mapping1['DP_psi'],label='Test ' + str(round(df_mapping1['QL_bpd'].mean())) + ' bpd', 
                            facecolor='C{}'.format(icon),marker=symbols[icon],linewidths=0.75, s=8)       # field
            else:
                df_model = ESP_case.error_analysis(df_mapping1.QL_bpd, df_mapping1['HG_%'], df_mapping1.RPM, df_mapping1.Ptank_psi, df_mapping1.Tin_F)
                df_model=df_model.reset_index(drop=True)
                df_mapping1=df_mapping1.reset_index(drop=True)
                df = pd.concat( [df_model, df_mapping1], axis=1 )       # 1 combine column
                df['error']=0.0
                for i in range(df.shape[0]):    # return column number
                    df['error'][i]=(df.zhu[i]-df['dp12_psi'][i])/df['dp12_psi'][i]*100.0

                df=df[(df.error < error_control_high) & (df.error > error_control_low)]
                df_reduced = df.sort_values(by=['QL_bpd'])
                # ax.scatter(df_reduced['HG_%']/150.96,df_reduced['dp12_psi']/1.4223,label='Test ' + str(round(df_mapping1['QL_bpd'].mean())) + ' m$^{3}$/h', 
                #             facecolor='C{}'.format(icon),marker=symbols[icon],linewidths=0.75, s=8)       # SI
                ax.scatter(df_reduced['HG_%'],df_reduced['dp12_psi'],label='Test ' + str(round(df_mapping1['QL_bpd'].mean())) + ' bpd', 
                            facecolor='C{}'.format(icon),marker=symbols[icon],linewidths=0.75, s=8)       # field
                # ax2.scatter(df_reduced['dp12_psi']/1.4223, df_reduced.zhu/1.4223, marker=symbols[icon], facecolor='C{}'.format(icon),linewidths=0.75, s=8,
                #                                         label='Qg = ' + str(round(df_mapping1['QL_bpd'].mean()/150.96,2)) + ' m$^{3}$/h')       # SI
                ax2.scatter(df_reduced['dp12_psi'], df_reduced.zhu, marker=symbols[icon], facecolor='C{}'.format(icon),linewidths=0.75, s=8,
                                                        label='Qg = ' + str(round(df_mapping1['QL_bpd'].mean())) + ' bpd')       # field
                df_stats = pd.DataFrame({'pre': df_reduced.zhu.values.tolist(), 'exp': df_reduced['dp12_psi'].values.tolist()})
                epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6 = stats_analysis(df_stats.pre, df_stats.exp)
                print('##########',epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, epsilon6,'#############################',)
                icon+=1



    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(frameon=False, fontsize=5)
    if pump_name == 'Flex31':
        pump_name = 'MTESP'
    title='Performance: '+pump_name+' '+test_type
    if Npump != 'none' or Pin_psi != 'none':
        title=title+' '+str(Npump)+' RPM'
    if Pin_psi != 'none':
        title=title+' '+str(Pin_psi)+' psi'
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0,df_data['HG_%'].max()*1.2)
    ax.set_ylim(0,df_data['dp12_psi'].max()*1.2)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    fig.tight_layout()
    title2=title.replace(":","")+'.jpg'
    fig.savefig(title2)

    x_max = np.arange(0, df_data['dp12_psi'].max()*1.2, 0.2)
    ax2.plot(x_max, x_max, color='black',linestyle='-', label='perfect match', linewidth=0.75)
    ax2.plot(x_max, x_max*0.8, 'r--', label='-20%', linewidth=0.75)
    ax2.plot(x_max, x_max*1.2, 'r-.', label='+20%',  linewidth=0.75)
    # ax2.set_xlabel('Exp head (m of water)', fontsize=8)       # SI
    # ax2.set_ylabel("Sim head (m of water)", fontsize=8)       # SI
    ax2.set_xlabel('Exp head (psi)', fontsize=8)       # field
    ax2.set_ylabel("Sim head (psi)", fontsize=8)       # field
    ax2.legend(frameon=False, fontsize=5)
    # ax2.set_xlim(0,x_max.max()/1.4223)       # SI
    # ax2.set_ylim(0,x_max.max()/1.4223)       # SI
    ax2.set_xlim(0,x_max.max())       # field
    ax2.set_ylim(0,x_max.max())       # field
    title='Error analysis: '+pump_name+' '+test_type+' '
    if Npump != 'none':
        title=title+str(Npump)+' RPM'
    if Pin_psi != 'none':
        title=title+' '+str(Pin_psi)+' psi'
    ax2.set_title(title, fontsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    fig2.tight_layout()
    title2=title.replace(":","")+'.jpg'
    fig2.savefig(title2)

    disconnect_db(conn)

def loss_study_single_phase(pump_name='Flex31', QBEM=5500.0,Qmax=5500.0,figure_name='Relevent Parameter Analysis'):
    conn, c = connect_db('ESP.db')
    df_data = pd.read_sql_query("SELECT * FROM Catalog_All;", conn)
    df_data = df_data[df_data.Pump == pump_name]
    df_data = df_data[df_data.Flow_bpd != 0]
    df_data=df_data.reset_index(drop=True)

    fig, ax = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    fig2, ax2 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    fig3, ax3 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    fig4, ax4 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)

    sgl = SinglePhaseModel(ESP[pump_name],QBEM)

    if sgl_model=='zhang_2015':
        sgl_cal = np.vectorize(sgl.sgl_calculate_old)           
    elif sgl_model=='zhang_2016':
        sgl_cal = np.vectorize(sgl.sgl_calculate_new)    
    elif sgl_model=='jiecheng_2017':
        sgl_cal = np.vectorize(sgl.sgl_calculate_jc) 
    elif sgl_model=='zhu_2018':
        sgl_cal = np.vectorize(sgl.sgl_calculate_2018)

    ABV = 1
    error = 1
    icon = 0
    QL = np.arange(Qmax/50)[1:] * 50.0 * bbl_to_m3 / 24.0 / 3600.0
    QBEM = QBEM * bbl_to_m3 / 24.0 / 3600.0 # * np.ones(QL.shape)
    DENL = DENW # * np.ones(QL.shape)
    N = 3600 # * np.ones(QL.shape)
    NS = ESP[pump_name]['NS'] # * np.ones(QL.shape)
    SGM = ESP[pump_name]['SGM'] # * np.ones(QL.shape)
    SN = ESP[pump_name]['SN']
    ST = ESP[pump_name]['ST']
    VISL = ESP[pump_name]['VISW']
    VISW = ESP[pump_name]['VISW']
    WC = ESP[pump_name]['WC']
    HP, HE, HF, HT, HD, HRE, HLKloss, QLK = sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)
    HI = HE-HD-HP

    QL = QL / (bbl_to_m3 / 24.0 / 3600.0)
    QBEM = QBEM / (bbl_to_m3 / 24.0 / 3600.0)

    data = pd.DataFrame({'QL':QL, 'HP':HP, 'HE': HE, 'HF':HF, 'HT':HT, 'HD':HD, 'HI':HI, 'HRE':HRE, 'HLKloss':HLKloss, 'QLK':QLK})
    data.drop(data[data.HP<0].index, inplace=True)

    if pump_name=='Flex31':
        pump_name='MTESP'
    
    ax.plot(data.QL, data.HE, linestyle='-', label='Eular head',
                                                    c='C{}'.format(0), linewidth=0.75)       # field
    ax.plot(data.QL, data.HP, linestyle='-', label='Pump head',
                                                    c='C{}'.format(5), linewidth=0.75)       # field
    ax.plot(data.QL, data.HF+data.HT*0.05, linestyle=':', label='Friction loss',
                                                    c='C{}'.format(1), linewidth=0.75)       # field
    ax.plot(data.QL, data.HT, linestyle=':', label='Turning loss',
                                                    c='C{}'.format(2), linewidth=0.75)       # field
    ax.plot(data.QL, data.HRE, linestyle=':', label='Recirculation loss',
                                                    c='C{}'.format(3), linewidth=0.75)       # field
    ax.plot(data.QL, data.HLKloss+data.HRE*0.05, linestyle=':', label='Leakage loss',
                                                    c='C{}'.format(4), linewidth=0.75)       # field
    # ax.scatter(df_data.Flow_bpd, df_data.DP_psi,linewidths=0.75,s=8)

    ax.set_ylabel('Head/losses (psi)', fontsize=8)
    ax.set_xlabel('Liquid flow rate Q$_{L}$ (BPD)', fontsize=8)
    ax.set_xlim(0, data.QL.max()*1.2)
    ax.set_ylim(0, data.HE.max()*1.2)
    ax.set_title('Head losses '+ '('+pump_name+')', fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=5)

    ax2.plot(data.QL, data.QLK, linestyle='-', label='Leakage flow rate Q$_{LK}$ (BPD)',
                                                    c='C{}'.format(0), linewidth=0.75)       # field

    ax2.set_ylabel('Leakage flow rate Q$_{LK}$ (BPD)', fontsize=8)
    ax2.set_xlabel('Liquid flow rate Q$_{L}$ (BPD)', fontsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    ax2.set_xlim(0, data.QL.max()*1.2)
    # ax2.set_ylim(0, QLK.max()*1.2)
    ax2.set_title('Leakage flow rate analysis '+ '('+pump_name+')', fontsize=8)
    ax2.legend(frameon=False, fontsize=8)

    ax3.plot(data.QL, data.HE, linestyle='-', label='Eular head',
                                                    c='black'.format(0), linewidth=0.75)       # field
    ax3.plot(data.QL, data.HE-data.HT, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.QL, data.HE, data.HE-data.HT*0.95,facecolor='C{}'.format(5), alpha = 0.7, label='Turning loss')

    ax3.plot(data.QL, data.HE-data.HF-data.HT*0.95, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.QL, data.HE-data.HT*0.95, data.HE-data.HF-data.HT,facecolor='C{}'.format(1), label='Friction loss')

    ax3.plot(data.QL, data.HE-data.HF-data.HT-data.HRE, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.QL, data.HE-data.HF-data.HT, data.HE-data.HF-data.HT-data.HRE*0.95, facecolor='C{}'.format(2), alpha = 0.7, label='Recirculation loss')

    ax3.plot(data.QL, data.HE-data.HF-data.HT-data.HRE*0.95-data.HLKloss, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.QL,data.HE-data.HF-data.HT-data.HRE*0.95, data.HE-data.HF-data.HT-data.HRE-data.HLKloss, facecolor='C{}'.format(3), alpha = 0.7, label='Leakage loss')

    ax3.plot(data.QL, data.HP, linestyle=':', label='Pump head',
                                                    c='black', linewidth=0.75)       # field
    
    # ax3.scatter(df_data.Flow_bpd, df_data.DP_psi,linewidths=0.75,s=8, label='Test head')

    ax3.set_ylabel('Head/losses (psi)', fontsize=8)
    ax3.set_xlabel('Liquid flow rate Q$_{L}$ (BPD)', fontsize=8)
    ax3.set_xlim(0, data.QL.max()*1.2)
    ax3.set_ylim(0, data.HE.max()*1.2)
    ax3.set_title('Pump head and head losses analysis ' + '('+pump_name+')', fontsize=8)
    ax3.xaxis.set_tick_params(labelsize=8)
    ax3.yaxis.set_tick_params(labelsize=8)
    ax3.legend(frameon=False, fontsize=5)

    # ax4.plot(data.QL, data.HE, linestyle='-', label='Eular head',
    #                                                 c='C{}'.format(0), linewidth=0.75)       # field
    ax4.plot(data.QL, data.HP, linestyle='-', label='Pump head',
                                                    c='C{}'.format(5), linewidth=0.75)       # field
    ax4.plot(data.QL, data.HI, linestyle=':', label='Impeller loss',
                                                    c='C{}'.format(1), linewidth=0.75)       # field
    ax4.plot(data.QL, data.HD, linestyle=':', label='Diffuser loss',
                                                    c='C{}'.format(2), linewidth=0.75)       # field
    # ax4.scatter(df_data.Flow_bpd, df_data.DP_psi,linewidths=0.75,s=8, label='Pump head')


    ax4.set_ylabel('Head/losses (psi)', fontsize=8)
    ax4.set_xlabel('Liquid flow rate Q$_{L}$ (BPD)', fontsize=8)
    ax4.set_xlim(0, data.QL.max()*1.2)
    ax4.set_ylim(0, data.HP.max()*1.2)
    ax4.set_title('Head losses in impeller and diffuser '+ '('+pump_name+')', fontsize=8)
    ax4.xaxis.set_tick_params(labelsize=8)
    ax4.yaxis.set_tick_params(labelsize=8)
    ax4.legend(frameon=False, fontsize=5)


    fig.tight_layout()
    figure_name1 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Relevent parameters/'+figure_name+' HE.jpg'
    fig.savefig(figure_name1)
    fig2.tight_layout()
    figure_name2 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Relevent parameters/'+figure_name+' QLK.jpg'
    fig2.savefig(figure_name2)
    fig3.tight_layout()
    figure_name3 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Relevent parameters/'+figure_name+' HP.jpg'
    fig3.savefig(figure_name3)
    fig4.tight_layout()
    figure_name4 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Relevent parameters/'+figure_name+' HI_HD.jpg'
    fig4.savefig(figure_name4)
    print('complete')

def loss_study_gas_liquid_mapping(pump_name='Flex31', QBEM=5500.0,Qmax=5500.0,QG=300,N=3600,Pin_psi=35,Tin_F=60,flg ='F',figure_name='Relevent Parameter Analysis',case_name = 'Flex31_mapping'):
    
    """
    :param QG: constant gas flow rate bpd
    :param maxQL: maximum liquid flow rate bpd
    :param N: rotational speed rpm
    :param p: array for gas pressure psi
    :param t: array for temperature F
    :return: dataframe of predicted pump heads under mapping flow, the column names: zhu, Barrios, sun
    """

    ''' mapping '''
    # data
    conn, c = connect_db('ESP.db')
    df_data = pd.read_sql_query("SELECT * FROM " + 'All_pump' + ";", conn)
    df_data=df_data[(df_data.Case == case_name) & (df_data.Test == 'Mapping')]
    if N != 'none':
        df_data=df_data[(df_data.TargetRPM == N)]
    if Pin_psi != 'none':
        df_data=df_data[(df_data.TargetP_psi == Pin_psi)]
    if df_data.shape[0] < 1:
        print('No data selected')
        return
    df_data=df_data[df_data.TargetQG_bpd == QG]
    df_data=df_data.reset_index(drop=True)

    gl = GasLiquidModel(ESP[pump_name], QBEM)
    gl_cal = np.vectorize(gl.gl_calculate_new)

    QL =  np.arange(0.01, 1.1, 0.02) * Qmax * bbl_to_m3 / 24.0 / 3600.0

    QG = df_data['QG_bpd'].mean() * bbl_to_m3 / 24.0 / 3600.0 * np.ones(QL.shape)

    QBEM = QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(QL.shape)
    DENL = DENW * np.ones(QL.shape)
    DENG = gasdensity(Pin_psi, Tin_F, 0)
    DENG = DENG * np.ones(QL.shape)
    NS = gl.NS * np.ones(QL.shape)
    N = N * np.ones(QL.shape)
    SGM = gl.SGM * np.ones(QL.shape)
    SN = gl.SN * np.ones(QL.shape)
    ST = gl.ST * np.ones(QL.shape)
    VISL = gl.VISL * np.ones(QL.shape)
    VISG = gl.VISG * np.ones(QL.shape)
    WC = gl.WC * np.ones(QL.shape)

    # flg for different drag coefficient model
    flgz = np.empty(QL.shape, dtype='str')
    flgs = np.empty(QL.shape, dtype='str')
    flgb = np.empty(QL.shape, dtype='str')
    flgzz = np.empty(QL.shape, dtype='str')
    flgh = np.empty(QL.shape, dtype='str')
    flgz[:] = 'F'   # new Zhu model
    flgb[:] = 'B'   # Barrios
    flgs[:] = 'S'   # Sun
    flgzz[:] = 'Z'  # old Zhu model
    flgh[:] = 'H'   # homogenous model
    if flg == 'all':
        PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgz)
        PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgs)
        PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                            SN, ST, VISG, VISL, VISW, WC, flgb)
        PPZZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgzz)
        PPH, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgh)
    if flg == 'F':
        PPZ, PE, PF, PT, DB, PRE, PLK, QLK, GV= gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgz)
        PPS=PPZ
        PPB=PPZ
        PPZZ=PPZ
        PPH=PPZ
    if flg == 'B':
        PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                            SN, ST, VISG, VISL, VISW, WC, flgb)
        PPZ=PPB
        PPS=PPB
        PPZZ=PPB
        PPH=PPB
    if flg == 'H':
        PPH, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgh)
        PPZ=PPB
        PPS=PPB
        PPB=PPB
        PPZZ=PPB
    if flg == 'S':
        PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgs)
        PPZ=PPS
        PPB=PPS
        PPZZ=PPS
        PPH=PPS
    if flg == 'Z':
        PPZZ, PE, PF, PT, DB, PRE, PLK, QLK, GV= gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgzz)
        PPZ=PPZZ
        PPS=PPZZ
        PPB=PPZZ
        PPH=PPZZ

    QL = QL / (bbl_to_m3 / 24.0 / 3600.0)
    data = pd.DataFrame({'QL':QL, 'HP':PPZ, 'HE':PE, 'HT':PT, 'HF':PF, 'HRE':PRE, 'HLKloss':PLK, 'QLK':QLK, 'GV':GV, 'DB':DB})
    data.drop(data[data.HP<0].index, inplace=True)

    fig, ax = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    fig2, ax2 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    fig3, ax3 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax3_2 = ax3.twinx()
    fig4, ax4 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax4_2 = ax4.twinx()
    if pump_name=='Flex31':
        pump_name='MTESP'

    ax3.plot(data.QL, data.HE, linestyle='-', label='Eular head',
                                                    c='black', linewidth=0.75)       # field
    ax3.plot(data.QL, data.HE-data.HT, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.QL, data.HE, data.HE-data.HT*0.95,facecolor='C{}'.format(5), alpha = 0.7, label='Turning loss')

    ax3.plot(data.QL, data.HE-data.HF-data.HT*0.95, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.QL, data.HE-data.HT*0.95, data.HE-data.HF-data.HT, facecolor='C{}'.format(1), label='Friction loss')

    ax3.plot(data.QL, data.HE-data.HF-data.HT-data.HRE, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.QL, data.HE-data.HF-data.HT, data.HE-data.HF-data.HT-data.HRE*0.95, facecolor='C{}'.format(2), alpha = 0.7, label='Recirculation loss')

    ax3.plot(data.QL, data.HE-data.HF-data.HT-data.HRE*0.95-data.HLKloss, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.QL,data.HE-data.HF-data.HT-data.HRE*0.95, data.HE-data.HF-data.HT-data.HRE-data.HLKloss, facecolor='C{}'.format(3), alpha = 0.7, label='Leakage loss')


    ax3.plot(data.QL, data.HP, linestyle=':', label='Pump head',
                                                    c='black', linewidth=0.75)       # field
    ax3.scatter(df_data['QL_bpd'],df_data['dp12_psi'],label='Test head', 
                facecolor='C{}'.format(0),marker=symbols[0],linewidths=0.75, s=8)       # field
    

    ax3_2.plot(data.QL, data.QLK, linestyle='-', label='Leakage flow rate (BPD)',
                                                    c='C{}'.format(8), linewidth=0.75)       # field

    ax3.set_ylabel('Head/losses (psi)', fontsize=8)
    ax3.set_xlabel('Liquid flow rate Q$_L$ (BPD)', fontsize=8)
    ax3.set_xlim(0, data.QL.max()*1.2)
    ax3.set_ylim(0, data.HE.max()*1.2)
    ax3.set_title(pump_name+' head losses analysis mapping test $Q_G$ = ' + str(round(df_data['QG_bpd'].mean())) + ' bpd ', fontsize=8)
    ax3.xaxis.set_tick_params(labelsize=8)
    ax3.yaxis.set_tick_params(labelsize=8)
    ax3.legend(frameon=False, fontsize=5,loc='upper left')
    ax3_2.legend(frameon=False, fontsize=5,loc='upper right')
    # ax3_2.legend(frameon=False, fontsize=5)
    ax3_2.set_ylabel('Leakage flow rate Q$_{LK}$ ((BPD)', fontsize=8)

    # ax4.plot(data.QL, data.HE, linestyle='-', label='Eular head',
    #                                                 c='C{}'.format(0), linewidth=0.75)       # field
    ax4.plot(data.QL, data.DB, linestyle='--', label='Bubble size (m)',
                                                    c='C{}'.format(5), linewidth=0.75)       # field
    # ax4.plot(data.QL, data.HI, linestyle=':', label='Impeller loss',
    #                                                 c='C{}'.format(1), linewidth=0.75)       # field
    # ax4.plot(data.QL, data.HD, linestyle=':', label='Diffuser loss',
    #                                                 c='C{}'.format(2), linewidth=0.75)       # field
    # ax4.plot(data.QL, data.HRE, linestyle=':', label='Recirculation loss',
    #                                                 c='C{}'.format(3), linewidth=0.75)       # field
    # ax4.plot(data.QL, data.HLKloss, linestyle=':', label='Leakage loss',
    #                                                 c='C{}'.format(4), linewidth=0.75)       # field
    # ax4.scatter(df_data.Flow_bpd, df_data.DP_psi,linewidths=0.75,s=8, label='Pump head')
    ax4_2.plot(data.QL, data.GV, linestyle='-', label='$\\alpha_{G}$ (-)',
                                                    c='C{}'.format(0), linewidth=0.75)       # field

    ax4.set_ylabel('Gas bubble size (m)', fontsize=8)
    ax4.set_xlabel('Liquid flow rate Q$_{L}$ (BPD)', fontsize=8)
    ax4.set_xlim(0, data.QL.max()*1.2)
    ax4.set_ylim(0, data.DB.max()*1.2)
    ax4.set_title(pump_name+ ' gas bubble size and $\\alpha_{G}$ mapping test', fontsize=8)
    ax4.xaxis.set_tick_params(labelsize=8)
    ax4.yaxis.set_tick_params(labelsize=8)
    ax4.legend(frameon=False, fontsize=5,loc='upper left')
    ax4_2.set_ylabel('$\\alpha_{G}$ (-)', fontsize=8)
    ax4_2.legend(frameon=False, fontsize=5,loc='upper right')




    # fig.tight_layout()
    # figure_name1 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+'GV.jpg'
    # fig.savefig(figure_name1)
    # fig2.tight_layout()
    # figure_name2 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+'QLK.jpg'
    # fig2.savefig(figure_name2)
    fig3.tight_layout()
    figure_name3 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+' HP mapping.jpg'
    fig3.savefig(figure_name3)
    fig4.tight_layout()
    figure_name4 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+' bubble size mapping.jpg'
    fig4.savefig(figure_name4)
    data.to_csv('/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+'.csv')

    print('complete')

def loss_study_gas_liquid_surging(pump_name='Flex31', QBEM=5500.0,HGmax=0.2,QL=3100,N=3600,Pin_psi=35,Tin_F=60,flg ='F',figure_name='Relevent Parameter Analysis', case_name = 'Flex31_surging'):
    
    """
    :param QG: constant gas flow rate bpd
    :param maxQL: maximum liquid flow rate bpd
    :param N: rotational speed rpm
    :param p: array for gas pressure psi
    :param t: array for temperature F
    :return: dataframe of predicted pump heads under mapping flow, the column names: zhu, Barrios, sun
    """

    ''' surging '''
    # data
    conn, c = connect_db('ESP.db')
    df_data = pd.read_sql_query("SELECT * FROM " + 'All_pump' + ";", conn)
    df_data=df_data[(df_data.Case == case_name) & (df_data.Test == 'Surging')]
    if N != 'none':
        df_data=df_data[(df_data.TargetRPM == N)]
    if Pin_psi != 'none':
        df_data=df_data[(df_data.TargetP_psi == Pin_psi)]
    if df_data.shape[0] < 1:
        print('No data selected')
        return
    df_data=df_data[df_data.TargetQL_bpd == QL]
    df_data=df_data.reset_index(drop=True)

    sgl = SinglePhaseModel(ESP[pump_name], QBEM)
    gl = GasLiquidModel(ESP[pump_name], QBEM)
    sgl_cal = np.vectorize(sgl.sgl_calculate_new) #four options: old (zhu and zhang), new (Dr. Zhang update), jc (jiecheng), 2018 (jiecheng-jianjun)
    gl_cal = np.vectorize(gl.gl_calculate_new)  #old, new

    GF = np.arange(0.01, HGmax + 0.02, 0.015)
    if pump_name == 'TE2700':
        QL = df_data.QL_bpd.mean()
    QL = QL * bbl_to_m3 / 24.0 / 3600.0 * np.ones(GF.shape)
    QBEM = QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(GF.shape)
    QG = GF / (1 - GF) * QL
    DENL = gl.DENL * np.ones(GF.shape)
    DENG = gasdensity(Pin_psi, Tin_F, 0)
    DENG = DENG * np.ones(GF.shape)
    N = N * np.ones(GF.shape)
    NS = gl.NS * np.ones(GF.shape)
    SGM = gl.SGM * np.ones(GF.shape)
    SN = gl.SN * np.ones(GF.shape)
    ST = gl.ST * np.ones(GF.shape)
    VISL = gl.VISL * np.ones(GF.shape)
    VISG = gl.VISG * np.ones(GF.shape)
    WC = gl.WC * np.ones(GF.shape)

    # flg for different drag coefficient model
    
    flgzz = np.empty(GF.shape, dtype='str')
    flgh = np.empty(GF.shape, dtype='str')
    flgz = np.empty(GF.shape, dtype='str')
    flgs = np.empty(GF.shape, dtype='str')
    flgb = np.empty(GF.shape, dtype='str')
    flgzz[:] = 'Z'
    flgz[:] = 'F'
    flgb[:] = 'B'
    flgs[:] = 'S'
    flgh[:] = 'H'

    HP, _, _, _, _, _, _, _ = sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)

    # PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
    #                                      WC, flgz)
    # df = pd.DataFrame({'gf': GF, 'zhu': PPZ/HP})

    # PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
    #                                      WC, flgs)
    #df = pd.DataFrame({'gf': GF, 'sun': PPS/HP})
    
    # PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
    #                                      SN, ST, VISG, VISL, VISW, WC, flgb)
    # df = pd.DataFrame({'gf': GF, 'Barrios': PPB/HP})
    

    # df = pd.DataFrame({'gf': GF, 'zhu': PPZ/HP, 'sun': PPS/HP, 'Barrios': PPB/HP})
    # df = pd.DataFrame({'gf': GF, 'zhu': PPZ, 'sun': PPS, 'Barrios': PPB})

    if flg == 'all':
        PPZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgz)
        PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgs)
        PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                            SN, ST, VISG, VISL, VISW, WC, flgb)
        PPZZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgzz)
        PPH, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgh)
    if flg == 'F':
        # PPZ, PE, PF, PT, DB, CD, DB, VSR, GV
        PPZ, PE, PF, PT, DB, PRE, PLK, QLK, GV= gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgz)
        PPS=PPZ
        PPB=PPZ
        PPZZ=PPZ
        PPH=PPZ
    if flg == 'B':
        PPB, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM,
                                            SN, ST, VISG, VISL, VISW, WC, flgb)
        PPZ=PPB
        PPS=PPB
        PPZZ=PPB
        PPH=PPB
    if flg == 'H':
        PPH, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgh)
        PPZ=PPB
        PPS=PPB
        PPB=PPB
        PPZZ=PPB
    if flg == 'S':
        PPS, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgs)
        PPZ=PPS
        PPB=PPS
        PPZZ=PPS
        PPH=PPS
    if flg == 'Z':
        PPZZ, _, _, _, _, _, _, _, _ = gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                            WC, flgzz)
        PPZ=PPZZ
        PPS=PPZZ
        PPB=PPZZ
        PPH=PPZZ


    QL = QL / (bbl_to_m3 / 24.0 / 3600.0)
    data = pd.DataFrame({'GVF':GF, 'HP':PPZ, 'HE':PE, 'HT':PT, 'HF':PF, 'HRE':PRE, 'HLKloss':PLK, 'QLK':QLK, 'GV':GV, 'DB':DB})
    data.to_csv('/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+'.csv')
    data.drop(data[data.HP<0].index, inplace=True)

    fig, ax = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    fig2, ax2 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    fig3, ax3 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax3_2 = ax3.twinx()
    fig4, ax4 = plt.subplots(dpi=600, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax4_2 = ax4.twinx()
    if pump_name=='Flex31':
        pump_name='MTESP'

    ax3.plot(data.GVF, data.HE, linestyle='-', label='Eular head',
                                                    c='black', linewidth=0.75)       # field
    ax3.plot(data.GVF, data.HE-data.HT, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.GVF, data.HE, data.HE-data.HT*0.95,facecolor='C{}'.format(5), alpha = 0.7, label='Turning loss')

    ax3.plot(data.GVF, data.HE-data.HF-data.HT*0.95, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.GVF, data.HE-data.HT*0.95, data.HE-data.HF-data.HT, facecolor='C{}'.format(1), label='Friction loss')

    ax3.plot(data.GVF, data.HE-data.HF-data.HT-data.HRE, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.GVF, data.HE-data.HF-data.HT, data.HE-data.HF-data.HT-data.HRE*0.95, facecolor='C{}'.format(2), alpha = 0.7, label='Recirculation loss')

    ax3.plot(data.GVF, data.HE-data.HF-data.HT-data.HRE*0.95-data.HLKloss, linestyle=' ',
                                                    c='black', linewidth=0.75)       # field
    ax3.fill_between(data.GVF,data.HE-data.HF-data.HT-data.HRE*0.95, data.HE-data.HF-data.HT-data.HRE-data.HLKloss, facecolor='C{}'.format(3), alpha = 0.7, label='Leakage loss')


    ax3.plot(data.GVF, data.HP, linestyle=':', label='Pump head',
                                                    c='black', linewidth=0.75)       # field
    ax3.scatter(df_data['HG_%']/100*0.9,df_data['dp12_psi'],label='Test head', 
                facecolor='C{}'.format(0),marker=symbols[0],linewidths=0.75, s=8)       # field
    

    ax3_2.plot(data.GVF, data.QLK, linestyle='-', label='Leakage flow rate (BPD)',
                                                    c='C{}'.format(8), linewidth=0.75)       # field
    
    ax3.set_ylabel('Head/losses (psi)', fontsize=8)
    ax3.set_xlabel('Inlet GVF', fontsize=8)
    ax3.set_xlim(0, data.GVF.max()*1.2)
    ax3.set_ylim(0, data.HE.max()*1.2)
    ax3.set_title(pump_name +' head losses analysis surging test $Q_L$ = ' + str(round(df_data['QL_bpd'].mean())) + ' bpd ', fontsize=8)
    ax3.xaxis.set_tick_params(labelsize=8)
    ax3.yaxis.set_tick_params(labelsize=8)
    ax3.legend(frameon=False, fontsize=5)
    ax3_2.legend(frameon=False, fontsize=5,loc='upper left')
    ax3_2.set_ylabel('Leakage flow rate Q$_{LK}$ (BPD)', fontsize=8)


    ax4.plot(data.GVF, data.DB, linestyle='--', label='Bubble size (m)',
                                                    c='C{}'.format(5), linewidth=0.75)       # field
    ax4_2.plot(data.GVF, data.GV, linestyle='-', label='$\\alpha_{G}$ (-)',
                                                    c='C{}'.format(0), linewidth=0.75)       # field

    ax4.set_ylabel('Gas bubble size (m)', fontsize=8)
    ax4.set_xlabel('Inlet GVF', fontsize=8)
    ax4.set_xlim(0, data.GVF.max()*1.2)
    ax4.set_ylim(0, data.DB.max()*1.2)
    ax4.set_title(pump_name +' gas bubble size and $\\alpha_{G}$ (surging test)', fontsize=8)
    ax4.xaxis.set_tick_params(labelsize=8)
    ax4.yaxis.set_tick_params(labelsize=8)
    ax4.legend(frameon=False, fontsize=5,loc='upper left')
    ax4_2.set_ylabel('$\\alpha_{G}$ (-)', fontsize=8)
    ax4_2.legend(frameon=False, fontsize=5,loc='upper right')




    # fig.tight_layout()
    # figure_name1 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+' GV_surging.jpg'
    # fig.savefig(figure_name1)
    # fig2.tight_layout()
    # figure_name2 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+' QLK_surging.jpg'
    # fig2.savefig(figure_name2)
    fig3.tight_layout()
    figure_name3 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+' HP_surging.jpg'
    fig3.savefig(figure_name3)
    fig4.tight_layout()
    figure_name4 = '/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/Results/Gas liquid parameters/'+figure_name+' bubble size_surging.jpg'
    fig4.savefig(figure_name4)

    print('complete')

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

def ESP_curve_plot (ESP_input, inputValues, QBEM, VISL_list=[1,10,50,100,300,500,700,1000]):
    try:
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

    except:
        print('输入流动数据有误')
        return
    try:
        fig, ax = plt.subplots(dpi=300, figsize = (3.33,2.5), nrows=1, ncols=1)
        for VISO_test in VISL_list:
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

            ax.plot(QL*bbl_to_m3, HP*0.3048/0.433*ESP_input['SN'], label='原油粘度='+str(int(VISO_test*1000))+'cp', linewidth=0.75)
            ax.set_ylim(0)

            ax.set_xlabel('排量（m3/d）', fontsize=8)
            ax.set_ylabel('扬程（m）', fontsize=8)
            ax.set_title('井电潜泵扬程预测', fontsize=8)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
        
        ax.legend(frameon=False, fontsize=6)
        fig.tight_layout()
    except:
        print('电潜泵参数有误')
        return

##########################
if __name__ == "__main__":

    conn, c = connect_db('ESP.db')
    '''水测试'''
    # All_pump_water('ESP.db', 'All_pump','Gamboa_GC6100', 'GC6100', 0,  QBEM_default['GC6100'], 'SGL', [3000,2400,1800,1500])
    # All_pump_water('ESP.db', 'All_pump','Flex31_water', 'Flex31', 0, QBEM_default['Flex31'], 'SGL', [3600,3000,2400,1800])
    # All_pump_water('ESP.db', 'All_pump','TE2700_water', 'TE2700', 0,  QBEM_default['TE2700'], 'SGL', [3500,3000,2400,1800]) 
    # All_pump_water('ESP.db', 'All_pump','P100_Viscosity', 'P100', 0,  QBEM_default['P100'], 'Viscosity', [3600]) 

    '''油测试'''
    # vis_te2700_plot('ESP.db')

    '''气液测试'''
    '''GC6100'''
    # pump_name = 'GC6100'
    # qglist = [1,2,3,4]
    # Nlist = [3000, 1500]
    # Plist = [250]
    # for N in Nlist:
    #     for P in Plist:
    #         All_pump_mapping('ESP.db', 'All_pump', 'Gamboa_GC6100', qglist, 'Qw (bpd)', 'Head (psi)','GC6100','Mapping',N,P,True)

    '''Flex31'''
    pump_name = 'Flex31'
    qglist = [80,300,460]
    Nlist = [3600]
    Plist = [35]
    for N in Nlist:
        for P in Plist:
            All_pump_mapping('ESP.db', 'All_pump', 'Flex31_mapping', qglist, 'Qw (bpd)', 'Head (psi)','Flex31','Mapping',N,P,True)
            
    '''TE2700'''
    # pump_name = 'TE2700'   #????
    # qglist = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    # qllist = [0.75,1,1.25]
    # Nlist = [3500,1800]

    # Plist = [150]
    # for N in Nlist:
    #     for P in Plist:
    #         All_pump_mapping('ESP.db', 'All_pump', 'TE2700_mapping_JJZ', qglist, 'Qw (bpd)', 'Head (psi)','TE2700','Mapping',N,P,True)

    # print('complete')
    plt.show()