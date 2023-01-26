
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import webbrowser
# from ESP_Class import *
from ESP_simple_all_in_one import *
from SPSA import *
from Pipe_model import *

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
m3s_to_bpd = 543439.65056533
sgl_model='zhang_2016'

def Temperature_C_K(T_C):
    T_K = T_C + 273.15
    return T_K

def Temperature_C_F(T_C):
    T_F = (T_C*9/5)+32
    return T_F

def read_input (filename= '22b.xlsx'):
    df_well_profile = pd.read_excel(filename, sheet_name='Sheet1')
    df_well_profile.rename(columns={df_well_profile.columns[0]: "ID", df_well_profile.columns[1]: "MD", df_well_profile.columns[2]: "DL", df_well_profile.columns[3]: "Angle", df_well_profile.columns[5]: "TVD", df_well_profile.columns[6]: "HD"}, inplace=True)

    # fig, ax = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
    # ax.plot(df_well_profile.HD, -df_well_profile.TVD, linewidth=0.75)

    # ax.set_xlabel('水平位移（m）', fontsize=8)
    # ax.set_ylabel('垂深（m）', fontsize=8)
    # ax.set_title('井眼轨迹', fontsize=8)
    # ax.xaxis.set_tick_params(labelsize=8)
    # ax.yaxis.set_tick_params(labelsize=8)
    # ax.legend(frameon=False, fontsize=6)
    # fig.tight_layout()
    # wellName = filename.replace('.xlsx','')
    # fig.savefig(wellName+' well profile'+'.jpg')

    return df_well_profile

def P_bottom_Cal (inputValues, df_well_profile, Qg_res = 10, Qo_res = 0.1, Qw_res = 0.1, QBEM=6000, ESP_input=ESP, ESP_empirical_H=None):
    P=inputValues['Surface_line_pressure']
    T=inputValues['Surface_T']
    
    
    A_pipe = 3.14/4*inputValues['Pipe_diameter']**2.0
    D = inputValues['Pipe_diameter']
    ED = inputValues['Roughness']
    DENO = inputValues['Liquid_relative_density'] * 1000
    WC = inputValues['Water_cut'] 
    DENL = DENO*(100-WC)/100+1000*WC/100
    GLR = inputValues['GLR'] 
    # ESP_bottom = inputValues['ESP_Depth']
    ESP_top = inputValues['ESP_Depth'] - inputValues['ESP_length']
    for i in range(0, df_well_profile.shape[0], 1):
        if i>1 and df_well_profile.MD[i-1] < ESP_top and df_well_profile.MD[i] > ESP_top:
            i_ESP_top = i
        # if i>1 and df_well_profile.MD[i-1] < ESP_bottom and df_well_profile.MD[i] > ESP_bottom:
        #     i_ESP_bottom = i

    

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
    
        # if i>1 and i >= i_ESP_bottom and ESP_complete == 'No':
        if i == i_ESP_top:
            if ESP_empirical_H != None:
                Pout = P
                P = P-ESP_empirical_H
                Pin = P
            else:
                # di = i_ESP_bottom - i_ESP_top
                P_ave = P/1.5
                GLR = GLR*DENG_stg/DENG

                while True:
                    QL = np.array((Qw_res+Qo_res)*m3s_to_bpd)
                    # DENG = gas_density(inputValues['Gas_relative_density'], T, P_ave)
                    # DENG_stg = gas_density(inputValues['Gas_relative_density'], 288, 101325)
                    # QG = Qg_res/DENG*DENG_stg
                    # GVF = QG/(QG+QL)
                    HP =ESP_head (QBEM, ESP_input, QL, 0.001, 1000, VISO, DENO, VISG, WC, GLR, P_ave, T, 0.035, None)
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
    return P, df, Pin, Pout

def P_profile_plot (df_well_profile, df, 
            wellName, P_in, P_out):
    '''pressure and temperature'''
    fig, ax = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax.plot(df_well_profile.MD, df.List_P, label='压力（MPA）', linewidth=0.75)
    # for i in range(1,len(df.List_P)):
    #     if df.List_P[i]<df.List_P[i-1]:
    #         ax.scatter(df_well_profile.MD[i],P_in,color = 'black', label=('泵入口压力(%.2fMPA)'%(df.List_P[i-1])), marker='*', linewidth=0.3)
    #         ax.scatter(df_well_profile.MD[i-1],P_out,color = 'red', label=('泵出口压力(%.2fMPA)'%(df.List_P[i])), marker='*', linewidth=0.3)


    ax.set_ylim(0,max(df.List_P)*1.2)

    ax.set_xlabel('测深 (m)', fontsize=8)
    ax.set_ylabel('压力（MPa）', fontsize=8)
    ax.set_title(wellName+'井的压力和温度剖面', fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=6)

    # axx = ax.twinx()
    # axx.plot(df_well_profile.MD, df.List_T, color = 'r', label='温度', linewidth=0.75)
    # axx.set_ylabel('温度 (C)', fontsize=8)
    # # ax3.set_title(wellName+'井眼轨迹')
    # axx.yaxis.set_tick_params(labelsize=8)
    # axx.legend(frameon=False, loc='lower right', fontsize=6)
    fig.tight_layout()
    fig.savefig('result/'+wellName+'P'+'.jpg')


    '''flow pattern and inclination angle'''
    fig2, ax2 = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax2.plot(df_well_profile.MD, df.List_ID, label='流态', linewidth=0.75)
    ax2.set_ylim(0,6)
    ax2.legend(frameon=False, loc='upper left', fontsize=6)

    ax2.set_xlabel('测深 (m)', fontsize=8)
    ax2.set_ylabel('1:层流 2:气泡流 3:段塞流 4:环空流, 5:均质', fontsize=8)
    ax2.set_title(wellName+'井测深 vs. 流态/井斜', fontsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)

    ax3 = ax2.twinx()
    ax3.plot(df_well_profile.MD, df_well_profile.Angle, color = 'r', label='井斜', linewidth=0.75)
    ax3.set_ylabel('井斜 (DEG)', fontsize=8)
    # ax3.set_title(wellName+'井眼轨迹')
    ax3.yaxis.set_tick_params(labelsize=8)
    ax3.legend(frameon=False, loc='lower right', fontsize=6)
    fig2.tight_layout()
    fig2.savefig('result/'+wellName+'ID'+'.jpg')

    '''velocity'''
    fig4, ax4 = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax4.plot(df_well_profile.MD, df.List_VL, label='液相流速', linewidth=0.75)
    ax4.plot(df_well_profile.MD, df.List_VO, label='油流速', linewidth=0.75)
    ax4.plot(df_well_profile.MD, df.List_VW, label='水流速', linewidth=0.75)
    ax4.set_ylim(0)
    ax4.legend(frameon=False, loc='upper left', fontsize=6)

    ax4.set_xlabel('测深 (m)', fontsize=8)
    ax4.set_ylabel('液相流速 (m/s)', fontsize=8)
    ax4.set_title(wellName+'流速剖面', fontsize=8)
    ax4.xaxis.set_tick_params(labelsize=8)
    ax4.yaxis.set_tick_params(labelsize=8)

    ax44 = ax4.twinx()
    ax44.plot(df_well_profile.MD, df.List_VG, color = 'r', label='气流速', linewidth=0.75)
    ax44.set_ylabel('气相流速 (m/s)', fontsize=8)
    ax44.yaxis.set_tick_params(labelsize=8)
    ax44.legend(frameon=False, loc='lower right', fontsize=6)
    fig4.tight_layout()
    fig4.savefig('result/'+wellName+'velocity'+'.jpg')


    '''liquid holdup'''
    fig5, ax5 = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax5.plot(df_well_profile.MD, df.List_HL*100, label='模型', linewidth=0.75)
    ax5.plot(df_well_profile.MD, df.List_VL*df.List_HL/(df.List_VL*df.List_HL+df.List_VG*(1-df.List_HL))*100, label='均质假设', linewidth=0.75)
    miny = int(min(df.List_HL)*90) if int(min(df.List_HL)*90) > 0 else 0
    maxy = int(max(df.List_HL)*110) if int(max(df.List_HL)*110) < 100 else 100
    ax5.set_ylim(miny*0.9, maxy*1.1)
    ax5.legend(frameon=False, loc='upper left', fontsize=6)

    ax5.set_xlabel('测深 (m)', fontsize=8)
    ax5.set_ylabel('体积分数（%)', fontsize=8)
    ax5.set_title(wellName+'液体体积分数', fontsize=8)
    ax5.xaxis.set_tick_params(labelsize=8)
    ax5.yaxis.set_tick_params(labelsize=8)

    fig5.tight_layout()
    fig5.savefig('result/'+wellName+'HL'+'.jpg')

    '''VISG and ST'''
    fig6, ax6 = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax6.plot(df_well_profile.MD, df.List_VISG*1000, label='气体粘度cp', linewidth=0.75)
    ax6.set_ylim(0)
    ax6.legend(frameon=False, loc='upper left', fontsize=6)

    ax6.set_xlabel('测深 (m)', fontsize=8)
    ax6.set_ylabel('气体粘度cp', fontsize=8)
    ax6.set_title(wellName+'气体粘度cp', fontsize=8)
    ax6.xaxis.set_tick_params(labelsize=8)
    ax6.yaxis.set_tick_params(labelsize=8)

    ax66 = ax6.twinx()
    ax66.plot(df_well_profile.MD, df.List_DENG, color = 'r', label='气体密度kg/m$^3$', linewidth=0.75)
    ax66.set_ylabel('气体密度kg/m$^3$', fontsize=8)
    ax66.yaxis.set_tick_params(labelsize=8)
    ax66.legend(frameon=False, loc='lower right', fontsize=6)


    # ax666 = ax6.twinx()
    # ax666.plot(df_well_profile.MD, df.List_ST, color = 'r', label='表面张力N/m', linewidth=0.75)
    # ax666.set_ylabel('表面张力N/m', fontsize=8)
    # ax666.yaxis.set_tick_params(labelsize=8)
    # ax666.legend(frameon=False, loc='lower right', fontsize=6)

    fig6.tight_layout()
    fig6.savefig('result/'+wellName+'VISG DENG ST'+'.jpg')

    ''''''

    df = pd.DataFrame({'井眼轨迹': df_well_profile.MD, '压力MPA': df.List_P, '流态': df.List_ID, '液体流速m/s': df.List_VL, '油流速m/s':df.List_VO, 
                    '水流速m/s': df.List_VW, '气流速m/s': df.List_VG, '液体体积分数': df.List_HL, '温度C':df.List_T, '气体密度kg/m3': df.List_DENG, 
                    '气体粘度pas': df.List_VISG, '界面张力N/m': df.List_ST})
    df.to_excel('result/'+wellName+'节点数据.xls')

def Nodal (inputValues, df_well_profile, QL_list, GLR, Pwf_list, WC, QBEM, ESP_input, Q_oil, H_oil, wellName):
    Q_bottom = []
    P_bottom = []
    for ql in QL_list:
        Qw_res = ql*WC/100
        Qo_res = ql*(100-WC)/100/inputValues['Liquid_relative_density']
        Qg_res = ql*GLR
        P, _= P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input)
        Q_bottom.append(ql*24*3600)
        P_bottom.append(P/1e6)

    fig, ax = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
    ax.plot(Q_bottom, P_bottom, label='节点流出OPR曲线', linewidth=0.75)
    ax.legend(frameon=False)

    ax.set_xlabel('产液量（m3/s）', fontsize=8)
    ax.set_ylabel('井底压力（MPa）', fontsize=8)
    ax.set_title(well_name+'井产液量 vs. 井底压力', fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    
    Q_bottom_IPR = []
    P_bottom_IPR = []
    for pwf in Pwf_list:
        Qg_res, Ql_res = fgreservoir(inputValues['Reservoir_P'], pwf, inputValues['Reservoir_C'], inputValues['Reservoir_n'], inputValues['GLR'])
        Q_bottom_IPR.append(Ql_res*24*3600)
        P_bottom_IPR.append(pwf/1e6)

    ax.plot(Q_bottom_IPR, P_bottom_IPR, label='节点流入IPR曲线', linewidth=0.75)
    ax.scatter(Q_oil, H_oil, label='现场（%.2fm$^3$/d，%.2fMPA）' % (Q_oil,H_oil), marker='*', linewidth=0.75)
    ax.legend(frameon=False, fontsize=6)

    ax.set_xlabel('产液量（m3/d）', fontsize=8)
    ax.set_ylabel('井底压力（MPa）', fontsize=8)
    ax.set_title(well_name+'井节点分析', fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylim(0)
    ax.set_xlim(0)
    fig.tight_layout()
    fig.savefig('result/'+wellName+'Nodal'+'.jpg')

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
    VISL = VISO_in
    VISW = VISW_in
    VISG = VISG_in
    DENL = DENO_in
    DENW = DENW_in
    DENG_std = gas_density(0.7, 288, 101325)
    DENG = gas_density(0.7, T, P)
    P = P * np.ones(QL.shape)
    T = T * np.ones(QL.shape)
    N = ESP_input['N'] * np.ones(QL.shape) # * np.ones(QL.shape)
    NS = ESP_input['NS'] * np.ones(QL.shape) # * np.ones(QL.shape)
    SGM = ESP_input['SGM'] * np.ones(QL.shape) # * np.ones(QL.shape)
    SN = ESP_input['SN'] * np.ones(QL.shape)
    ST = ESP_input['ST'] * np.ones(QL.shape)
    WC = WC * np.ones(QL.shape)
    QL =  QL * bbl_to_m3 / 24.0 / 3600.0
    QBEM = QBEM * bbl_to_m3 / 24.0 / 3600.0 * np.ones(QL.shape)# * np.ones(QL.shape)
    if GVF == None:
        QG = QL * GLR * DENG_std/DENG
    else: 
        QG = QL * GVF
    if GLR == 0 or GVF == 0:
        sgl = SinglePhaseModel(ESP_input,QBEM)
        if sgl_model=='zhang_2016':
            sgl_cal = np.vectorize(sgl.sgl_calculate_new)  
        elif sgl_model=='zhu_2018':
            sgl_cal = np.vectorize(sgl.sgl_calculate_2018)
        HP, HE, HF, HT, HD, HRE, HLKloss, QLK= sgl_cal(QL, QBEM, DENL, DENW, N, NS, SGM, SN, ST, VISL, VISW, WC)
    else:
        sgl = SinglePhaseModel(ESP_input,QBEM)
        if sgl_model=='zhang_2016':
            sgl_cal = np.vectorize(sgl.sgl_calculate_new)  
        elif sgl_model=='zhu_2018':
            sgl_cal = np.vectorize(sgl.sgl_calculate_2018)
        gl = GasLiquidModel(ESP_input,QBEM)
        gl_cal = np.vectorize(gl.gl_calculate_new)
        flgz = 'Z'   # Jianjun Zhu simple version model
        HP, HE, HF, HT, HD, HRE, HLKloss, QLK, GV= gl_cal(QL, QG, QBEM, DENG, DENL, DENW, N, NS, SGM, SN, ST, VISG, VISL, VISW,
                                                WC, flgz)

    return HP

def ESP (ESP_input, inputValues, QBEM,well_name,Q_water=80,H_water=1650,Q_Oil=103,H_Oil=1215, VisL_target=523):
    QL= np.arange(0.005, 1.1, 0.025) * 5000.0
    VISO_in=inputValues['Liquid_viscosity']
    WC=inputValues['Water_cut']
    VISW_in=0.001
    VISG_in=inputValues['Gas_viscosity']
    DENO_in=inputValues['Liquid_relative_density']*1000
    DENW_in=1000
    GLR = inputValues['GLR']
    GOR = GLR/(1-WC/100)
    P = (inputValues['P_in']+inputValues['P_out'])/2*1e6
    T=inputValues['Pump_intake_T']
    O_W_ST = 0.035
    fig, ax = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)
    for VISO_test in [1,10,50,100,300,500,700,1000]:
        if VISO_test>VISO_in*1000:
            break
        if VISO_test == 1: 
            VISO_test = 1
            VISO_test = VISO_test/1000
            HP =ESP_head (QBEM, ESP_input, QL, VISW_in, DENW_in, VISO_test, DENO_in, VISG_in, WC, 0, P, T, O_W_ST, None)
        else:
            VISO_test = VISO_test/1000
            HP =ESP_head (QBEM, ESP_input, QL, VISW_in, DENW_in, VISO_test, DENO_in, VISG_in, WC, GLR, P, T, O_W_ST, None)
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
    HP =ESP_head (QBEM, ESP_input, QL, VISW_in, DENW_in, VISO_in, DENO_in, VISG, WC, GLR, P, T, O_W_ST, None)
    print ('ESP head at working condition: ',QL/(m3s_to_bpd/24/3600), 'm3d', HP*0.3048/0.433*ESP_input['SN'], 'm', GLR, ' GLR')
    fig.tight_layout()
    fig.savefig('ESP performance/'+well_name+'.jpg')

def ESP_curve_plot (ESP_input, inputValues, QBEM, VISL_list=[1,10,50,100,300,500,700,1000], ESP_empirical_class = None):
    try:
        QL= np.arange(0.005, 1., 0.025) * QBEM

        VISO_in=inputValues['Liquid_viscosity']
        WC=inputValues['Water_cut']
        # VISO_in=0.001
        VISW_in=0.001
        VISG_in=inputValues['Gas_viscosity']
        DENO_in=inputValues['Liquid_relative_density']*1000
        DENW_in=1000
        DENL = DENO_in*(100-WC)/100+DENW_in*WC/100
        DENG_std = gas_density(inputValues['Gas_relative_density'], 288, 101325)
        DENG = gas_density(inputValues['Gas_relative_density'], inputValues['Pump_intake_T'], 
                    (inputValues['P_in']+inputValues['P_out'])/2*1e6)
        GLR = inputValues['GLR']
        GOR = GLR/(1-WC/100)
        GLR=GLR*DENG_std/DENG
        # 22b well
        P = (inputValues['P_in']+inputValues['P_out'])/2*1e6
        T=inputValues['Pump_intake_T']
        O_W_ST = 0.035

    except:
        print('输入流动数据有误')
        return
    # try:
    fig, ax = plt.subplots(dpi=300, figsize = (3.33,2.5), nrows=1, ncols=1)


    # Power.HP = Power.HP * (90-10)/(1800-200)
    # BX.BX = BX.BX * (90-10)/(1800-200)
    # ZX.ZX = ZX.ZX * (90-10)/(1800-200)

    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # fig.subplots_adjust(left=0.25)
    # ax_twinx = ax.twinx()
    # ax_twinx2 = ax.twinx()
    # ax_twinx3 = ax.twinx()
    # ax_twinx2.spines.right.set_position(("axes", 1.2))
    # ax_twinx3.spines.left.set_position(("axes", -0.2))

    # p1, = ax.plot(Head.Q, Head.H, 'r', label='Head')
    # p2, = ax_twinx.plot(BX.Q,BX.BX, 'b', label = 'BX')
    # ax_twinx.plot(ZX.Q,ZX.ZX, 'b:', label = 'ZX')
    # p4, = ax_twinx3.plot(Power.Q,Power.HP, 'c', label = 'HP')

    # ax.set_ylim(0,Head.H.max()*1.2)
    # ax_twinx.set_ylim(0,100)
    # ax_twinx2.set_ylim(0,1)
    # ax_twinx3.set_ylim(0,100)

    # ax.set_xlabel("Flow rate")
    # ax.set_ylabel("Head")
    # ax_twinx.set_ylabel("Eff")
    # ax_twinx3.set_ylabel("Power")

    # ax.yaxis.label.set_color(p1.get_color())
    # ax_twinx.yaxis.label.set_color(p2.get_color())
    # ax_twinx3.yaxis.label.set_color(p4.get_color())

    # tkw = dict(size=4, width=1.5)
    # ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    # ax_twinx.tick_params(axis='y', colors=p2.get_color(), **tkw)
    # ax_twinx3.tick_params(axis='y', colors=p4.get_color(), **tkw)

    # ax_twinx3.yaxis.set_label_position('left')
    # ax_twinx3.yaxis.set_ticks_position('left')

    # ax.legend(loc='upper left')
    # ax_twinx.legend(loc='lower right')
    # ax_twinx2.legend(loc='upper right')
    # ax_twinx3.legend(loc='lower left')
    # fig.savefig('ori.jpg')
    # plt.show()



    ESP_input['N']=inputValues['ESP_N']

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
            # HP =Oil_curve(QBEM=QBEM, ESP_input=ESP_input, QL=QL, VISL_in=VISL_in, DENL_in=DENL_in)
            HP = ESP_head (QBEM, ESP_input, QL, VISW_in, DENW_in, VISO_test, DENO_in, VISG_in, WC, GLR, P, T, O_W_ST, None)
            if VISO_test>0.5:
                # check point
                QL1= np.arange(1,2,1) * 86 * m3s_to_bpd/24/3600
                HP1 = ESP_head (QBEM, ESP_input, QL1, VISW_in, DENW_in, VISO_test, DENO_in, VISG_in, WC, GLR, P, T, O_W_ST, None)

        ax.plot(QL*bbl_to_m3, HP*0.3048/0.433/(DENL/DENW_in), label='原油粘度='+str(int(VISO_test*1000))+'cp', linewidth=0.75)

    if ESP_empirical_class != None:
        pump = ESP_empirical_class.pump_name
        # ESP_empirical_class.calibrate_water_curve(pump)
        SSU = inputValues['SSU']
        _ = ESP_empirical_class.viscosity_calibrate(SSU, inputValues['ESP_N']*60/3600) 
        # _ = ESP_empirical_class.viscosity_calibrate(SSU, 60) 
        
        # solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP)
        Q = np.linspace(1, QBEM*bpd_to_m3d*1.5, 100)
        #viscosity pump curve
        ax.scatter(ESP_empirical_class.field_pump_curve['Q_m3d'], ESP_empirical_class.field_pump_curve['H_m'], label='现场:井'+str(pump))
        # ax.scatter(ESP_empirical_class.pump_curve_water_3600RPM['Q_m3d']*inputValues['ESP_N']/3600, 
        #                 ESP_empirical_class.pump_curve_water_3600RPM['H_m']*(inputValues['ESP_N']/3600)**2, label='出场水：井'+str(pump))
        
    ax.set_ylim(0, max(ESP_empirical_class.field_pump_curve['H_m'])*1.5)
    ax.set_xlim(0, max(ESP_empirical_class.field_pump_curve['Q_m3d'])*1.5)
    ax.set_xlabel('排量（m3/d）', fontsize=8)
    ax.set_ylabel('扬程（m）', fontsize=8)
    ax.set_title('井电潜泵扬程预测', fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig('ESP performance of well %s' %(inputValues['well_name']))
    # except:
    #     print('电潜泵参数有误')
    #     return

if __name__ == "__main__":


    '''绘制扬程曲线'''
                        
    # inputs = {"Surface_line_pressure": 2.8*1e6, "Pipe_diameter":0.088,        
    #             "Roughness":0,                   "Liquid_viscosity":0.523,
    #             "Liquid_relative_density":0.9521,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
    #             "WC": 75, 
    #             "Reservoir_C":4.880E-17,     "Reservoir_n":1, 
    #             "Reservoir_P":1.25e7,                "GOR":6.7,     "GLR":5, 
    #             "Geothermal_gradient":0.03, 
    #             "Surface_T":288, "ESP_stage":282, "ESP_RPM":2890, "ESP_length": 8, "ESP_D":0.098,
    #             "Surface_tension_GL":0.075,   "Surface_tension_OW":0.035,   ####
    #             "ESP_GOR1":6.7,    "ESP_GOR2":6.7, "ESP_GOR3":6.7,
    #             "ESP_RPM1":2890, "ESP_RPM2":2890,    "ESP_RPM3":2890, 
    #             "ESP_WC1":75,    "ESP_WC2":75,   "ESP_WC3":75,   
    #             "ESP_VISL1":1,    "ESP_VISL2":200,    "ESP_VISL3":500,     
    #             "SPSA_a":1e-5,   'SPSA_n':2,     "SPSA_iter":50,
    #                 # missing input
    #             "Pump_intake_T": 380, "P_in":3.12, "P_out":14.11, "ESP_Depth": 1320, "Q_ESP": 102.92
    #                 }
    # ESP_in = ESP_default['Flex31']
    # ESP_curve_plot(ESP_in,inputs,QBEM_default['Flex31'],[1,10,50,100,300,500,700,1000])
    # plt.show()

    ''''22b'''
    # well_name = '22b'
    # df_well_profile = read_input('22b.xlsx')
    # inputValues = {"Surface_line_pressure": 2.8*1e6, "Pipe_diameter":0.088,      
    #                 "Roughness":0,                   "Liquid_viscosity":0.523,
    #                 "Liquid_relative_density":0.9521,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
    #                     "Water_cut": 75, 
    #                 "Reservoir_C":4.880E-17,     "Reservoir_n":1, 
    #                 "Reservoir_P":1.25e7,               "GLR":5,                   "Geothermal_gradient":0.03, 
    #                 "Surface_T":288, "Pump_intake_T": 380, "P_in": 3.12, "P_out":14.11, 
    #                 "ESP_Depth": 1320, "ESP_SN":282, "ESP_N":2890, "ESP_length": 8}
    # Q_water = 80.36
    # H_water=1651.41
    # Q_oil = 103
    # H_oil = 1215
    # WC=inputValues['Water_cut']
    # GLR = inputValues['GLR']
    # Qg_res = Q_oil*GLR/24/3600
    # Qo_res = Q_oil*(1-WC/100)/24/3600
    # Qw_res = Q_oil*WC/100/24/3600
    # ESP_input=ESP_default['Flex31'].copy()
    # QBEM=6500
    # ESP_input['R2']*=0.94
    # ESP_input['SN']=inputValues['ESP_SN']
    # ESP_input['N']=inputValues['ESP_N']
    # ESP_Depth = inputValues['ESP_Depth']
    # VisL_target = inputValues['Liquid_viscosity']*1000
    # '''ESP'''
    # ESP(ESP_input,inputValues, QBEM, well_name,Q_water,H_water,Q_oil,H_oil,VisL_target)
    # '''Pressure profile'''
    # ql, P = Nodal_solve(inputValues, df_well_profile, GLR, WC, QBEM, ESP_input)
    # print('Nodal: ', ql,' m3d', P,' Mpa')
    # Qg_res = ql*GLR/24/3600
    # Qo_res = ql*(1-WC/100)/24/3600
    # Qw_res = ql*WC/100/24/3600
    # P, df = P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input)
    # P_profile_plot (df_well_profile, df, well_name, inputValues['P_in'], inputValues['P_out'])
    # '''Nodel'''
    # QL = np.array([ql/32,ql/16,ql/8,ql/4,ql/2,ql,ql*1.2,ql*1.5])    # m3/d
    # QL = QL/24/3600
    # Pwf = np.arange(0, int(inputValues['Reservoir_P']))
    # Nodal (inputValues, df_well_profile, QL, GLR, Pwf, WC, QBEM, ESP_input, Q_oil, df.List_P.iloc[-1], well_name)
    # print('complete')
    # plt.show()

    '''251E-1'''
    # well_name = '251E-1'
    # Q_water = 121
    # H_water=1783
    # Q_oil = 86  # field mix QL
    # H_oil = 1320    # pump head at field mix QL
    # P_in=1.24
    # P_out=14.28
    # df_well_profile = read_input(well_name+'.xlsx')
    # inputValues = {"Surface_line_pressure": 1.9*1e6, "Pipe_diameter":0.088,      
    #                 "Roughness":0,                   "Liquid_viscosity":0.734,
    #                 "Liquid_relative_density":0.9521,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
    #                     "Water_cut": 80, 
    #                 "Reservoir_C":1.68E-17,     "Reservoir_n":1, 
    #                 "Reservoir_P":1.83e7,               "GLR":5.4,                   "Geothermal_gradient":0.03, 
    #                 "Surface_T":288, "Pump_intake_T": 380, "P_in": 1.24, "P_out":14.28, 
    #                 "ESP_Depth": 1582, "ESP_SN":269, "ESP_N":2900, "ESP_length": 5.58+4.67}

    # '''pressure profile'''
    # WC=inputValues['Water_cut']
    # GLR = inputValues['GLR']
    # ESP_input=ESP_default['Flex31'].copy()
    # QBEM=6000
    # ESP_input['R2']*=1
    # ESP_input['SN']=inputValues['ESP_SN']
    # ESP_input['N']=inputValues['ESP_N']
    # ESP_Depth = inputValues['ESP_Depth']
    # VisL_target = inputValues['Liquid_viscosity']*1000
    # '''ESP'''
    # ESP(ESP_input,inputValues, QBEM, well_name,Q_water,H_water,Q_oil,H_oil,VisL_target)
    # '''Pressure profile'''
    # ql, P = Nodal_solve(inputValues, df_well_profile, GLR, WC, QBEM, ESP_input)
    # print('Nodal: ', ql,' m3d', P,' Mpa')
    # Qg_res = ql*GLR/24/3600
    # Qo_res = ql*(1-WC/100)/24/3600
    # Qw_res = ql*WC/100/24/3600
    # P, df = P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input)
    # P_profile_plot (df_well_profile, df, well_name, inputValues['P_in'], inputValues['P_out'])
    # '''Nodel'''
    # QL = np.array([ql/32,ql/16,ql/8,ql/4,ql/2,ql,ql*1.2,ql*1.5])    # m3/d
    # QL = QL/24/3600
    # Pwf = np.arange(0, int(inputValues['Reservoir_P']))
    # Nodal (inputValues, df_well_profile, QL, GLR, Pwf, WC, QBEM, ESP_input, Q_oil, df.List_P.iloc[-1], well_name)
    # print('complete')
    # plt.show()

    '''201B-7'''
    # well_name = '201B-7'
    # Q_water = 100
    # H_water=1700
    # Q_oil = 16.7  # field mix QL
    # H_oil = (14.21-2.05)*Mpa_to_m    # pump head at field mix QL
    # P_in=1.24
    # P_out=14.28
    # df_well_profile = read_input(well_name+'.xlsx')
    # inputValues = {"Surface_line_pressure": 2.4*1e6, "Pipe_diameter":0.088,      
    #                 "Roughness":0,                   "Liquid_viscosity":1.07955,
    #                 "Liquid_relative_density":0.9657,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
    #                     "Water_cut": 20, 
    #                 "Reservoir_C":5.23E-18,     "Reservoir_n":1, 
    #                 "Reservoir_P":1.75e7,               "GLR":6/0.755,                   "Geothermal_gradient":0.03, 
    #                 "Surface_T":288, "Pump_intake_T": 380, "P_in": 2.05, "P_out":14.21, 
    #                 "ESP_Depth": 1901.78, "ESP_SN":296, "ESP_N":2940, "ESP_length": 10}

    # '''pressure profile'''
    # WC=inputValues['Water_cut']
    # GLR = inputValues['GLR']
    # ESP_input=ESP_default['Flex31'].copy()
    # QBEM=7000
    # ESP_input['R2']*=0.915
    # ESP_input['SN']=inputValues['ESP_SN']
    # ESP_input['N']=inputValues['ESP_N']
    # ESP_Depth = inputValues['ESP_Depth']
    # VisL_target = inputValues['Liquid_viscosity']*1000
    # '''ESP'''
    # ESP(ESP_input,inputValues, QBEM, well_name,Q_water,H_water,Q_oil,H_oil,VisL_target)
    # '''Pressure profile'''
    # ql, P = Nodal_solve(inputValues, df_well_profile, GLR, WC, QBEM, ESP_input)
    # print('Nodal: ', ql,' m3d', P,' Mpa')
    # Qg_res = ql*GLR/24/3600
    # Qo_res = ql*(1-WC/100)/24/3600
    # Qw_res = ql*WC/100/24/3600
    # P, df = P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input)
    # P_profile_plot (df_well_profile, df, well_name, inputValues['P_in'], inputValues['P_out'])
    # '''Nodel'''
    # QL = np.array([ql/32,ql/16,ql/8,ql/4,ql/2,ql,ql*1.2,ql*1.5])    # m3/d
    # QL = QL/24/3600
    # Pwf = np.arange(0, int(inputValues['Reservoir_P']))
    # Nodal (inputValues, df_well_profile, QL, GLR, Pwf, WC, QBEM, ESP_input, Q_oil, df.List_P.iloc[-1], well_name)
    # print('complete')
    # plt.show()


    '''201A-7'''
    well_name = '201A-7'
    Q_water = 150.53
    H_water=1606.84
    Q_oil = 48  # field mix QL
    H_oil = (13.53-3.85)*Mpa_to_m    # pump head at field mix QL
    df_well_profile = read_input(well_name+'.xlsx')
    inputValues = {"Surface_line_pressure": 3.28*1e6, "Pipe_diameter":0.088,      
                    "Roughness":0,                   "Liquid_viscosity":0.5006,
                    "Liquid_relative_density":0.9456,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
                        "Water_cut": 18, 
                    "Reservoir_C":1.83E-17,     "Reservoir_n":1, 
                    "Reservoir_P":1.87e7,               "GLR":10,                   "Geothermal_gradient":0.03, 
                    "Surface_T":288, "Pump_intake_T": 380, "P_in": 3.85, "P_out":13.53, 
                    "ESP_Depth": 1509.03, "ESP_SN":286, "ESP_N":2950, "ESP_length": 10}
    '''pressure profile'''
    WC=inputValues['Water_cut']
    GLR = inputValues['GLR']
    ESP_input=ESP_default['Flex31'].copy()
    QBEM=6000
    ESP_input['R2']*=0.92
    ESP_input['SN']=inputValues['ESP_SN']
    ESP_input['N']=inputValues['ESP_N']
    ESP_Depth = inputValues['ESP_Depth']
    VisL_target = inputValues['Liquid_viscosity']*1000
    '''ESP'''
    ESP(ESP_input,inputValues, QBEM, well_name,Q_water,H_water,Q_oil,H_oil,VisL_target)
    '''Pressure profile'''
    ql, P = Nodal_solve(inputValues, df_well_profile, GLR, WC, QBEM, ESP_input)
    print('Nodal: ', ql,' m3d', P,' Mpa')
    Qg_res = ql*GLR/24/3600
    Qo_res = ql*(1-WC/100)/24/3600
    Qw_res = ql*WC/100/24/3600
    P, df = P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input)
    P_profile_plot (df_well_profile, df, well_name, inputValues['P_in'], inputValues['P_out'])
    '''Nodel'''
    QL = np.array([ql/32,ql/16,ql/8,ql/4,ql/2,ql,ql*1.2,ql*1.5])    # m3/d
    QL = QL/24/3600
    Pwf = np.arange(0, int(inputValues['Reservoir_P']))
    Nodal (inputValues, df_well_profile, QL, GLR, Pwf, WC, QBEM, ESP_input, Q_oil, df.List_P.iloc[-1], well_name)
    print('complete')
    plt.show()

    
