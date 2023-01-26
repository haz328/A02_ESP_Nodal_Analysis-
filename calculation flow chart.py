
import sys
import os
path_abs = os.path.dirname(os.path.abspath(__file__))   # current path
path = os.path.join(path_abs, 'C:\\Users\\haz328\\Desktop\\Github') # join path
sys.path.append(path)  # add to current py
from Common import *
from Utility import *
from TUALP_Models import ESP_TUALP

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'     #disable dataframe warning msg
# from ESP_simple_all_in_one import *
from Wellbore import *

np_VISL_SSU = np.array([50, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 5000])
np_Capacity = np.array([1, 0.98, 0.97, 0.947, 0.924, 0.886, 0.847, 0.819, 0.792, 0.766, 0.745, 0.727, 0.708, 0.659, 0.621, 0.59, 0.562, 0.518, 0.479])
np_Head = np.array([1, 0.99, 0.985, 0.97, 0.958, 0.933, 0.909, 0.897, 0.883, 0.868, 0.858, 0.846, 0.833, 0.799, 0.771, 0.75, 0.733, 0.702, 0.677])
np_Eff = np.array([0.945, 0.87, 0.825, 0.736, 0.674, 0.566, 0.497, 0.462, 0.434, 0.41, 0.39, 0.368, 0.349, 0.307, 0.272, 0.245, 0.218, 0.278, 0.149])
np_BHP = np.array([1.058, 1.115, 1.158, 1.248, 1.341, 1.46, 1.549, 1.59, 1.611, 1.622, 1.639, 1.671, 1.69, 1.715, 1.76, 1.806, 1.89, 2.043, 2.176])
bpd_to_m3d = 0.159
ft_to_m = 0.3048
hp_to_kW = 0.7457
m_to_pa = 9806.65
# cp = cst * sp (specific gravity)
# SSU = 4.6128 * cst + 6.6205

class ESP_curve(object):
    def __init__(self, pump_name='1', well_name='1'):
        self.pump_name=pump_name
        self.well_name=well_name

    def input_pump(self, source='SQL', SQLname = 'SQL_database/Default_pump.db', op_string = "SELECT * FROM Sheng_li_pump", Total_SN='default', Motor_coeff = 0.7):
        if source=='SQL':    
            conn = sqlite3.connect(SQLname)
            pump_curve = pd.read_sql_query(op_string, conn)
            conn.commit()   #apply changes to the database
            conn.close()

            pump_curve['Total_kW']=pump_curve['BHP_kW']*pump_curve['Efficiency']/pump_curve['Total_eff']

            self.pump_curve_water_3600RPM=pump_curve.copy()
            if Total_SN=='default': Total_SN=pump_curve.SN
            # calibrate to 3600 RPM
            # calibrate by stage number (change to single stage pump curve)
            self.pump_curve_water_3600RPM.Q_m3d *= (3600/pump_curve.RPM.mean())
            self.pump_curve_water_3600RPM.H_m *= (3600/pump_curve.RPM.mean())**2/Total_SN
            self.pump_curve_water_3600RPM.BHP_kW *= (3600/pump_curve.RPM.mean())**3/Total_SN
            try:
                # if total efficiency is provided
                self.pump_curve_water_3600RPM.Total_kW = self.pump_curve_water_3600RPM.BHP_kW*self.pump_curve_water_3600RPM.Efficiency/self.pump_curve_water_3600RPM.Total_eff
            except:
                # if total efficiency not provided
                # assume motor efficiency is 0.7 (as shown in default input)
                self.pump_curve_water_3600RPM.Total_eff *= self.pump_curve_water_3600RPM.Efficiency*Motor_coeff
                self.pump_curve_water_3600RPM.Total_kW = self.pump_curve_water_3600RPM.BHP_kW/Motor_coeff
            # change RPM to 3600
            self.pump_curve_water_3600RPM.RPM = 3600
            
            # coefficients
            self.Q_H_coeff = np.polynomial.Polynomial.fit(self.pump_curve_water_3600RPM['Q_m3d'], self.pump_curve_water_3600RPM['H_m'], 5)
            self.Q_BHP_coeff = np.polynomial.Polynomial.fit(self.pump_curve_water_3600RPM['Q_m3d'], self.pump_curve_water_3600RPM['BHP_kW'], 5)
            self.Q_EFF_coeff = np.polynomial.Polynomial.fit(self.pump_curve_water_3600RPM['Q_m3d'], self.pump_curve_water_3600RPM['Efficiency'], 5)
        elif source =='csv':
            pass

    def choose_pump_from_database(self, QBEP_range=0.12, QD_range=0.1, Target_D_mm=False, Target_RPM=False, 
                    Target_Q=False, Target_H=False, Target_BHP=False, Target_EFF=False, Total_SN='1', SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump'):
        # database from literature, all pump curve calibrated to: 
        # pump speed: 3600
        # pump stage: 1
        self.SQLname = SQLname
        self.Table_name = Table_name

        self.Q_BEP_range = QBEP_range
        self.QD_range = QD_range
        self.Target_RPM = Target_RPM

        
        self.Target_BHP = Target_BHP
        self.Target_Q = Target_Q
        self.Target_H = Target_H
        self.Target_EFF = Target_EFF
        self.Total_SN = Total_SN
        self.Target_D_mm = Target_D_mm

        '''select pump'''
        conn, c = connect_db(SQLname)
        c = conn.cursor()
        # know pump info
        
        if pump_name != True:
            # command_str = "SELECT * FROM " + Table_name + " Where Pump = "+ str(pump_name)
            self.calibrate_water_curve(self.pump_name)
        else:
            # don't know pump info, select from database
            command_str = "SELECT * FROM " + self.Table_name
            df_pump = pd.read_sql_query (command_str,   conn)    # only data for ND20 is single phase

            df_select = df_pump.loc[(df_pump['Q_BEP']>self.Target_Q*(1-QBEP_range)) & (df_pump['Q_BEP']<self.Target_Q*(1+QBEP_range))
                                & (df_pump['BHP_BEP']>self.Target_BHP*(1-QD_range*3)) & (df_pump['BHP_BEP']<self.Target_BHP*(1+QD_range*3))
                                & (df_pump['D_mm']>self.Target_D_mm*(1-QD_range)) & (df_pump['D_mm']<self.Target_D_mm*(1+QD_range))]

            if df_select['Pump'].unique()>1:
                Matched_pump = []
                Ave_error = []
                Q_error = []
                H_error = []
                BHP_error = []
                EFF_error = []
                for pump in df_select['Pump'].unique():
                    df_1 = df_select[df_select['Pump']==pump]
                    Matched_pump.append(pump)
                    Q_error_1 = (abs(df_1['Q_BEP'].mean()-self.Target_Q)/self.Target_Q)
                    H_error_1 = (abs(df_1['H_BEP'].mean()-self.Target_H)/self.Target_H)
                    BHP_error_1 = (abs(df_1['BHP_BEP'].mean()-self.Target_BHP)/self.Target_BHP)
                    EFF_error_1 = (abs(df_1['EFF_BEP'].mean()/100-self.Target_EFF)/self.Target_EFF)
                    Q_error.append(abs(df_1['Q_BEP'].mean()-self.Target_Q)/self.Target_Q)
                    H_error.append(abs(df_1['H_BEP'].mean()-self.Target_H)/self.Target_H)
                    BHP_error.append(abs(df_1['BHP_BEP'].mean()-self.Target_BHP)/self.Target_BHP)
                    EFF_error.append(abs(df_1['EFF_BEP'].mean()-self.Target_EFF)/self.Target_EFF)
                    Ave_error.append((Q_error_1+H_error_1+BHP_error_1)/4)
                df_error = pd.DataFrame({'Pump':Matched_pump, 'Q_error':Q_error, 'H_error':H_error, 
                                'BHP_error':BHP_error, 'EFF_error':EFF_error, 'Ave_error':Ave_error})
                df_error.sort_values(by=['Ave_error'], inplace=True)
                self.df_error = df_error
                self.pump_name=df_error.iloc[0]['Pump']
                pump_curve = df_select[df_select['Pump']==self.pump_name]
                self.calibrate_water_curve(pump_curve)
                self.field_pump_curve = self.pump_curve_water_3600RPM

    def calibrate_water_curve(self, pump_curve, Motor_coeff=0.7):
        
        '''initialize'''
        self.pump_curve_water_3600RPM = pump_curve
        self.Q_H_coeff = np.polynomial.Polynomial.fit(self.pump_curve_water_3600RPM['Q_m3d'], self.pump_curve_water_3600RPM['H_m'], 5)
        self.Q_BHP_coeff = np.polynomial.Polynomial.fit(self.pump_curve_water_3600RPM['Q_m3d'], self.pump_curve_water_3600RPM['BHP_kW'], 5)
        self.Q_EFF_coeff = np.polynomial.Polynomial.fit(self.pump_curve_water_3600RPM['Q_m3d'], self.pump_curve_water_3600RPM['Efficiency'], 5)
        '''calibrate'''

        fit_Q = self.Target_Q
        fit_H = self.Q_H_coeff(fit_Q)
        fit_BHP = self.Q_BHP_coeff(fit_Q)
        fit_EFF = self.Q_EFF_coeff(fit_Q)
        self.Q_H_coeff += ((self.Target_H-fit_H),0,0,0,0,0)
        self.Q_BHP_coeff += ((self.Target_BHP-fit_BHP),0,0,0,0,0)
        self.Q_EFF_coeff += ((self.Target_EFF-fit_EFF),0,0,0,0,0)
        # 2323-1-12: change to 60HZ(3600RPM), consider pump stage in the new database file
        self.pump_curve_water_3600RPM.Q_m3d *= (3600/pump_curve.RPM.mean())
        self.pump_curve_water_3600RPM.H_m *= (3600/pump_curve.RPM.mean())**2
        self.pump_curve_water_3600RPM.BHP_kW *= (3600/pump_curve.RPM.mean())**3
        try:
            # if total efficiency is provided
            self.pump_curve_water_3600RPM.Total_kW = self.pump_curve_water_3600RPM.BHP_kW*self.pump_curve_water_3600RPM.Efficiency/self.pump_curve_water_3600RPM.Total_eff
        except:
            # if total efficiency not provided
            # assume motor efficiency is 0.7 (as shown in default input)
            self.pump_curve_water_3600RPM.Total_eff *= self.pump_curve_water_3600RPM.Efficiency*Motor_coeff
            self.pump_curve_water_3600RPM.Total_kW = self.pump_curve_water_3600RPM.BHP_kW/Motor_coeff
            
        self.pump_curve_water_3600RPM.RPM = 3600

        self.pump_curve_water_3600RPM['H_m']*=(self.Target_H/fit_H)    
        self.pump_curve_water_3600RPM['BHP_kW']*=(self.Target_BHP/fit_BHP)
        self.pump_curve_water_3600RPM['Efficiency']*=(self.Target_EFF/fit_EFF*100) #2023-1-12 change to % since new pump curve is %

        # self.pump_curve_water_3600RPM['H_m']+=(self.Target_H-fit_H)
        # self.pump_curve_water_3600RPM['BHP_kW']+=(self.Target_BHP-fit_BHP)
        # self.pump_curve_water_3600RPM['Efficiency']+=(self.Target_EFF-fit_EFF)

        return self.pump_curve_water_3600RPM

    @staticmethod
    def solve_y_from_ployfit(poly_coeffs, y):
        pc = poly_coeffs.copy()
        roots = (poly_coeffs - y).roots()
        for i in range(len(roots)):
            if np.isreal(roots[i]) and roots[i]>0:
                    return np.real(roots[i])
        return 0

    def viscosity_calibrate(self, SSU, target_Hz,Motor_coeff=0.7):
        ''' pump curve at giving frequence and viscosity '''
        Q_coef,H_coef,BHP_coef,EFF_coef = self.pump_coef(SSU)
        pump_curve = self.pump_curve_water_3600RPM.copy()
        pump_curve['Q_m3d']=pump_curve['Q_m3d']*(target_Hz/60*Q_coef)
        pump_curve['H_m']=pump_curve['H_m']*(H_coef*(target_Hz/60)**2)
        pump_curve['BHP_kW']=pump_curve['BHP_kW']*BHP_coef*(target_Hz/60)**3
        pump_curve['Efficiency']=pump_curve['Efficiency']*EFF_coef
        try:
            pump_curve['Total_eff']=pump_curve['Total_eff']*EFF_coef
            pump_curve['Total_kW']= pump_curve.BHP_kW*pump_curve.Efficiency/pump_curve.Total_eff
        except:
            pump_curve['Total_eff']=pump_curve['Efficiency']*Motor_coeff
            pump_curve.Total_kW = pump_curve.BHP_kW/Motor_coeff
        self.field_pump_curve = pump_curve
        return pump_curve
        
    def ESP_Model_calibrate(self, Q_model, H_model):
        
        poly_model = np.polynomial.Polynomial.fit(Q_model, H_model, 5)
        Qmax = self.solve_y_from_ployfit(poly_model,0)
        self.field_pump_curve['Q_m3d'] *= Qmax/max(self.field_pump_curve['Q_m3d'])
        self.field_pump_curve['H_m'] = poly_model(self.field_pump_curve['Q_m3d'])

    def solve_performance_from_BHP(self, BHP, pump_curve='field'):
        '''solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP)'''
        try:
            if pump_curve == 'field':
                pump_curve = self.field_pump_curve
            elif pump_curve == 'water':
                pump_curve = self.pump_curve_water_3600RPM
        except:
            pump_curve=pump_curve
        Q_H_coeff = np.polynomial.Polynomial.fit(pump_curve['Q_m3d'], pump_curve['H_m'], 5)
        Q_BHP_coeff = np.polynomial.Polynomial.fit(pump_curve['Q_m3d'], pump_curve['BHP_kW'], 5)
        try:
            Q_TBHP_coeff = np.polynomial.Polynomial.fit(pump_curve['Q_m3d'], pump_curve['Total_kW'], 5)
        except:
            pass
        Q_EFF_coeff = np.polynomial.Polynomial.fit(pump_curve['Q_m3d'], pump_curve['Efficiency'], 5)
        Q = self.solve_y_from_ployfit(Q_TBHP_coeff, BHP)
        # Q = self.solve_y_from_ployfit(Q_BHP_coeff, BHP)
        if Q > self.solve_y_from_ployfit(Q_H_coeff, 0): Q=self.solve_y_from_ployfit(Q_H_coeff, 0)*0.95
        H = Q_H_coeff(Q)
        EFF = Q_EFF_coeff(Q)
        return Q, H, EFF

    def train_ESP_GEO(self):
        global ESP_default, base_pump

        Qmax = self.solve_y_from_ployfit(self.Q_H_coeff, 0)
        # Q = np.linspace(10, Qmax*0.8, 30)
        # Q = np.linspace(10, Qmax*0.8, 30)
        try:
            Q = np.array([self.Target_Q, Qmax*0.05, Qmax*0.1, Qmax*0.2, Qmax*0.3, Qmax*0.4, Qmax*0.5, Qmax*0.55, Qmax*0.6, Qmax*0.65, Qmax*0.7, Qmax*0.75, Qmax*0.8, Qmax*0.9,
                    Qmax*0.95,Qmax, Qmax*1.05, Qmax*1.1, Qmax*1.15 ])
        except:
            Q = np.array([Qmax*0.05, Qmax*0.1, Qmax*0.2, Qmax*0.3, Qmax*0.4, Qmax*0.5, Qmax*0.55, Qmax*0.6, Qmax*0.65, Qmax*0.7, Qmax*0.75, Qmax*0.8, Qmax*0.9,
                    Qmax*0.95,Qmax, Qmax*1.05, Qmax*1.1, Qmax*1.15 ])   
        Input = pd.DataFrame({'QL':Q/bpd_to_m3d, 
                            'HP':self.Q_H_coeff(Q)/(psi_to_ft*ft_to_m), 
                            'VISO':1/1000, 'DENO':1000,
                                'VISG':0.000018, 'WC':100, 'GLR':0, 
                                'P':100*psi_to_pa, 'TT':288, 'STOW':0.035})
        Input.sort_values(by=['QL'], inplace=True)
        Target_HP = self.Q_H_coeff(Q)/(psi_to_ft*ft_to_m)
    
        noise_var=0.1
        a_par=5e-7
        max_iter=500
        report=10
        # Train_parameter = np.ones(12)
        # min_vals=np.ones(12)*0.5
        # min_vals[0]=0.3
        # min_vals[3]=0.95
        # max_vals = np.ones(12)*2
        # max_vals[0]=100
        # max_vals[3]=1.5

        # ori
        Train_parameter = np.array( [1.21972458, 1.15445868, 0.95000022, 1.01323074, 1.02883895, 0.94690516,
                            1.07350576, 1.1550835 , 0.79165659, 0.71448075, 0.94944346, 0.98978539,
                            1.23058711, 1.13269302, 0.64402034, 0.75975723, 1.09095647, 0.97642078,
                            1.02850508, 0.91159665, 0.9805586 , 0.89685751] )
        min_vals=np.ones(22)*0.5
        min_vals[0]=0.3
        min_vals[2]=0.95
        max_vals = np.ones(22)*2
        max_vals[0]=100
        max_vals[2]=1.5


        base_pump = 'Flex31'
        ESP, QBEM = ESP_geometry(base_pump = base_pump, Train_parameter=Train_parameter)
        HP_0 = ESP_head (QBEM, ESP, np.array(60), 0.001, 1000, 0.001, 1000, 0.000018, 100, 0, 100*psi_to_pa, 288, 0.035, None)
        ESP_default[base_pump]['R2'] = ESP_default[base_pump]['R2']*(self.Q_H_coeff(0)/(psi_to_ft*ft_to_m)/HP_0)**0.5*1.05
        
        _, Train_parameter1, _, _ = SPSA_match(Train_parameter, Input, Target_HP, noise_var, a_par, min_vals, max_vals, max_iter, report)
        ESP, QBEM = ESP_geometry(base_pump = base_pump, Train_parameter=Train_parameter1)

        # save water curve and ESP geometry
        fig, ax = plt.subplots()
        ax.scatter(Input['QL']*bpd_to_m3d, Input['HP']*psi_to_ft*ft_to_m, c='r', label='数据库曲线')
        try:  # if provided
            ax.scatter(self.Target_Q, self.Target_H, marker='*', c='g', label='出厂测试')
        except:
            pass
        HP = ESP_head (QBEM, ESP, Input['QL'], 0.001, 1000, 0.001, 1000, 0.000018, 100, 0, 100*psi_to_pa, 288, 0.035, None)
        ax.plot(Input['QL']*bpd_to_m3d, HP*psi_to_ft*ft_to_m, c='b', label='拟合曲线')

        ax.set_ylim(0)
        ax.set_xlim(0)
        # ax.set_xlabel('排量(m3/d)')
        # ax.set_ylabel('扬程(m3/d)')
        # ax.set_title('井'+str(self.well_name)+'电潜泵水性能曲线')
        # ax.legend()
        fig.savefig(str(self.well_name)+' water curve.jpg')

        ESP_GEO = pd.DataFrame.from_dict([ESP])
        ESP_GEO.insert(0, 'QBEM', QBEM)
        ESP_GEO.to_excel(str(self.well_name)+' ESP GEO.xlsx')

        return ESP, QBEM

    @staticmethod
    def pump_coef (SSU):
        if SSU <= 400:
            Q_coef = (-1.5134e-9*SSU**3+1.3051e-6*SSU**2-7.4607e-4*SSU+1.0334)
            H_coef = 1.0088*np.exp(-2e-4*SSU)
            BHP_coef = (-1e-6*SSU**2+0.0018*SSU+1)
        else:
            Q_coef = -0.143*np.log(SSU)+1.705
            H_coef = -0.093*np.log(SSU)+1.4766
            BHP_coef = 1.5521*np.exp(7e-5*SSU)
        BHP_coef = (BHP_coef-1)/3+1
        EFF_coef = Q_coef*H_coef/BHP_coef

        return Q_coef,H_coef,BHP_coef,EFF_coef

    def coef_plot(self):
        Q_coef,H_coef,BHP_coef,EFF_coef = np.ones(19), np.ones(19), np.ones(19), np.ones(19)
        for i in range(np_VISL_SSU.shape[0]):
            Q_coef[i],H_coef[i],BHP_coef[i],EFF_coef[i] = self.pump_coef(np_VISL_SSU[i])

        fig, [(ax, bx), (cx, dx)]= plt.subplots(ncols=2, nrows=2, dpi=128)

        ax.scatter(np_VISL_SSU, np_Capacity)
        bx.scatter(np_VISL_SSU, np_Head)
        cx.scatter(np_VISL_SSU, np_Eff)
        dx.scatter(np_VISL_SSU, np_BHP)
    
        ax.plot(np_VISL_SSU, Q_coef, 'r')
        bx.plot(np_VISL_SSU, H_coef, 'r')
        cx.plot(np_VISL_SSU, EFF_coef, 'r')
        dx.plot(np_VISL_SSU, BHP_coef, 'r')

        ax.set_title('Capacity')
        bx.set_title('Head')
        cx.set_title('Efficiency')
        dx.set_title('BHP')

        fig.tight_layout()

    def coef_compare(self):
        df_pump = pd.read_excel('Data/BHI_P100.xlsx', sheet_name='BHI_P100')
        df_3600 = df_pump[df_pump['RPM']==3600]
        df_3600['Viscosity_cP'] = df_3600['Viscosity_cP'].round()
        df_3600['Qw_bpd'] = df_3600['Qw_bpd']*bpd_to_m3d
        df_3600['Head_ft'] = df_3600['Head_ft']*ft_to_m
        df_3600['BHP_hp'] = df_3600['BHP_hp']*hp_to_kW
        print(df_3600)
        df_water = df_3600[df_3600['Viscosity_cP']==0]
        for viscosity in df_3600['Viscosity_cP'].unique():
            df_u = df_3600[df_3600['Viscosity_cP']==viscosity]
            # plt.scatter(df_u['Qw_bpd'], df_u['Head_ft'], label=viscosity)
            plt.scatter(df_u['Qw_bpd'], df_u['BHP_hp'], label=viscosity)
            # plt.scatter(df_u['Qw_bpd'], df_u['Qw_bpd']*df_u['Head_ft']/df_u['BHP_hp']/8333.33, label=viscosity)
            # SSU = 4.6128 * viscosity / 0.7 + 6.6205
            SSU = viscosity / 0.7
            Q_coef,H_coef,BHP_coef,EFF_coef = self.pump_coef(SSU)
            # plt.plot(df_water['Qw_bpd']*Q_coef, df_water['Head_ft']*H_coef)
            plt.plot(df_water['Qw_bpd']*Q_coef, df_water['BHP_hp']*BHP_coef)
            # plt.plot(df_u['Qw_bpd']*Q_coef, df_u['Qw_bpd']*df_u['Head_ft']/df_u['BHP_hp']/8333.33, label=viscosity)
        plt.legend()
        plt.show()

def Generate_pump_curve_SQL():

    SQLname = '/Users/haiwenzhu/Desktop/work/020 SQL/ESP.db'
    ''' connect SQL '''
    conn, c = connect_db(SQLname)
    c = conn.cursor()
    # ''' download data '''
    df_pump = pd.read_sql_query("SELECT Manufacturer, Pump, RPM, Flow_bpd, Head_ft, BHP_hp, Efficiency, N_stages, Ns, D_in FROM Catalog_Prosper_whole "
                                , conn)
    df_pump_SI = pd.DataFrame({'Pump':df_pump['Manufacturer']+' '+df_pump['Pump']+' '+(round(df_pump['D_in']*25.4)).astype(str)+'mm', 'RPM':df_pump['RPM'], 'Q_m3d': df_pump['Flow_bpd']*bpd_to_m3d, 
                              'H_m': df_pump['Head_ft']/df_pump['N_stages']*ft_to_m, 'BHP_kW':df_pump['BHP_hp']*hp_to_kW, 'Efficiency':df_pump['Efficiency'],
                              'Ns': df_pump['Ns'], 'D_mm': df_pump['D_in']*25.4, 
                              'Q_BEP':0, 'H_BEP':0, 'BHP_BEP':0, 'EFF_BEP':0})
    
    i = 0
    for pump in df_pump_SI['Pump'].unique():
        df_pump_curve = df_pump_SI[df_pump_SI['Pump']==pump]
        df_EFF_max = df_pump_curve['Efficiency'].max()
        df_Q_BEP = df_pump_curve[df_pump_curve['Efficiency']==df_EFF_max]['Q_m3d'].values[0]
        df_H_BEP = df_pump_curve[df_pump_curve['Efficiency']==df_EFF_max]['H_m'].values[0]
        df_BHP_BEP = df_pump_curve[df_pump_curve['Efficiency']==df_EFF_max]['BHP_kW'].values[0]
        # if pump == 'R14' or i > 15898:
        #     print(df_pump_curve)
        #     kkk=1
        #     pass
        for j in range(df_pump_curve.shape[0]):
            df_pump_SI.loc[i, 'Q_BEP']=df_Q_BEP
            df_pump_SI.loc[i, 'H_BEP']=df_H_BEP
            df_pump_SI.loc[i, 'BHP_BEP']=df_BHP_BEP
            df_pump_SI.loc[i, 'EFF_BEP']=df_EFF_max
            i+=1
        print(i)

    df_pump_SI.to_excel("Catalog_SI.xlsx")
    df_pump_SI.to_sql("Catalog_SI", conn, if_exists="replace")

    c.execute('''CDROP INDEX Catalog_SI''')
    # drop indices in SQLit
    # indices help speed up SELECT and WHERE, but slow down UPDATE and INSERT

    
    ''' close database '''
    conn.commit()   #apply changes to the database
    conn.close()
 
def test_calculation():
    QBEP_range = 0.12
    QD_range = 0.1
    # for i in range (2,13):
    # # for i in [1,3,4,5,6,7,8,9,10,11,12,13]:

    # # for i in [1,5,7,8,9]:   # low speed
    # # for i in [4,5,6,9]:   # emulsion
    # # for i in [2,3]:   # high gas
    # for i in [10,11,12]: # general
    for i in [1]:
        # 1, 7, 10 wrong data, pump damage?
        # 2, 5, 9 high error
        # rest ok
        '''pump infomation'''
        df_pump_info = pd.read_excel('Data/Pump data/'+str(i)+'.xlsx', index_col=None)
        # print(df_pump_info)
        # print(df_pump_info['粘度'])
        '''well infomation'''
        df_well = pd.read_excel('Data/Production data/'+str(i)+'.xls',skiprows=1, index_col=None, na_values=['NA'])
        df_well.drop(0, axis=0,inplace=True)
        df_well.drop(df_well.tail(1).index, axis=0,inplace=True)
        df_well['电流'] = df_well['电流'].astype(float)
        df_well['电压'] = df_well['电压'].astype(float)
        # print(df_well)
        
        ''' select pump and generate pump curve at 3600 RPM'''
        ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
                        df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
                        df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
                        QBEP_range=QBEP_range, QD_range=QD_range,Total_SN=df_pump_info['泵总级数'][0],
                        SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)

        # print(ESP_well_1.pump_name)
        '''inputs'''
        SQLname = 'SQL_database/Default_pump.db'


        VISL = df_pump_info['粘度'][0]
        WC = df_pump_info['含水'][0]
        DENL = 0.95 # not sure
        # VISL = SinglePhaseModel.emulsion(ESP_well_1.ESP_GEO['VOI'], ESP_well_1.ESP_GEO['R2'], VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
        #                 df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
        VISL = SinglePhaseModel.emulsion(6e-6, 0.04, VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
                        df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
        VISL = VISL * 1000
        # print(VISL)
        GOR = df_pump_info['油汽比'][0]
        # assume Pave_pump = 500 psi
        Pave_pump = 500
        GOR = GOR*14.7/Pave_pump
        GLR = GOR*(100-WC)/100
        HG = GLR/(GLR+1)
        HL = 1 - HG
        VISM = VISL*HL+0.000018*HG
        DENM = (DENL*1000*HL+1.224*Pave_pump/14.7*HG)/1000
        # VISM = VISL
        # DENM = DENL
        # field_BHP = df_well['电压'].mean()*df_well['电流'].mean()/1000/df_pump_info['泵总级数'][0] * (df_pump_info['轴功率'][0]/df_pump_info['额定功率'][0])
        field_BHP = df_well['电压'].mean()*df_well['电流'].mean()/1000/df_pump_info['泵总级数'][0]
        # print('well '+str(i)+' field BHP: ', field_BHP)

        ''' pump curve calibration based on (speed and SSU (viscosity and density (air can be included)) '''
        # Assume/calculate Q, P

        # calculate mixture density (oil, water, air), initial assumption homongenous model

        # calculate mixture viscosity (oil, water, air), initial assumption homongenous model

        # calculate SSU
        SSU = VISM / DENM
        # pump_curve calibration
        _ = ESP_well_1.viscosity_calibrate(SSU, df_pump_info['生产频率'][0]) 
        ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''
        Q, H, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)
        print('泵:',ESP_well_1.pump_name,
                '日液:',df_pump_info['日液'][0], 'Q:',round(Q*HL), 'Q误差:',round((Q-df_pump_info['日液'][0])/df_pump_info['日液'][0],2), 
                'H:',round(H,2), 'EFF:',round(EFF,2), 
                '现场功率:',round(field_BHP,2), 
                'BHP:',round(df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],2),
                'VISL:',round(VISL))
        ''' velocity, pressure, HL ... profile based on Q and H '''

        
        '''pump curve plot '''
        final_pump = 'none'
        final_error = 10
        for pump in ESP_well_1.df_error['Pump'].unique():
            ESP_well_1.calibrate_water_curve(pump)
            _ = ESP_well_1.viscosity_calibrate(SSU, df_pump_info['生产频率'][0]) 
            ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''
            Q, H, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)
            
            print('泵%30s'%ESP_well_1.pump_name,
                '日液:',df_pump_info['日液'][0], 'Q:%6.2f'%(Q*HL), 'Q误差:',round((Q-df_pump_info['日液'][0])/df_pump_info['日液'][0],2), 
                'H:',round(H,2), 'EFF:',round(EFF,2), 
                '现场功率:',round(field_BHP,2), 
                'BHP:',round(df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],2),
                'VISL:',round(VISL))
            if abs(round((Q-df_pump_info['日液'][0])/df_pump_info['日液'][0],2)) < abs(final_error):
                final_pump = pump
                final_error = round((Q-df_pump_info['日液'][0])/df_pump_info['日液'][0],2)


        '''pump curve plot '''
        
        print(final_pump,' error:%4.2f'%final_error)
        pump = final_pump
        ESP_well_1.calibrate_water_curve(pump)
        _ = ESP_well_1.viscosity_calibrate(SSU, df_pump_info['生产频率'][0]) 
        ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''
        Q, H, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)
        fig, ax = plt.subplots()
        ax_twin = ax.twinx()
        Q = np.linspace(1, round(df_pump_info['试验排量']/100*2)*100, 100)
        ax.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_H_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**2, 'b', label='水扬程')
        ax_twin.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_BHP_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**3, 'r', label='水功率')
        # ax_twin.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_EFF_coeff(Q), 'g')
        ax.scatter(df_pump_info['试验排量'][0], df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],marker='*', c='b')
        ax_twin.scatter(df_pump_info['试验排量'][0], df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],marker='*', c='r')
        # ax_twin.scatter(df_pump_info['试验排量'][0], df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0],marker='*', c='g')

        '''viscosity pump curve'''
        ax.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['H_m'], c='b', label='现场扬程')
        ax_twin.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['BHP_kW'], c='r', label='现场功率')
        # ax_twin.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['Efficiency'], c='g')

        ax.set_ylim(0)
        ax.set_xlim(0)
        ax_twin.set_ylim(0)
        ax_twin.set_xlim(0)
        ax.legend()
        ax_twin.legend()
        plt.show()

def test_calculation_2():
    QBEP_range = 0.12
    QD_range = 0.1
    i=12 
    # well number
    # 1, 7, 10 wrong data, pump damage?
    # 2, 5, 9 high error
    # rest ok 6, 12, 4, 3, 11, 8
    for i in [9]:
        j = 0 # production data
        '''pump infomation'''
        df_pump_info = pd.read_excel('Data/Pump data/'+str(i)+'.xlsx', index_col=None)

        '''well infomation'''
        df_well = pd.read_excel('Data/Production data/'+str(i)+'.xls',skiprows=1, index_col=None, na_values=['NA'])
        df_well.drop(0, axis=0,inplace=True)
        df_well.drop(df_well.tail(1).index, axis=0,inplace=True)
        df_well['电流'] = df_well['电流'].astype(float)
        df_well['电压'] = df_well['电压'].astype(float)
        df_well.replace('', np.nan, inplace=True)
        df_well.dropna(subset=['电流','电压','含水'], inplace=True)

        ''' select pump and generate pump curve at 3600 RPM'''
        ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
                        df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
                        df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
                        QBEP_range=QBEP_range, QD_range=QD_range,Total_SN=df_pump_info['泵总级数'][0],
                        SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)
        # print(ESP_well_1.df_error)

        ''' pump curve calibration based on (speed and SSU (viscosity and density (air can be included)) '''
        # Assume/calculate Q, P
        # calculate mixture density (oil, water, air), initial assumption homongenous model
        # calculate mixture viscosity (oil, water, air), initial assumption homongenous model
        # calculate SSU

        ''' option 1: by average inputs'''
        SQLname = 'SQL_database/Default_pump.db'
        VISL = df_pump_info['粘度'][0]
        WC = df_pump_info['含水'][0]
        DENL = 0.95 # not sure
        # VISL = SinglePhaseModel.emulsion(ESP_well_1.ESP_GEO['VOI'], ESP_well_1.ESP_GEO['R2'], VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
        #                 df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
        VISL = SinglePhaseModel.emulsion(6e-6, 0.04, VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
                        df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
        VISL = VISL * 1000
        GOR = df_pump_info['油汽比'][0]
        
        Pave_pump = 500         # assume Pave_pump = 500 psi
        GOR = GOR*14.7/Pave_pump
        GLR = GOR*(100-WC)/100
        HG = GLR/(GLR+1)
        HL = 1 - HG
        VISM = VISL*HL+0.000018*HG
        DENM = (DENL*1000*HL+1.224*Pave_pump/14.7*HG)/1000
        SSU = VISM / DENM
        # field_BHP = df_well['电压'].mean()*df_well['电流'].mean()/1000/df_pump_info['泵总级数'][0] * (df_pump_info['轴功率'][0]/df_pump_info['额定功率'][0])
        field_BHP = df_well['电压'].mean()*df_well['电流'].mean()/1000/df_pump_info['泵总级数'][0]
        # pump_curve calibration
        _ = np.vectorize(ESP_well_1.viscosity_calibrate)(SSU, df_pump_info['生产频率'][0]) 
        ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''
        Q, H, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)

        ''' option 2: by list inputs (original production data)'''
        WC_list = df_well['含水']
        GLR_list = GOR*(100-WC_list)/100
        HG_list = GLR_list/(GLR_list+1)
        HL_list = 1 - HG_list
        VISM_list = VISL*HL_list+0.000018*HG_list
        DENM_list = (DENL*1000*HL_list+1.224*Pave_pump/14.7*HG_list)/1000
        SSU_list = VISM_list / DENM_list
        field_BHP_list = df_well['电压']*df_well['电流']/1000/df_pump_info['泵总级数'][0]
        # print('well '+str(i)+' field BHP: ', field_BHP)

        RPM = df_pump_info['生产频率'][0]*np.ones(len(SSU_list))
        pump_curve = np.vectorize(ESP_well_1.viscosity_calibrate)(SSU_list, RPM) 
        Q_list, H_list, EFF_list = np.vectorize(ESP_well_1.solve_performance_from_BHP)(field_BHP_list, pump_curve)
        
        '''Q result'''
        print('泵:',ESP_well_1.pump_name,
                '日液:',df_pump_info['日液'][0], 'Q:',round(Q*HL), 'Q误差:',round((Q-df_pump_info['日液'][0])/df_pump_info['日液'][0],2), 
                'H:',round(H,2), 'EFF:',round(EFF,2), 
                '现场功率:',round(field_BHP,2), 
                'BHP:',round(df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],2),
                'VISL:',round(VISL))
        df_well.insert(df_well.shape[1], "预测排量", Q_list)
        df_well.insert(df_well.shape[1], "预测扬程", H_list*df_pump_info['泵总级数'][0])
        df_well.to_excel('井'+str(i)+'.xlsx')
        
        plt.show()

        
        
        ''' velocity, pressure, HL ... profile based on Q and H '''

        

        '''Head curve comparison'''
        # fig, ax = plt.subplots(dpi=300)
        # ax_twin = ax.twinx()
        # fig2, bx = plt.subplots(dpi=300)
        # bx_twin = ax.twinx()
        # for i in range(ESP_well_1.df_error.shape[0]):
        #     pump = ESP_well_1.df_error['Pump'][i]
        #     ESP_well_1.calibrate_water_curve(pump)
        #     _ = ESP_well_1.viscosity_calibrate(SSU, df_pump_info['生产频率'][0]) 
            
        #     # solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP)
        #     Q = np.linspace(1, round(df_pump_info['试验排量']/100*2)*100, 100)
        #     ax.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_H_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**2, label='水:'+pump)
        #     bx.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_BHP_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**3, label='水:'+pump)
        #     #viscosity pump curve
        #     ax.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['H_m'], label='现场:'+pump)
        #     bx.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['BHP_kW'], label='现场:'+pump)
        # # real points
        # ax.scatter(df_pump_info['试验排量'][0], df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],marker='*', s=50, label='测试点')
        # bx.scatter(df_pump_info['试验排量'][0], df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],marker='*', s=50, label='测试点')
        # # plot settings
        # # head
        # ax.set_xlim(0,(Q*(df_pump_info['电泵额定转速'][0]/3600)).max()*1.5)
        # ax.set_ylim(0,(ESP_well_1.Q_H_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**2).max()*1.2)
        # ax.legend(loc='upper right', fontsize = 6)
        # # horsepower
        # bx.set_xlim(0,(Q*(df_pump_info['电泵额定转速'][0]/3600)).max()*1.5)
        # bx.set_ylim(0,ESP_well_1.field_pump_curve['BHP_kW'].max()*1.2)
        # bx.legend(loc='lower right', fontsize = 6)
        # # save fig
        # fig.tight_layout()
        # fig.savefig('井'+str(ESP_well_1.well_name)+'扬程.jpg')
        # fig2.tight_layout()
        # fig2.savefig('井'+str(ESP_well_1.well_name)+'功率.jpg')


    '''pump curve plot '''
    # final_pump = 'none'
    # final_error = 10
    # for i in rang(ESP_well_1.df_error.shape[0]):
        # pump = ESP_well_1.df_error['Pump']
        # ESP_well_1.calibrate_water_curve(pump)
        # _ = ESP_well_1.viscosity_calibrate(SSU, df_pump_info['生产频率'][0]) 
        # ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''
        # Q_field, H_field, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)
        
        # print('泵%30s'%ESP_well_1.pump_name,
        #     '日液:',df_pump_info['日液'][0], 'Q:%6.2f'%(Q_field*HL), 'Q误差:',round((Q_field-df_pump_info['日液'][0])/df_pump_info['日液'][0],2), 
        #     'H:',round(H,2), 'EFF:',round(EFF,2), 
        #     '现场功率:',round(field_BHP,2), 
        #     'BHP:',round(df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],2),
        #     'VISL:',round(VISL))
        # if abs(round((Q_field-df_pump_info['日液'][0])/df_pump_info['日液'][0],2)) < abs(final_error):
        #     final_pump = pump
        #     final_error = round((Q_field-df_pump_info['日液'][0])/df_pump_info['日液'][0],2)
            
        # ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''
        # fig, ax = plt.subplots(dpi=300)
        # ax_twin = ax.twinx()
        # Q = np.linspace(1, round(df_pump_info['试验排量']/100*2)*100, 100)
        # ax.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_H_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**2, 'b', label='水扬程')
        # ax_twin.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_BHP_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**3, 'r', label='水功率')
        # # ax_twin.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_EFF_coeff(Q), 'g')
        # # ax_twin.scatter(df_pump_info['试验排量'][0], df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0],marker='*', c='g')

        # '''viscosity pump curve'''
        # ax.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['H_m'], c='b', label='现场扬程')
        # ax_twin.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['BHP_kW'], c='r', label='现场功率')
        # # ax_twin.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['Efficiency'], c='g')
        
        # '''real points'''
        # ax.scatter(df_pump_info['试验排量'][0], df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],marker='*', c='b', s=50, label='水扬程')
        # ax_twin.scatter(df_pump_info['试验排量'][0], df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],marker='*', c='r', s=50, label='水功率')
        # ax_twin.scatter(df_pump_info['日液'][0], field_BHP,marker='s', c='black', label='现场', s=30)
        # ax_twin.scatter(Q_field*HL, field_BHP,marker='x', c='c', label='拟合', s=30, linewidths=1.5)

        # ax.set_xlim(0,(Q*(df_pump_info['电泵额定转速'][0]/3600)).max()*1.5)
        # ax.set_ylim(0,(ESP_well_1.Q_H_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**2).max()*1.2)
        # ax_twin.set_xlim(0,(Q*(df_pump_info['电泵额定转速'][0]/3600)).max()*1.5)
        # ax_twin.set_ylim(0,ESP_well_1.field_pump_curve['BHP_kW'].max()*1.2)
        # ax.legend(loc='upper right')
        # ax_twin.legend(loc='lower right')
        # fig.tight_layout()
        # fig.savefig('井'+str(ESP_well_1.well_name)+'泵 '+pump+'.jpg')

    '''pump curve plot '''
    
    print(final_pump,' error:%4.2f'%final_error)
    pump = final_pump
    ESP_well_1.calibrate_water_curve(pump)
    _ = ESP_well_1.viscosity_calibrate(SSU, df_pump_info['生产频率'][0]) 
    ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''
    Q_field, H_field, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)
    fig, ax = plt.subplots(dpi=300)
    ax_twin = ax.twinx()
    Q = np.linspace(1, round(df_pump_info['试验排量']/100*2)*100, 100)
    ax.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_H_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**2, 'b', label='水扬程')
    ax_twin.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_BHP_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**3, 'r', label='水功率')
    # ax_twin.plot(Q*(df_pump_info['电泵额定转速'][0]/3600), ESP_well_1.Q_EFF_coeff(Q), 'g')
    # ax_twin.scatter(df_pump_info['试验排量'][0], df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0],marker='*', c='g')

    '''viscosity pump curve'''
    ax.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['H_m'], c='b', label='现场扬程')
    ax_twin.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['BHP_kW'], c='r', label='现场功率')
    # ax_twin.scatter(ESP_well_1.field_pump_curve['Q_m3d'], ESP_well_1.field_pump_curve['Efficiency'], c='g')
    
    '''real points'''
    ax.scatter(df_pump_info['试验排量'][0], df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],marker='*', c='b', s=50, label='水扬程')
    ax_twin.scatter(df_pump_info['试验排量'][0], df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],marker='*', c='r', s=50, label='水功率')
    ax_twin.scatter(df_pump_info['日液'][0], field_BHP,marker='s', c='black', label='现场', s=30)
    ax_twin.scatter(Q_field*HL, field_BHP,marker='x', c='c', label='拟合', s=30, linewidths=1.5)

    ax.set_xlim(0,(Q*(df_pump_info['电泵额定转速'][0]/3600)).max()*1.5)
    ax.set_ylim(0,(ESP_well_1.Q_H_coeff(Q)*(df_pump_info['电泵额定转速'][0]/3600)**2).max()*1.2)
    ax_twin.set_xlim(0,(Q*(df_pump_info['电泵额定转速'][0]/3600)).max()*1.5)
    ax_twin.set_ylim(0,ESP_well_1.field_pump_curve['BHP_kW'].max()*1.2)
    ax.legend(loc='upper right')
    ax_twin.legend(loc='lower right')
    fig.tight_layout()
    plt.show()

def empricial_calculation():
    
    QBEP_range = 0.12
    QD_range = 0.1
    # well number
    # 1, 7, 10 wrong data, pump damage?
    # 2, 5, 9 high error
    # rest ok 6, 12, 4, 3, 11, 8
    i = 2 # well name/number
    j = 1 # production data (pick first point as an example)

    for i in [2]:
        '''well profile'''
        try:
            well_name = str(i)
            df_well_profile = read_input('Well profile/'+well_name+'.xlsx')
        except:
            print(i)
        
        '''pump infomation'''
        df_pump_info = pd.read_excel('Data/Pump data/'+str(i)+'.xlsx', index_col=None)
        '''well production infomation'''
        df_well = pd.read_excel('Data/Production data/'+str(i)+'.xls',skiprows=1, index_col=None, na_values=['NA'])
        df_well.drop(0, axis=0,inplace=True)
        df_well.drop(df_well.tail(1).index, axis=0,inplace=True)
        df_well['电流'] = df_well['电流'].astype(float)
        df_well['电压'] = df_well['电压'].astype(float)
        df_well.replace('', np.nan, inplace=True)
        df_well.dropna(subset=['电流','电压','含水'], inplace=True)
        df_well.reset_index(inplace=True)

        ''' select pump and generate pump curve at 3600 RPM'''
        ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
                        df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
                        df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
                        QBEP_range=QBEP_range, QD_range=QD_range,Total_SN=df_pump_info['泵总级数'][0],
                        SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)
        # print(ESP_well_1.df_error)

        ''' prediction with field production data'''
        Q_list = []
        Pin_list = []
        Pout_list = []
        Liquid_level_list = []
        for j in range(0, df_well.shape[0]):
            ''' pump curve calibration based on (speed and SSU (viscosity and density (air can be included)) '''
            # Assume/calculate Q, P
            # calculate mixture density (oil, water, air), initial assumption homongenous model
            # calculate mixture viscosity (oil, water, air), initial assumption homongenous model
            # calculate SSU
            SQLname = 'SQL_database/Default_pump.db'
            VISL = df_pump_info['粘度'][0]
            WC = df_well['含水'][j]
            DENL = 0.95 # not sure
            # VISL = SinglePhaseModel.emulsion(ESP_well_1.ESP_GEO['VOI'], ESP_well_1.ESP_GEO['R2'], VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
            #                 df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
            VISL = SinglePhaseModel.emulsion(6e-6, 0.04, VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
                            df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
            VISL = VISL * 1000
            GOR = df_pump_info['油汽比'][0]
            
            Pave_pump = 500         # assume Pave_pump = 500 psi
            GOR = GOR*14.7/Pave_pump
            GLR = GOR*(100-WC)/100
            HG = GLR/(GLR+1)
            HL = 1 - HG
            VISM = VISL*HL+0.000018*HG
            DENM = (DENL*1000*HL+1.224*Pave_pump/14.7*HG)/1000
            SSU = VISM / DENM
            field_BHP = df_well['电压'][j]*df_well['电流'][j]/1000/df_pump_info['泵总级数'][0]
            # pump_curve calibration
            _ = np.vectorize(ESP_well_1.viscosity_calibrate)(SSU, df_pump_info['生产频率'][0]) 
            ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''
            Q, H, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)

            ''' pressure profile '''
            inputValues = {"well_name": well_name, 
                            "Surface_line_pressure": df_well['油压'][j]*1e6, "Pipe_diameter":df_pump_info['油管内径'][0],      
                            "Roughness":1e-5,                   "Liquid_viscosity":df_pump_info['粘度'][0]/1000,
                            "Liquid_relative_density":0.9456,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
                                "Water_cut": df_well['含水'][j], 
                            "Reservoir_C":1.83E-17,     "Reservoir_n":1, 
                            "Reservoir_P":1.87e7,               "GLR":df_pump_info['油汽比'][0]*((100-df_well['含水'][j])/100),   
                            "Geothermal_gradient":0.03, 
                            "Surface_T":df_well['井口温度'][j]+273.15, "Pump_intake_T": 380, "P_in": 3.85, "P_out":13.53, 
                            "ESP_Depth": df_pump_info['泵挂'][0], "ESP_SN":df_pump_info['泵总级数'][0], 
                            "ESP_N":df_pump_info['生产频率'][0]/60*3600, "ESP_length": 10, 'SSU':SSU}
                            
            '''pressure profile'''
            ESP_input=ESP_default['Flex31']
            QBEM = 10000
            WC=inputValues['Water_cut']
            GLR = inputValues['GLR']
            ESP_input['SN']=inputValues['ESP_SN']
            ESP_input['N']=inputValues['ESP_N']

            '''Pressure profile'''
            Qg_res = Q*GLR/24/3600
            Qo_res = Q*(1-WC/100)/24/3600
            Qw_res = Q*WC/100/24/3600
            P, df, P_ESP_in, P_ESP_out = P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input, 
                        ESP_empirical_H=H*m_to_pa*df_pump_info['泵总级数'][0])
            Liquid_level = (P-df_well['油压'][j]*1e6)/m_to_pa
            Q_list.append(Q*HL)
            Pin_list.append(P_ESP_in/1e6)
            Pout_list.append(P_ESP_out/1e6)
            Liquid_level_list.append(Liquid_level)
            print('日期：',df_well['日期'][j], ' 预测排量：',Q,  '基准排量：',df_pump_info['日液'][0])

        # P_profile_plot (df_well_profile, df, Table_name='Sheng_li_pump', well_name, inputValues['P_in'], inputValues['P_out'])

        df_well.insert(df_well.shape[1], "预测排量", Q_list)
        df_well.insert(df_well.shape[1], "预测泵入口压力", Pin_list)
        df_well.insert(df_well.shape[1], "预测泵出口压力", Pout_list)
        df_well.insert(df_well.shape[1], "预测动液位", Liquid_level_list)
        df_well.to_excel('井'+str(i)+'.xlsx')
    plt.show()

def mechanistic_model_calculation ():
    
    # empricial_calculation()
    QBEP_range = 0.12
    QD_range = 0.1
    # well number
    # 1, 7, 10 wrong data, pump damage?
    # 2, 5, 9 high error
    # rest ok 6, 12, 4, 3, 11, 8
    i = 9 # well name/number
    j = 1 # production data (pick first point as an example)
    # for i in range(1,13):
    # for i in [6,12,5,6,7,8,10,12]:
    for i in [2]:
        '''well profile'''
        try:
            well_name = str(i)
            df_well_profile = read_input('Well profile/'+well_name+'.xlsx')
        except:
            print(i)
        
        '''pump infomation'''
        df_pump_info = pd.read_excel('Data/Pump data/'+str(i)+'.xlsx', index_col=None)
        '''well infomation'''
        df_well = pd.read_excel('Data/Production data/'+str(i)+'.xls',skiprows=1, index_col=None, na_values=['NA'])
        df_well.drop(0, axis=0,inplace=True)
        df_well.drop(df_well.tail(1).index, axis=0,inplace=True)
        df_well['电流'] = df_well['电流'].astype(float)
        df_well['电压'] = df_well['电压'].astype(float)
        df_well.replace('', np.nan, inplace=True)
        df_well.dropna(subset=['电流','电压','含水'], inplace=True)
        df_well.reset_index(inplace=True)

        ''' select pump and generate pump curve at 3600 RPM'''
        ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
                        df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
                        df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
                        QBEP_range=QBEP_range, QD_range=QD_range,Total_SN=df_pump_info['泵总级数'][0],
                        SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)
        # print(ESP_well_1.df_error)

        ''' prediction with field production data'''
        Q_list = []
        Pin_list = []
        Pout_list = []
        Liquid_level_list = []
        '''ESP'''
        ''' Generate ESP geometry '''
        # SPSA train
        # ESP_input, QBEM = ESP_well_1.train_ESP_GEO()
        # use saved trained SPSA data
        ESP_input_ori = pd.read_excel(well_name+' ESP GEO.xlsx', header=0, index_col=0)
        ESP_input = ESP_input_ori.to_dict()
        for column in ESP_input_ori:
            ESP_input[column] = ESP_input_ori[column][0]
        QBEM = ESP_input['QBEM']
        Pave_pump = 500
        WC = df_pump_info['含水'][0]
        GOR = df_pump_info['油汽比'][0]
        GOR = GOR*14.7/Pave_pump
        GLR = GOR*(100-WC)/100
        VISL = df_pump_info['粘度'][0]/1000
        
        Q_model = np.arange(0.05, 1, 0.05)*max(ESP_well_1.field_pump_curve['Q_m3d'])/bpd_to_m3d
        H_model = ESP_head (QBEM, ESP_input, Q_model, 0.001, 1000, VISL, 
                        0.9456*1000, VISG_in = 0.000018, 
                        WC=WC, GLR=GLR, P = Pave_pump*psi_to_pa, T=288, O_W_ST = 0.035, GVF = None)
        ESP_well_1.ESP_Model_calibrate(Q_model*bpd_to_m3d, H_model*psi_to_ft*ft_to_m)


        for j in range(df_well.shape[0]):
            ''' pump curve calibration based on (speed and SSU (viscosity and density (air can be included)) '''
            # Assume/calculate Q, P
            # calculate mixture density (oil, water, air), initial assumption homongenous model
            # calculate mixture viscosity (oil, water, air), initial assumption homongenous model
            # calculate SSU
            SQLname = 'SQL_database/Default_pump.db'
            VISL = df_pump_info['粘度'][0]
            WC = df_well['含水'][j]
            DENL = 0.95 # not sure
            # VISL = SinglePhaseModel.emulsion(ESP_well_1.ESP_GEO['VOI'], ESP_well_1.ESP_GEO['R2'], VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
            #                 df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
            VISL = SinglePhaseModel.emulsion(6e-6, 0.04, VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
                            df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
            VISL = VISL * 1000
            GOR = df_pump_info['油汽比'][0]
            
            Pave_pump = 500         # assume Pave_pump = 500 psi
            GOR = GOR*14.7/Pave_pump
            GLR = GOR*(100-WC)/100
            HG = GLR/(GLR+1)
            HL = 1 - HG
            VISM = VISL*HL+0.000018*HG
            DENM = (DENL*1000*HL+1.224*Pave_pump/14.7*HG)/1000
            SSU = VISM / DENM
            field_BHP = df_well['电压'][j]*df_well['电流'][j]/1000/df_pump_info['泵总级数'][0]
            # pump_curve calibration
            _ = np.vectorize(ESP_well_1.viscosity_calibrate)(SSU, df_pump_info['生产频率'][0]) 
            ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''


            ''' pressure profile '''
            inputValues = {"well_name": well_name, 
                            "Surface_line_pressure": df_well['油压'][j]*1e6, "Pipe_diameter":df_pump_info['油管内径'][0],      
                            "Roughness":1e-5,                   "Liquid_viscosity":df_pump_info['粘度'][0]/1000,
                            "Liquid_relative_density":0.9456,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
                                "Water_cut": df_well['含水'][j], 
                            "Reservoir_C":1.83E-17,     "Reservoir_n":1, 
                            "Reservoir_P":1.87e7,               "GLR":df_pump_info['油汽比'][0]*((100-df_well['含水'][j])/100),   
                            "Geothermal_gradient":0.03, 
                            "Surface_T":df_well['井口温度'][j]+273.15, "Pump_intake_T": 380, "P_in": 3.85, "P_out":13.53, 
                            "ESP_Depth": df_pump_info['泵挂'][0], "ESP_SN":df_pump_info['泵总级数'][0], 
                            "ESP_N":df_pump_info['生产频率'][0]/60*3600, "ESP_length": 10, 'SSU':SSU}
                            
            '''pressure profile'''
            ESP_Depth = inputValues['ESP_Depth']
            VisL_target = inputValues['Liquid_viscosity']*1000

            ESP_input['SN']=inputValues['ESP_SN']
            ESP_input['N']=inputValues['ESP_N']
            ''' Q, H predictioon'''
            Q, H, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)
            
            # ESP_curve_plot (ESP_input, inputValues, QBEM, VISL_list=[1,10,50,100,300,500,700,1000], ESP_empirical_class = ESP_well_1)



            '''Pressure profile'''
            if Q<=0: Q=1
            Qg_res = Q*GLR/24/3600
            Qo_res = Q*(1-WC/100)/24/3600
            Qw_res = Q*WC/100/24/3600
            P, df, P_ESP_in, P_ESP_out = P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input, 
                        ESP_empirical_H=H*m_to_pa*df_pump_info['泵总级数'][0])
            Liquid_level = (P-df_well['套压'][j]*1e6)/m_to_pa
            Q_list.append(Q*HL)
            Pin_list.append(P_ESP_in/1e6)
            Pout_list.append(P_ESP_out/1e6)
            Liquid_level_list.append(Liquid_level)
            print('日期：',df_well['日期'][j], ' 预测排量：',Q,  '基准排量：',df_pump_info['日液'][0])

            # P_profile_plot (df_well_profile, df, Table_name='Sheng_li_pump', well_name, inputValues['P_in'], inputValues['P_out'])

        df_well.insert(df_well.shape[1], "预测排量", Q_list)
        df_well.insert(df_well.shape[1], "预测泵入口压力", Pin_list)
        df_well.insert(df_well.shape[1], "预测泵出口压力", Pout_list)
        df_well.insert(df_well.shape[1], "预测动液位", Liquid_level_list)

        df_well.to_excel('井'+str(i)+'.xlsx')
    plt.show()

def version_2022():
    QBEP_range = 0.12
    QD_range = 0.1
    # empricial_calculation()
    # mechanistic_model_calculation()
    j=1
    for i in [2]:
        well_name=str(i)
        '''pump infomation'''
        df_pump_info = pd.read_excel('Data/Pump data/'+str(i)+'.xlsx', index_col=None)
        '''well infomation'''
        df_well = pd.read_excel('Data/Production data/'+str(i)+'.xls',skiprows=1, index_col=None, na_values=['NA'])
        df_well.drop(0, axis=0,inplace=True)
        df_well.drop(df_well.tail(1).index, axis=0,inplace=True)
        df_well['电流'] = df_well['电流'].astype(float)
        df_well['电压'] = df_well['电压'].astype(float)
        df_well.replace('', np.nan, inplace=True)
        df_well.dropna(subset=['电流','电压','含水'], inplace=True)
        df_well.reset_index(inplace=True)

        ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
                        df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
                        df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
                        QBEP_range=QBEP_range, QD_range=QD_range,Total_SN=df_pump_info['泵总级数'][0],
                        SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)

        SQLname = 'SQL_database/Default_pump.db'
        VISL = df_pump_info['粘度'][0]
        WC = df_well['含水'][j]
        DENL = 0.95 # not sure
        # VISL = SinglePhaseModel.emulsion(ESP_well_1.ESP_GEO['VOI'], ESP_well_1.ESP_GEO['R2'], VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
        #                 df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
        VISL = SinglePhaseModel.emulsion(6e-6, 0.04, VISL/1000, 0.001, DENL*1000, 1000, WC, 0.025, 
                        df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0]/24/3600, df_pump_info['泵总级数'][0])
        VISL = VISL * 1000
        GOR = df_pump_info['油汽比'][0]
        Pave_pump = 500         # assume Pave_pump = 500 psi
        GOR = GOR*14.7/Pave_pump
        GLR = GOR*(100-WC)/100
        HG = GLR/(GLR+1)
        HL = 1 - HG
        VISM = VISL*HL+0.000018*HG
        DENM = (DENL*1000*HL+1.224*Pave_pump/14.7*HG)/1000
        SSU = VISM / DENM
        field_BHP = df_well['电压'][j]*df_well['电流'][j]/1000/df_pump_info['泵总级数'][0]
        # pump_curve calibration
        _ = np.vectorize(ESP_well_1.viscosity_calibrate)(SSU, df_pump_info['生产频率'][0]) 
        ''' select pump and generate pump curve at 3600 RPM'''
        ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
                        df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
                        df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
                        QBEP_range=QBEP_range, QD_range=QD_range,Total_SN=df_pump_info['泵总级数'][0],
                        SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)
        # print(ESP_well_1.df_error)

        '''ESP'''
        ''' Generate ESP geometry '''
    #     # SPSA train
    #     # ESP_input, QBEM = ESP_well_1.train_ESP_GEO()
        # use saved trained SPSA data
        ESP_input_ori = pd.read_excel(well_name+' ESP GEO.xlsx', header=0, index_col=0)
        ESP_input = ESP_input_ori.to_dict()
        for column in ESP_input_ori:
            ESP_input[column] = ESP_input_ori[column][0]
        QBEM = ESP_input['QBEM']
        Pave_pump = 500
        WC = df_pump_info['含水'][0]
        GOR = df_pump_info['油汽比'][0]
        GOR = GOR*14.7/Pave_pump
        GLR = GOR*(100-WC)/100
        VISL = df_pump_info['粘度'][0]/1000
        
        ''' pressure profile '''
        inputValues = {"well_name": well_name, 
                        "Surface_line_pressure": df_well['油压'][j]*1e6, "Pipe_diameter":df_pump_info['油管内径'][0],      
                        "Roughness":1e-5,                   "Liquid_viscosity":df_pump_info['粘度'][0]/1000,
                        "Liquid_relative_density":0.9456,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
                            "Water_cut": df_well['含水'][j], 
                        "Reservoir_C":1.83E-17,     "Reservoir_n":1, 
                        "Reservoir_P":1.87e7,               "GLR":df_pump_info['油汽比'][0]*((100-df_well['含水'][j])/100),   
                        "Geothermal_gradient":0.03, 
                        "Surface_T":df_well['井口温度'][j]+273.15, "Pump_intake_T": 380, "P_in": 3.85, "P_out":13.53, 
                        "ESP_Depth": df_pump_info['泵挂'][0], "ESP_SN":df_pump_info['泵总级数'][0], 
                        "ESP_N":df_pump_info['生产频率'][0]/60*3600, "ESP_length": 10, 'SSU':SSU}
               
        ESP_curve_plot (ESP_input, inputValues, QBEM, VISL_list=[1,10,50,100,300,500,700,1000], ESP_empirical_class = ESP_well_1)


    # # well number
    # # 1, 7, 10 wrong data, pump damage?
    # # 2, 5, 9 high error
    # # rest ok 6, 12, 4, 3, 11, 8
    # i = 8 # well name/number
    # j = 1 # production data (pick first point as an example)
    # '''well profile'''
    # # try:
    # #     well_name = str(i)
    # #     df_well_profile = read_input('Well profile/'+well_name+'.xlsx')
    # # except:
    # #     print(i)
    # for i in [5,6,7,8,9,10,12]:
    #     '''pump infomation'''
    #     df_pump_info = pd.read_excel('Data/Pump data/'+str(i)+'.xlsx', index_col=None)
    #     '''well infomation'''
    #     df_well = pd.read_excel('Data/Production data/'+str(i)+'.xls',skiprows=1, index_col=None, na_values=['NA'])
    #     df_well.drop(0, axis=0,inplace=True)
    #     df_well.drop(df_well.tail(1).index, axis=0,inplace=True)
    #     df_well['电流'] = df_well['电流'].astype(float)
    #     df_well['电压'] = df_well['电压'].astype(float)
    #     df_well.replace('', np.nan, inplace=True)
    #     df_well.dropna(subset=['电流','电压','含水'], inplace=True)
    #     df_well.reset_index(inplace=True)

    #     ''' select pump and generate pump curve at 3600 RPM'''
        # ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
        #                 df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
        #                 df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
        #                 QBEP_range=QBEP_range, QD_range=QD_range,Total_SN=df_pump_info['泵总级数'][0]
        #                 SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)
    #     print(ESP_well_1.df_error)

    #     ''' Generate ESP geometry '''
    #     # SPSA train
    #     ESP_input, QBEM = ESP_well_1.train_ESP_GEO()

    plt.show()

def SPSA_Train_ESP_GEO():

    '''get ESP geometry, and test ESP performance prediction'''
    j=1 # production data (pick first point as an example)

    for i in [1,3,4,5,6,7,8,9,10,11,12]:        # well/pump name/number
    # for i in [2]: # well/pump name/number

        well_name=str(i)
        '''pump infomation'''
        df_pump_info = pd.read_excel(path_abs+'/Data/Pump data/'+str(i)+'.xlsx', index_col=None)
        '''well infomation'''
        df_well = pd.read_excel(path_abs+'/Data/Production data/'+str(i)+'.xls',skiprows=1, index_col=None, na_values=['NA'])
        df_well.drop(0, axis=0,inplace=True)
        df_well.drop(df_well.tail(1).index, axis=0,inplace=True)
        df_well['电流'] = df_well['电流'].astype(float)
        df_well['电压'] = df_well['电压'].astype(float)
        df_well.replace('', np.nan, inplace=True)
        df_well.dropna(subset=['电流','电压','含水'], inplace=True)
        df_well.reset_index(inplace=True)


        ''' Generate ESP geometry (SPSA) '''
        
        # Gnerate ESP clase based on pump curve
        ESP_well_1 = ESP_curve(i, i)
        ESP_well_1.input_pump(op_string = "SELECT * FROM Sheng_li_pump Where Pump = "+str(i))

        # ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
        #                 df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
        #                 df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
        #                 QBEP_range=QBEP_range, QD_range=QD_range, Total_SN=df_pump_info['泵总级数'][0],
        #                 SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)
        # print(ESP_well_1.df_error)
        # SPSA train
        ESP_input, QBEM = ESP_well_1.train_ESP_GEO()

        '''Plot pump curves'''
    #     # use saved trained SPSA data
    #     ESP_input_ori = pd.read_excel(path_abs+'/'+well_name+' ESP GEO.xlsx', header=0, index_col=0)
    #     ESP_input = ESP_input_ori.to_dict()
    #     for column in ESP_input_ori:
    #         ESP_input[column] = ESP_input_ori[column][0]

    #     QBEM = ESP_input['QBEM']
    #     Pave_pump = 500
    #     WC = df_pump_info['含水'][0]
    #     GOR = df_pump_info['油汽比'][0]
    #     GOR = GOR*14.7/Pave_pump
    #     GLR = GOR*(100-WC)/100
    #     VISL = df_pump_info['粘度'][0]/1000
    #     VISO = df_pump_info['粘度'][0]
    #     WC = df_well['含水'][j]
    #     DENO = 0.95 # not sure

    #     ESP_class1 = ESP_TUALP(ESP_GEO=ESP_input)
    #     VISL = ESP_class1.emulsion(VISO=VISO/1000, DENO=DENO*1000, WC=WC/100, STOW=0.025, RPM=df_pump_info['电泵额定转速'][0], 
    #             QL=df_pump_info['试验排量'][0]/24/3600, SN=df_pump_info['泵总级数'][0])
    #     VISL = VISL * 1000
    #     DENL = (DENO * (100-WC) + 1000*WC)/100
    #     GOR = df_pump_info['油汽比'][0]
    #     Pave_pump = 500         # assume Pave_pump = 500 psi
    #     GOR = GOR*14.7/Pave_pump
    #     GLR = GOR*(100-WC)/100
    #     HG = GLR/(GLR+1)
    #     HL = 1 - HG
    #     VISM = VISL*HL+0.000018*HG
    #     DENM = (DENL*1000*HL+1.224*Pave_pump/14.7*HG)/1000
    #     SSU = VISM / DENM
    #     field_BHP = df_well['电压'][j]*df_well['电流'][j]/1000/df_pump_info['泵总级数'][0]
    #     ''' select pump and generate pump curve at 3600 RPM'''
        
    #     # Gnerate ESP clase based on pump curve
    #     ESP_well_1 = ESP_curve(i, i)
    #     ESP_well_1.input_pump(op_string = "SELECT * FROM Sheng_li_pump Where Pump = "+str(i))
    #     # pump_curve calibration
    #     _ = np.vectorize(ESP_well_1.viscosity_calibrate)(SSU, df_pump_info['生产频率'][0]) 

    #     # ESP_well_1 = ESP_curve(df_pump_info['泵直径'][0], df_pump_info['电泵额定转速'][0], df_pump_info['试验排量'][0], 
    #     #                 df_pump_info['试验扬程'][0]/df_pump_info['泵总级数'][0],df_pump_info['轴功率'][0]/df_pump_info['泵总级数'][0],
    #     #                 df_pump_info['输出功率'][0]/df_pump_info['轴功率'][0], 
    #     #                 QBEP_range=QBEP_range, QD_range=QD_range, Total_SN=df_pump_info['泵总级数'][0],
    #     #                 SQLname = 'SQL_database/Default_pump.db', Table_name='Sheng_li_pump', pump_name=i, well_name=i)
    #     # print(ESP_well_1.df_error)

    #     ''' pressure profile '''
    #     inputValues = {"well_name": well_name, 
    #                     "Surface_line_pressure": df_well['油压'][j]*1e6, "Pipe_diameter":df_pump_info['油管内径'][0],      
    #                     "Roughness":1e-5,                   "Liquid_viscosity":df_pump_info['粘度'][0]/1000,
    #                     "Liquid_relative_density":0.9456,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
    #                         "Water_cut": df_well['含水'][j], 
    #                     "Reservoir_C":1.83E-17,     "Reservoir_n":1, 
    #                     "Reservoir_P":1.87e7,               "GLR":df_pump_info['油汽比'][0]*((100-df_well['含水'][j])/100),   
    #                     "Geothermal_gradient":0.03, 
    #                     "Surface_T":df_well['井口温度'][j]+273.15, "Pump_intake_T": 380, "P_in": 3.85, "P_out":13.53, 
    #                     "ESP_Depth": df_pump_info['泵挂'][0], "ESP_SN":df_pump_info['泵总级数'][0], 
    #                     "ESP_N":df_pump_info['生产频率'][0]/60*3600, "ESP_length": 10, 'SSU':SSU}
               
        # ESP_curve_plot (ESP_input, inputValues, QBEM, VISL_list=[1,10,50,100,300,500,700,1000], ESP_empirical_class = ESP_well_1)

    plt.show()

def Plot_ESP_field_curve(ESP_input, inputValues, QBEM, VISL_list=[1,10,50,100,300,500,700,1000], ESP_empirical_class = None, Cal_Q=None, Target_HP=None, Target_Q=None):
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

        
        ESP_class1 = ESP_TUALP(ESP_GEO=ESP_input)
        try:
            VISL = ESP_class1.emulsion(VISO=VISO_in, DENO=DENO_in, WC=WC/100, STOW=0.025, RPM=inputValues['ESP_N'], 
                        QL = Cal_Q/24/3600, SN=inputValues['ESP_SN'])
        except: pass
        

    except:
        print('输入流动数据有误')
        return
    # try:
    # fig, ax = plt.subplots(dpi=300, figsize = (3.33,2.5), nrows=1, ncols=1)

    fig, ax = plt.subplots(dpi=300, figsize = (3.33,2.5))    # ax: head 压头
    fig.subplots_adjust(right=0.75)
    fig.subplots_adjust(left=0.25)
    ax_twinx = ax.twinx()       # efficiency 效率
    ax_twinx3 = ax.twinx()      # horsepower 功率
    ax_twinx3.spines.left.set_position(("axes", -0.2))

    if ESP_empirical_class != None:
        pump_name = ESP_empirical_class.pump_name

    # field 现场
    p1, = ax.plot(ESP_empirical_class.field_pump_curve['Q_m3d'], ESP_empirical_class.field_pump_curve['H_m'], 'r', label='现场压头:井'+str(pump_name))
    p2, = ax_twinx.plot(ESP_empirical_class.field_pump_curve['Q_m3d'],ESP_empirical_class.field_pump_curve['Total_eff'], 'b', label='现场总效率:井'+str(pump_name))
    p4, = ax_twinx3.plot(ESP_empirical_class.field_pump_curve['Q_m3d'],ESP_empirical_class.field_pump_curve['Total_kW'], 'c', label='现场总功率:井'+str(pump_name))
    p4, = ax_twinx3.plot(ESP_empirical_class.field_pump_curve['Q_m3d'],ESP_empirical_class.field_pump_curve['BHP_kW'], 'g', label='现场泵功率:井'+str(pump_name))
    if Cal_Q != None: ax_twinx3.scatter(Cal_Q,Target_HP,color = 'black',marker='.',label='计算值('+str(int(Target_Q))+' m3d,'+str(round(Target_HP,2))+' kW)')
    if Target_Q != None: ax_twinx3.scatter(Target_Q,Target_HP,color = 'black',marker='x',label='现场值('+str(int(Target_Q))+' m3d,'+str(round(Target_HP,2))+' kW)')

    ax.set_xlim(0, max(ESP_empirical_class.field_pump_curve['Q_m3d'])*1.2)
    ax.set_ylim(0, max(ESP_empirical_class.field_pump_curve['H_m'])*1.2)
    ax_twinx.set_ylim(0,100)
    ax_twinx3.set_ylim(0,max(ESP_empirical_class.field_pump_curve['Total_kW'])*1.2)

    ax.set_xlabel('排量（m3/d）', fontsize=8)
    ax.set_ylabel('扬程（m）', fontsize=8)
    ax_twinx.set_ylabel("效率（%）")
    ax_twinx3.set_ylabel("功率（kW）")

    ax.set_title("井%s WC:%d 油:%dcp 乳化液:%dcp:" % (pump_name,WC,VISO_in*1000,VISL*1000), fontsize=8)
    # ax.set_title('井'+pump_name+'WC:'+str(WC)+'%', fontsize=8)
    # ax.xaxis.set_tick_params(labelsize=8)
    # ax.yaxis.set_tick_params(labelsize=8)

    ax.yaxis.label.set_color(p1.get_color())
    ax_twinx.yaxis.label.set_color(p2.get_color())
    ax_twinx3.yaxis.label.set_color(p4.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    ax_twinx.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax_twinx3.tick_params(axis='y', colors=p4.get_color(), **tkw)

    ax_twinx3.yaxis.set_label_position('left')
    ax_twinx3.yaxis.set_ticks_position('left')

    ax.legend(loc='upper left',fontsize=6)
    ax_twinx.legend(loc='lower left',fontsize=6)
    ax_twinx3.legend(loc='lower right',fontsize=6)

    fig.tight_layout()
    fig.savefig('井'+str(pump_name)+'TUALP曲线.jpg')
    # plt.show()

if __name__ == "__main__":    
    # empricial_calculation()
    # mechanistic_model_calculation()
    
    # empricial_calculation()
    QBEP_range = 0.12
    QD_range = 0.1
    # well number
    # 1, 7, 10 wrong data, pump damage?
    # 2, 5, 9 high error
    # rest ok 6, 12, 4, 3, 11, 8
    i = 9 # well name/number
    j = 1 # production data (pick first point as an example)
    # for i in range(1,13):
    # for i in [6,12,5,6,7,8,10,12]:

    # good: 2, 4, 8
    # mid: 5
    # bad: 1, 6, 7, 9,10,11,12
    # wrong: 3
    # for i in [11]:
    for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
        '''well profile'''
        # try:
        well_name = str(i)
        df_well_profile = read_input(path_abs+'/'+'Data/Well profile/'+well_name+'.xlsx')
        # except:
        #     print(i)
        
        '''pump infomation'''
        df_pump_info = pd.read_excel(path_abs+'/'+'Data/Pump data/'+str(i)+'.xlsx', index_col=None)
        '''well infomation'''
        df_well = pd.read_excel(path_abs+'/'+'Data/Production data/'+str(i)+'.xls',skiprows=1, index_col=None, na_values=['NA'])
        df_well.drop(0, axis=0,inplace=True)
        df_well.drop(df_well.tail(1).index, axis=0,inplace=True)
        df_well['电流'] = df_well['电流'].astype(float)
        df_well['电压'] = df_well['电压'].astype(float)
        df_well.replace('', np.nan, inplace=True)
        df_well.dropna(subset=['电流','电压','含水'], inplace=True)
        df_well.reset_index(inplace=True)

        ''' prediction with field production data'''
        Q_list = []
        Pin_list = []
        Pout_list = []
        Liquid_level_list = []
        '''ESP'''
        ''' Generate ESP geometry '''
        # SPSA train
        # ESP_input, QBEM = ESP_well_1.train_ESP_GEO()
        # use saved trained SPSA data
        ESP_input_ori = pd.read_excel(path_abs+'/'+well_name+' ESP GEO.xlsx', header=0, index_col=0)
        ESP_input = ESP_input_ori.to_dict()
        for column in ESP_input_ori:
            ESP_input[column] = ESP_input_ori[column][0]
       
        QBEM = ESP_input['QBEM']
        Pave_pump = 500
        WC = df_pump_info['含水'][0]
        GOR = df_pump_info['油汽比'][0]
        GOR = GOR*14.7/Pave_pump
        GLR = GOR*(100-WC)/100
        VISL = df_pump_info['粘度'][0]/1000
        VISO = df_pump_info['粘度'][0]
        WC = df_well['含水'][j]
        DENO = 0.95 # not sure

        ESP_class1 = ESP_TUALP(ESP_GEO=ESP_input)
        VISL = ESP_class1.emulsion(VISO=VISO/1000, DENO=DENO*1000, WC=WC/100, STOW=0.025, RPM=df_pump_info['电泵额定转速'][0], 
                QL=df_pump_info['试验排量'][0]/24/3600, SN=df_pump_info['泵总级数'][0])
        VISL = VISL * 1000
        DENL = (DENO *(100-WC) + WC)/100
        GOR = df_pump_info['油汽比'][0]
        Pave_pump = 500         # assume Pave_pump = 500 psi
        GOR = GOR*14.7/Pave_pump
        GLR = GOR*(100-WC)/100
        HG = GLR/(GLR+1)
        HL = 1 - HG
        VISM = VISL*HL+0.000018*HG
        DENM = (DENL*1000*HL+1.224*Pave_pump/14.7*HG)/1000
        SSU = VISM / DENM
        field_BHP = df_well['电压'][j]*df_well['电流'][j]/1000/df_pump_info['泵总级数'][0]
        
        # Gnerate ESP clase based on pump curve
        ESP_well_1 = ESP_curve(i, i)
        ESP_well_1.input_pump(op_string = "SELECT * FROM Sheng_li_pump Where Pump = "+str(i),Total_SN=df_pump_info['泵总级数'][0])
        
        '''calibrate by empirical correlation'''
        _ = np.vectorize(ESP_well_1.viscosity_calibrate)(1, df_pump_info['生产频率'][0]) 
        # _ = np.vectorize(ESP_well_1.viscosity_calibrate)(SSU, df_pump_info['生产频率'][0]) 
        
        '''calibrate by mechanisitic model'''
        Q_model = np.arange(0.05, 1, 0.05)*max(ESP_well_1.field_pump_curve['Q_m3d'])/bpd_to_m3d
        H_model = ESP_head (QBEM, ESP_input, Q_model, 0.001, 1000, VISL/1000, 
                        0.9456*1000, VISG_in = 0.000018, 
                        WC=WC, GLR=GLR, P = Pave_pump*psi_to_pa, T=288, O_W_ST = 0.035, GVF = None)
        
        
        ESP_well_1.ESP_Model_calibrate(Q_model*bpd_to_m3d, H_model*psi_to_ft*ft_to_m)

        '''test calculation'''
        field_BHP = df_well['电压'][j]*df_well['电流'][j]/1000/df_pump_info['泵总级数'][0]
        Q, H, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)
        print('日期：',df_well['日期'][j], ' 预测排量：',Q,  '基准排量：',df_pump_info['日液'][0])

        inputValues = {"well_name": well_name, 
                        "Surface_line_pressure": df_well['油压'][j]*1e6, "Pipe_diameter":df_pump_info['油管内径'][0],      
                        "Roughness":1e-5,                   "Liquid_viscosity":df_pump_info['粘度'][0]/1000,
                        "Liquid_relative_density":0.9456,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
                            "Water_cut": df_well['含水'][j], 
                        "Reservoir_C":1.83E-17,     "Reservoir_n":1, 
                        "Reservoir_P":1.87e7,               "GLR":df_pump_info['油汽比'][0]*((100-df_well['含水'][j])/100),   
                        "Geothermal_gradient":0.03, 
                        "Surface_T":df_well['井口温度'][j]+273.15, "Pump_intake_T": 380, "P_in": 3.85, "P_out":13.53, 
                        "ESP_Depth": df_pump_info['泵挂'][0], "ESP_SN":df_pump_info['泵总级数'][0], 
                        "ESP_N":df_pump_info['生产频率'][0]/60*3600, "ESP_length": 10, 'SSU':SSU}
               
        Plot_ESP_field_curve (ESP_input, inputValues, QBEM, VISL_list=[1,10,50,100,300,500,700,1000], ESP_empirical_class = ESP_well_1, 
                        Cal_Q=Q, Target_HP=field_BHP,Target_Q=df_pump_info['日液'][0])

        ''' production calculation '''
        # for j in range(df_well.shape[0]):
        #     ''' pump curve calibration based on (speed and SSU (viscosity and density (air can be included)) '''
        #     # Assume/calculate Q, P
        #     # calculate mixture density (oil, water, air), initial assumption homongenous model
        #     # calculate mixture viscosity (oil, water, air), initial assumption homongenous model
        #     # calculate SSU
        #     WC = df_well['含水'][j]             
        #     VISL = ESP_class1.emulsion(VISO=VISO/1000, DENO=DENO*1000, WC=WC/100, STOW=0.025, RPM=df_pump_info['电泵额定转速'][0], 
        #             QL=df_pump_info['试验排量'][0]/24/3600, SN=df_pump_info['泵总级数'][0])
        #     VISL = VISL * 1000
        #     GLR = GOR*(100-WC)/100
        #     HG = GLR/(GLR+1)
        #     HL = 1 - HG
        #     VISM = VISL*HL+0.000018*HG
        #     DENM = (DENL*1000*HL+1.224*Pave_pump/14.7*HG)/1000
        #     SSU = VISM / DENM
        #     field_BHP = df_well['电压'][j]*df_well['电流'][j]/1000/df_pump_info['泵总级数'][0]
        #     # pump_curve calibration
        #     _ = np.vectorize(ESP_well_1.viscosity_calibrate)(SSU, df_pump_info['生产频率'][0]) 
        #     ''' solve (Q, H, EFF) from (pump curve (speed, viscosity, gas effect should be considered) and BHP) '''


        #     ''' pressure profile '''
        #     inputValues = {"well_name": well_name, 
        #                     "Surface_line_pressure": df_well['油压'][j]*1e6, "Pipe_diameter":df_pump_info['油管内径'][0],      
        #                     "Roughness":1e-5,                   "Liquid_viscosity":df_pump_info['粘度'][0]/1000,
        #                     "Liquid_relative_density":0.9456,  "Gas_viscosity":0.000018,   "Gas_relative_density":0.7,
        #                         "Water_cut": df_well['含水'][j], 
        #                     "Reservoir_C":1.83E-17,     "Reservoir_n":1, 
        #                     "Reservoir_P":1.87e7,               "GLR":df_pump_info['油汽比'][0]*((100-df_well['含水'][j])/100),   
        #                     "Geothermal_gradient":0.03, 
        #                     "Surface_T":df_well['井口温度'][j]+273.15, "Pump_intake_T": 380, "P_in": 3.85, "P_out":13.53, 
        #                     "ESP_Depth": df_pump_info['泵挂'][0], "ESP_SN":df_pump_info['泵总级数'][0], 
        #                     "ESP_N":df_pump_info['生产频率'][0]/60*3600, "ESP_length": 10, 'SSU':SSU}
                            
        #     '''pressure profile'''
        #     ESP_Depth = inputValues['ESP_Depth']
        #     VisL_target = inputValues['Liquid_viscosity']*1000

        #     ESP_input['SN']=inputValues['ESP_SN']
        #     ESP_input['N']=inputValues['ESP_N']
        #     ''' Q, H predictioon'''
        #     Q, H, EFF = ESP_well_1.solve_performance_from_BHP(field_BHP)
            
        #     # ESP_curve_plot (ESP_input, inputValues, QBEM, VISL_list=[1,10,50,100,300,500,700,1000], ESP_empirical_class = ESP_well_1)



        #     '''Pressure profile'''
        #     if Q<=0: Q=1
        #     Qg_res = Q*GLR/24/3600
        #     Qo_res = Q*(1-WC/100)/24/3600
        #     Qw_res = Q*WC/100/24/3600
        #     P, df, P_ESP_in, P_ESP_out = P_bottom_Cal (inputValues, df_well_profile, Qg_res, Qo_res, Qw_res, QBEM, ESP_input, 
        #                 ESP_empirical_H=H*m_to_pa*df_pump_info['泵总级数'][0])
        #     Liquid_level = (P-df_well['套压'][j]*1e6)/m_to_pa
        #     Q_list.append(Q*HL)
        #     Pin_list.append(P_ESP_in/1e6)
        #     Pout_list.append(P_ESP_out/1e6)
        #     Liquid_level_list.append(Liquid_level)
        #     print('日期：',df_well['日期'][j], ' 预测排量：',Q,  '基准排量：',df_pump_info['日液'][0])

        #     # P_profile_plot (df_well_profile, df, Table_name='Sheng_li_pump', well_name, inputValues['P_in'], inputValues['P_out'])

        # df_well.insert(df_well.shape[1], "预测排量", Q_list)
        # df_well.insert(df_well.shape[1], "预测泵入口压力", Pin_list)
        # df_well.insert(df_well.shape[1], "预测泵出口压力", Pout_list)
        # df_well.insert(df_well.shape[1], "预测动液位", Liquid_level_list)

        # df_well.to_excel(path_abs+'\\result\\'+'井'+str(i)+'.xlsx')
    plt.show()



