"""
A class to implement Simultaneous Perturbation Stochastic Approximation.
"""
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from pandas.core import api
from ESP_simple_all_in_one import *
from Pipe_model import *
# from ESP_Basic_Model import *
# from ESP_Class import *

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

base_pump = 'Flex31'
pump_name = 'Flex31'
dpi = 300
m3s_to_bpd = 543439.65056533
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

QBEM_default = {'TE2700': 4500, 'DN1750': 3000, 'GC6100': 8543.69, 'P100': 11000, 'Flex31': 5000, 'Other': 5000}
sgl_model = 'zhu_2018'

ESP_default = {'TE2700':
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
                "QL": 2700,         "QG": 500,           "GVF": 10,          "WC": 0.,           "SN": 1,
                "P": 150,           "T":100,            "SGL":'zhu_2018', "GL":'None',        "Tune_type":'complex'
            },
    'GC6100':
        {
                "R1": 0.027746,     "R2": 0.050013,     "TB": 0.0019862,    "TV": 0.002894,     "RD1": 0.0547,
                "RD2": 0.017517,    "YI1": 0.017399,    "YI2": 0.013716,    "VOI": 1.512E-5,    "VOD": 1.9818E-5,
                "AIW":0.00287127,   "ADW":0.0048477,
                "ASF": 8.9654E-4,   "ASB": 9.4143E-4,   "AB": 1.0333E-3,    "AV": 1.769E-3,     "ADF": 2.0486E-3,
                "ADB": 1.0301E-3,   # ASF to ADB not necessary
                "LI": 0.0529,       "LD": 0.0839,       "RLK": 4.35E-2,   "LG": 0.0015475,
                "SL": 3.81E-4,      "EA": 0.0003,       "ZI": 7,            "ZD": 8,            "B1": 33.375,
                "B2": 41.387,       "NS": 3220,         "DENL": 1000,       "DENG": 11.2,       "DENW": 1000,       "VISL": 0.001,
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3600,          "SGM": 0.3,
                "QL": 6100,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 1,
                "P": 250,           "T":100,            "GL":'None',        "Tune_type":'complex'
        },
        # original R1 = 0.014351   TB = 0.0025896   "RLK": 6.1237E-2, 
        # new "RLK": 4.35E-2


    'Flex31':
            {
                "R1": 0.018,        "R2": 0.039403,     "TB": 0.0018875,    "TV": 0.0030065,    "RD1": 0.042062,
                "RD2": 0.018841,    "YI1": 0.015341,    "YI2": 0.01046,     "VOI": 0.000010506, "VOD": 0.000010108,
                "AIW":0.0020663,   "ADW":0.0025879,
                "ASF": 6.888e-4,    "ASB": 6.888e-4,    "AB": 6.888e-4,     "AV": 8.626e-4,     "ADF": 8.626e-4, 
                "ADB": 8.626e-4,    # ASF to ADB not necessary
                "LI": 0.04252,      "LD": 0.060315,     "RLK": 0.033179,    "LG": 0.005, 
                "SL": 0.000254,     "EA": 0.0003,       "ZI": 6,            "ZD": 8,            "B1": 21.99,        
                "B2": 57.03,        "NS": 2975,         "DENL": 1000,       "DENG": 11.2,       "DENW": 1000,       "VISL": 0.001,      
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3600,          "SGM": 0.3,
                "QL": 4000,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 1,
                "P": 160,           "T":100,            "GL":'None',        "Tune_type":'complex'
            },
        # ori   "ASF": 6.888e-4,           "ASB": 6.888e-4,           "AB": 0.0020663,    "AV": 0.0025879,    "ADF": 0, 

    'DN1750':
        {
                "R1": 1.9875E-2,    "R2": 3.5599E-2,    "TB": 1.7E-3,       "TV": 3.12E-3,      "RD1": 0.04,
                "RD2": 0.01674,     "YI1": 1.3536E-2,   "YI2": 7.13E-3,     "VOI": 6.283E-6,    "VOD": 7.063E-6,
                "AIW":0.00203005,   "ADW":0.00243763,
                "ASF": 6.8159E-04,  "ASB": 6.549E-04,   "AB": 6.9356E-04,   "AV": 7.1277E-04,   "ADF": 1.0605E-03,
                "ADB": 6.6436E-04,  # ASF to ADB not necessary
                "LI": 0.039,        "LD": 5.185E-02,    "RLK": 0.04,        "LG": 0.01,
                "SL": 0.00005,      "EA": 0.000254,     "ZI": 6,            "ZD": 8,            "B1": 20.3,
                "B2": 36.2,         "NS": 2815,         "DENL": 1000,       "DENG": 11.2,       "DENW": 1000,"VISL": 0.001,
                "VISG": 0.000018,   "VISW": 0.001,      "ST": 0.073,        "N": 3500,          "SGM": 0.3,
                "QL": 1750,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 3,
                "P": 150,           "T":100,            "GL":'None',        "Tune_type":'complex'
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
                "QL": 9000,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 1,
                "P": 150,           "T":100,            "GL":'None',        "Tune_type":'complex'
        },

    'Auto Matching ESP':
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
                "QL": 9000,         "QG": 50,           "GVF": 10,          "WC": 0.,           "SN": 1,
                "P": 150,           "T":100,            "GL":'None',        "Tune_type":'complex'
        }
        
    }

class SimpleSPSA ( object ):
    """Simultaneous Perturbation Stochastic Approximation. 
    """
    # These constants are used throughout
    alpha = 0.602
    gamma = 0.101
    def __init__ ( self, loss_function, a_par = 1e-6, noise_var=0.01, args=(), \
            min_vals=None, max_vals=None, param_tolerance=None, \
            function_tolerance=None, max_iter=5000000 ):
        self.args = args
        self.loss = loss_function
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.param_tolerance = param_tolerance
        self.function_tolerance = function_tolerance
        self.c_par = noise_var
        self.max_iter = max_iter
        self.big_a_par = self.max_iter/10.
        self.a_par = a_par
    def calc_loss ( self, theta ):
        retval = self.loss ( theta, *(self.args ) )
        return retval
    def minimise ( self, theta_0, ens_size=2, report=500 ):
        n_iter = 0
        num_p = theta_0.shape[0]
        print ("Starting theta=", theta_0)
        theta = theta_0
        theta1 = theta
        theta2 = theta
        theta3 = theta
        j_old = self.calc_loss ( theta )
        j1 = j_old
        j2 = j_old
        j3 = j_old
        i1 = 0
        i2 = 0
        i3 = 0
        # Calculate the initial cost function
        theta_saved = theta_0*100
        j_list = []
        j_list.append(j_old)
        while  (np.linalg.norm(theta_saved-theta)/np.linalg.norm(theta_saved) >\
                1e-8) and (n_iter < self.max_iter):
            # The optimisation carried out until the solution has converged, or
            # the maximum number of itertions has been reached.
            theta_saved = theta # Store theta at the start of the iteration
                                # as we may well be restoring it later on.
            # Calculate the ak and ck scalars. Note that these require
            # a degree of tweaking
            ak = self.a_par/( n_iter + 1 + self.big_a_par)**self.alpha
            ck = self.c_par/( n_iter + 1 )**self.gamma  
            ghat = 0.  # Initialise gradient estimate
            for j in np.arange ( ens_size ):
                # This loop produces ``ens_size`` realisations of the gradient
                # which will be averaged. Each has a cost of two function runs.
                # Bernoulli distribution with p=0.5
                delta = (np.random.randint(0, 2, num_p) * 2 - 1)
                # Stochastic perturbation, innit
                theta_plus = theta + ck*delta
                theta_plus = np.minimum ( theta_plus, self.max_vals )
                theta_minus = theta - ck*delta
                theta_minus = np.maximum ( theta_minus, self.min_vals )
                # Funcion values associated with ``theta_plus`` and 
                #``theta_minus``
                j_plus = self.calc_loss ( theta_plus )
                j_minus = self.calc_loss ( theta_minus )
                # Estimate the gradient
                ghat = ghat + ( j_plus - j_minus)/(2.*ck*delta)
            # Average gradient...
            ghat = ghat/float(ens_size)
            # The new parameter is the old parameter plus a scaled displacement
            # along the gradient.
            not_all_pass = True
            this_ak = ( theta*0 + 1 )*ak
            theta_new = theta
            while not_all_pass:
                out_of_bounds = np.where ( np.logical_or ( \
                    theta_new - this_ak*ghat > self.max_vals, 
                    theta_new - this_ak*ghat < self.min_vals ) )[0]
                theta_new = theta - this_ak*ghat
                if len ( out_of_bounds ) == 0:
                    theta = theta - this_ak*ghat
                    not_all_pass = False
                else:
                    this_ak[out_of_bounds] = this_ak[out_of_bounds]/2.
            
            # The new value of the gradient.
            j_new = self.calc_loss ( theta )
            j_sort = j_list.copy()
            j_sort.sort()
            if j_new<j_sort[0]:
                theta3=theta2
                theta2=theta1
                theta1=theta
                j3 = j2
                j2 = j1
                j1 = j_new
                i3 = i2
                i2 = i1 
                i1 = n_iter
            j_list.append(j_new)

            # Be chatty to the user, tell him/her how it's going...
            if n_iter % report == 0:
                print ("\tIter %05d" % n_iter, j_new, ak, ck)
                print ("\tTrained_parameter" , theta1)
            # Functional tolerance: you can specify to ignore new theta values
            # that result in large shifts in the function value. Not a great
            # way to keep the results sane, though, as ak and ck decrease
            # slowly.
            if self.function_tolerance is not None:    
                if np.abs ( j_new - j_old ) > self.function_tolerance:
                    print ("\t No function tolerance!", np.abs ( j_new - j_old ))
                    theta = theta_saved
                    continue
                else:
                    j_old = j_new
            # You can also specify the maximum amount you want your parameters
            # to change in one iteration.
            if self.param_tolerance is not None:
                theta_dif = np.abs ( theta - theta_saved ) 
                if not np.all ( theta_dif < self.param_tolerance ):
                    print ("\t No param tolerance!", theta_dif < \
                        self.param_tolerance)
                    theta = theta_saved
                    continue
            # Ignore results that are outside the boundaries
            if (self.min_vals is not None) and (self.max_vals is not None):      
                i_max = np.where ( theta >= self.max_vals )[0]
                i_min = np.where ( theta <= self.min_vals )[0]
                if len( i_max ) > 0:
                    theta[i_max] = self.max_vals[i_max]*0.9
                if len ( i_min ) > 0:
                    theta[i_min] = self.min_vals[i_min]*1.1
            if report == 1:
                plt.plot ( theta, '-r' )
                plt.title ( "Iter %08d, J=%10.4G" % ( n_iter, j_new ))
                plt.grid ( True )
                plt.savefig ("/tmp/SPSA_%08d.png" % n_iter, dpi=72 )
                plt.close()
            n_iter += 1
        return ( theta, j_new, n_iter, j_list, theta1, theta2, theta3, j1, j2, j3, i1, i2, i3)
def test_spsa ( p_in, noise_var ):
    fitfunc = lambda p, x: p[0]*x*x + p[1]*x + p[2]
    errfunc = lambda p, x, y, noise_var: np.sum ( (fitfunc( p, x ) - y)**2/ \
        noise_var**2 )
    x_arr = np.arange(100) * 0.3
    obs = p_in[0] * x_arr**2 + p_in[1] * x_arr + p_in[2]
    np.random.seed(76523654)
    noise = np.random.normal(size=100) * noise_var  # add some noise to the obs
    obs += noise
    opti = SimpleSPSA ( errfunc, args=( x_arr, obs, noise_var), \
        noise_var=noise_var, min_vals=np.ones(3)*(-5), max_vals = np.ones(3)*5 )
    theta0 = np.random.rand(3)
    ( xsol, j_opt, niter ) = opti.minimise (theta0 )
    print (xsol, j_opt, niter)
def ESP_geometry(base_pump = 'GC6100', Train_parameter=[0.8,0.8,0.8,5000]):
    '''From base pump'''
    ESP_input = ESP_default[base_pump]
    if Train_parameter.shape[0] == 1:
        QBEM = QBEM_default[base_pump]*Train_parameter[0]
        ESP = ESP_input.copy()
    elif Train_parameter.shape[0] == 2:
        QBEM = QBEM_default[base_pump]*Train_parameter[0]
        ESP = ESP_input.copy()
        ESP['R2'] = ESP_input['R2']*Train_parameter[1]
    elif Train_parameter.shape[0] < 5:
        QBEM = QBEM_default[base_pump]*Train_parameter[0]
        coeff_D=Train_parameter[1]
        coeff_L= Train_parameter[2]
        coeff_B = Train_parameter[3]
        coeff_ZI = 1
        coeff_ZD = 1
        ESP_input = ESP_default[base_pump]
        ESP = ESP_input.copy()
        ESP['R1'] = ESP_input['R1']*coeff_D
        ESP['R2'] = ESP_input['R2']*coeff_D
        ESP['TB'] = ESP_input['TB']*coeff_D
        ESP['TV'] = ESP_input['TV']*coeff_D
        ESP['RD1'] = ESP_input['RD1']*coeff_D
        ESP['RD2'] = ESP_input['RD2']*coeff_D
        ESP['B2'] = ESP_input['B2']*coeff_B
        ESP['ZI'] = ESP_input['ZI']*coeff_ZI
        ESP['ZD'] = ESP_input['ZD']*coeff_ZD
        ESP['YI1'] = ESP_input['YI1']*coeff_D
        ESP['YI2'] = ESP_input['YI2']*coeff_D
        ESP['LI'] = ESP_input['LI']*coeff_D
        ESP['LD'] = ESP_input['LD']*coeff_D
        ESP['VOI'] = ESP_input['VOI']*coeff_L*coeff_D
        ESP['VOD'] = ESP_input['VOD']*coeff_L*coeff_D
        ESP['ASF'] = ESP_input['ASF']*coeff_D
        ESP['ASB'] = ESP_input['ASB']*coeff_D
        ESP['AB'] = ESP_input['AB']*coeff_D
        ESP['AV'] = ESP_input['AV']*coeff_D
        ESP['ADF'] = ESP_input['ADF']*coeff_D
        ESP['ADB'] = ESP_input['ADB']*coeff_D
        ESP['RLK'] = ESP_input['RLK']*coeff_D
        ESP['LG'] = ESP_input['LG']*coeff_D
        ESP['SL'] = ESP_input['SL']*coeff_D
        ESP['B1'] = ESP_input['B1']*coeff_B
        ESP['NS'] = ESP_input['NS']
    else :
        # QBEM = QBEM_default[base_pump]*Train_parameter[0]
        # ESP = ESP_input.copy()
        # coeff_L= Train_parameter[1]
        # ESP['R1'] = ESP_input['R1']*Train_parameter[2]
        # ESP['R2'] = ESP_input['R2']*Train_parameter[3]
        # ESP['B1'] = ESP_input['B1']*Train_parameter[4]
        # ESP['B2'] = ESP_input['B2']*Train_parameter[5]
        # coeff_D=(Train_parameter[2]+Train_parameter[3])/2
        # coeff_B = (Train_parameter[4]+Train_parameter[5])/2
        # ESP['TB'] = ESP_input['TB']*coeff_D
        # ESP['ZI'] = ESP_input['ZI']*Train_parameter[6]
        # ESP['ZD'] = ESP_input['ZD']*Train_parameter[7]
        # ESP['YI1'] = ESP_input['YI1']*coeff_D*coeff_L
        # ESP['YI2'] = ESP_input['YI2']*coeff_D*coeff_L
        # ESP['LI'] = ESP_input['LI']*coeff_D
        # ESP['LD'] = ESP_input['LD']*coeff_D
        # ESP['VOI'] = ESP_input['VOI']*coeff_D*coeff_D*coeff_L*Train_parameter[8]
        # ESP['VOD'] = ESP_input['VOD']*coeff_D*coeff_D*coeff_L*Train_parameter[9]
        # ESP['AIW'] = ESP_input['AIW']*coeff_D*coeff_L*Train_parameter[10]
        # ESP['ADW'] = ESP_input['ADW']*coeff_D*coeff_L*Train_parameter[11]
        # ESP['RLK'] = ESP_input['RLK']*coeff_D
        # ESP['LG'] = ESP_input['LG']*coeff_D
        # ESP['SL'] = ESP_input['SL']*coeff_D
        # ESP['NS'] = ESP_input['NS']


        QBEM = QBEM_default[base_pump]*Train_parameter[0]
        ESP = ESP_input.copy()
        ESP['R1'] = ESP_input['R1']*Train_parameter[1]
        ESP['R2'] = ESP_input['R2']*Train_parameter[2]
        ESP['TB'] = ESP_input['TB']*Train_parameter[3]
        ESP['TV'] = ESP_input['TV']*Train_parameter[4]
        ESP['RD1'] = ESP_input['RD1']*Train_parameter[5]
        ESP['RD2'] = ESP_input['RD2']*Train_parameter[6]
        ESP['B2'] = ESP_input['B2']*Train_parameter[7]
        ESP['ZI'] = ESP_input['ZI']*Train_parameter[8]
        ESP['ZD'] = ESP_input['ZD']*Train_parameter[9]
        ESP['YI1'] = ESP_input['YI1']*Train_parameter[10]
        ESP['YI2'] = ESP_input['YI2']*Train_parameter[11]
        ESP['LI'] = ESP_input['LI']*Train_parameter[12]
        ESP['LD'] = ESP_input['LD']*Train_parameter[13]
        ESP['VOI'] = ESP_input['VOI']*Train_parameter[14]
        ESP['VOD'] = ESP_input['VOD']*Train_parameter[15]
        ESP['AIW'] = ESP_input['AIW']*Train_parameter[16]
        ESP['ADW'] = ESP_input['ADW']*Train_parameter[17]
        ESP['RLK'] = ESP_input['RLK']*Train_parameter[18]
        ESP['LG'] = ESP_input['LG']*Train_parameter[19]
        ESP['SL'] = ESP_input['SL']*Train_parameter[20]
        ESP['B1'] = ESP_input['B1']*Train_parameter[21]
        ESP['NS'] = ESP_input['NS']
    return ESP, QBEM
def ESP_fit_test(Train_parameter, Input):
    ESP, QBEM = ESP_geometry(base_pump = base_pump, Train_parameter=Train_parameter)
    HP = ESP_head (QBEM=QBEM, ESP_input=ESP, QL=Input.QL, VISO_in=Input.VISO, DENO_in=Input.DENO, 
            VISG_in=Input.VISG, WC=Input.WC, GLR=Input.GLR, P=Input.P, T=Input.TT, O_W_ST=Input.STOW, GVF = None)
    return HP
def ESP_fit_loss(Train_parameter, Input, HP_target, noise_var):
    loss = np.sum((ESP_fit_test(Train_parameter, Input)-HP_target)**2/noise_var**2)
    return loss  
def SPSA_match(Train_parameter, Input, Target_HP, noise_var, a_par, min_vals, max_vals, max_iter, report):
    # Pump curve before training
    global pump_name, ESP_default
    try:
        ESP_default['Flex31']['N'] = Input.RPM.mean()
    except:
        pass
    opti = SimpleSPSA ( ESP_fit_loss, a_par = a_par, args=(Input, Target_HP, noise_var), \
        noise_var=noise_var, min_vals=min_vals, max_vals = max_vals,max_iter=max_iter  )
    ( Train_parameter, j_opt, niter, J_list, Train_parameter1, Train_parameter2, Train_parameter3, j1, j2, j3, i1, i2, i3 ) = opti.minimise (Train_parameter, report=report)
    print (Train_parameter, j_opt, niter)
    return Train_parameter, Train_parameter1, Train_parameter2, Train_parameter3
def ESP_head (QBEM=6000, ESP_input=ESP_default['Flex31'], QL=np.arange(0.001, 1.1, 0.002) *5000, VISW_in=0.001, DENW_in=1000, VISO_in = 0.5, DENO_in = 950, VISG_in = 0.000018, WC=0.8, GLR=5, P = 350, T=288, O_W_ST = 0.035, GVF = None):
    VISL = VISO_in
    VISW = VISW_in
    VISG = VISG_in
    DENL = DENO_in
    DENW = DENW_in
    DENG_std = gas_density(0.7, 288, 101325)
    try:
        DENG = gas_density(0.7, T, P)
    except:
        DENG = []
        for i in range(P.shape[0]):
            DENG.append(gas_density(0.7, T.iloc[i], P.iloc[i]))
    N = ESP_input['N'] * np.ones(QL.shape) # * np.ones(QL.shape)
    NS = ESP_input['NS'] * np.ones(QL.shape) # * np.ones(QL.shape)
    SGM = ESP_input['SGM'] * np.ones(QL.shape) # * np.ones(QL.shape)
    SN = ESP_input['SN'] * np.ones(QL.shape)
    ST = ESP_input['ST'] * np.ones(QL.shape)
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

def Oil_validation(Train_parameter, pump_name, TargetRPM):
    unity=[]
    for i in range(len(Train_parameter)):
        unity.append(1)

    ESP_real = ESP_default[pump_name]
    QBEM_real = QBEM_default[pump_name]
    ESP, QBEM = ESP_geometry(base_pump = base_pump, Train_parameter=Train_parameter)
    
    bar_plot = []
    for i in range (26):
        bar_plot.append (list(ESP.values())[i]/list(ESP_real.values())[i])

    ESP['N']=TargetRPM

    # test oil curve
    conn, c = connect_db('ESP.db')
    df_data = pd.read_sql_query("SELECT *"
                              + "FROM df_Viscosity "
                              + "ORDER BY RPM, TargetVISL_cp, QL_bpd"
                              + ";", conn)
                              
    
    df_data.drop(df_data[df_data['Pump']=='DN1750_Solano'].index, inplace=True)
    df_data.drop(df_data[df_data['Pump']=='DN1750_Banjar'].index, inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data[(df_data.Pump.str.startswith(pump_name)) & (df_data.RPM == TargetRPM)]
    df_data = df_data[df_data.QL_bpd != 0]
    df_data=df_data.reset_index(drop=True)
    disconnect_db(conn)
    VISL = df_data['TargetVISL_cp'].unique()
    DENL = 1000
    fig2, (bx, bx2) = plt.subplots(dpi = dpi, figsize = (6.66,2.5), nrows=1, ncols=2)
    for visl in VISL:
        df_plot = df_data[df_data['TargetVISL_cp'] == visl].copy()
        QL = df_plot.QL_bpd
        QL_cal = np.arange(0.01, 1.1, 0.02) * df_plot.QL_bpd.max() * 1.5            
        HP = ESP_head (QBEM=QBEM, ESP_input=ESP, QL=QL_cal, VISO_in=visl/1000, 
                        DENO_in=df_plot.DENL_kgm3.mean(), VISG_in=0.000018, WC=0, GLR=0, 
                        P=100*psi_to_pa, T=288, O_W_ST=0.035, GVF = None)

        bx.scatter(QL, df_plot.DP_psi,linewidths=0.75, s=8)
        bx.plot(QL_cal, HP, label=str(int(visl))+' cP', linewidth=0.75)
        bx.set_xlim(0, df_data.QL_bpd.max() * 1.2)
        bx.set_ylim(0)
        bx.legend(frameon=False)

    bx.set_xlabel('QL bpd', fontsize=8)
    bx.set_ylabel('Head psi', fontsize=8)
    bx.legend(frameon=False, fontsize=5)
    if pump_name == 'Flex31':
        title='Oil performance MTESP at '+str(TargetRPM)+' RPM'
    else:
        title='Oil performance '+pump_name+' at '+str(TargetRPM)+ ' RPM'
    bx.set_title(title, fontsize=8)
    bx.xaxis.set_tick_params(labelsize=8)
    bx.yaxis.set_tick_params(labelsize=8)

    bx2.bar(list(ESP)[:26],bar_plot)
    bx2.set_xticklabels(list(ESP)[:26], rotation=300)
    bx2.set_ylim(0,2)
    bx2.xaxis.set_tick_params(labelsize=5)
    bx2.yaxis.set_tick_params(labelsize=8)
    bx2.set_title('Geometry Fitted('+pump_name+')/Based('+base_pump+')'+' (-)', fontsize=8)

    fig2.tight_layout()
    # fig2.savefig('/Users/haiwenzhu/Desktop/work/006 code/001 ESP/Python/ESP model py 2021/SPSA/'+title)
    # fig2.savefig('SPSA/'+str(title)+'.jpg')

def field_match_example():

    '''
    以 '22b'井为例
    根据实验条件设置初始参数，挑选文献数据Flex31电潜泵为拟合初始值
    '''
    global well_name, base_pump, pump_name, ESP_default
    well_name = '22b'
    base_pump, pump_name  = 'Flex31', 'Flex31'
    ESP_default[pump_name]['N'] = 2890
    ESP_default[pump_name]['SN'] = 282
    ESP_input=ESP_default['Flex31'].copy()

    #电潜泵出厂测试点，现场测试点
    Q_water = 80.36*543439.65056533/24/3600   # m3d to bpd
    H_water=1651.41*1.4223/ESP_input['SN']      # m of water to psi/stage#
    Q_liquid = 103*543439.65056533/24/3600   # m3d to bpd
    H_liquid = (14.11-3.12)*145.0381/ESP_input['SN']      # Mpa to psi/stage#
    Input = pd.DataFrame({'QL':[Q_water, Q_liquid], 'HP':[H_water, H_liquid], 'VISO':[0.001,0.523], 'DENO':[1000, 952.1],
                            'VISG':[0.000018, 0.000018], 'WC':[100, 75], 'GLR':[0,6.7], 
                            'P':[350*psi_to_pa, (14.11+3.12)*1e6], 'TT':[288, 288], 'STOW':[0.035, 0.035]})

    Input = pd.DataFrame({'QL':[Q_liquid], 'HP':[H_liquid], 'VISO':[0.523], 'DENO':[952.1],
                            'VISG':[0.000018], 'WC':[75], 'GLR':[6.7], 
                            'P':[(14.11+3.12)*1e6], 'TT':[288], 'STOW':[ 0.035]})
    
    noise_var, a_par, max_iter, report=0.1, 1e-5, 20, 10
    Train_parameter = np.array([1, 1])
    min_vals=np.array([0.5,0.5])
    max_vals = np.array([1.5,1.5])

    _, Train_parameter1, _, _ = SPSA_match(Train_parameter, Input, Input.HP, noise_var, a_par, min_vals, max_vals, max_iter, report)
    # Train_parameter1 = Train_parameter
    

    '''validation'''
    ESP_input, QBEM = ESP_geometry(base_pump = base_pump, Train_parameter=Train_parameter1)

    QL= np.arange(0.005, 1.1, 0.025) * 5000.0

    Q_water = 80.36
    H_water=1651.41
    Q_Oil = 103
    H_Oil = (14.11-3.12)*101.97 # mpa to m of water

    ESP_input['SN'] = 282
    VISO_in=0.523
    WC=75
    VISW_in=0.001
    VISG_in=0.000018
    DENO_in=952.1
    DENW_in=1000
    DENG_std = gas_density(0.7, 288, 101325)
    GLR = 6.7
    GOR = GLR/(1-WC/100)
    P = (14.11+3.12)*1e6
    T = 288
    O_W_ST = 0.035
    fig, ax = plt.subplots(dpi=dpi, figsize = (3.33,2.5), nrows=1, ncols=1)

    for VISO_test in [1,10,50,100,300,500]:
    # for VISO_test in [1,200,300,500,800,1000]:
        if VISO_test>VISO_in*1000:
            break
        if VISO_test == 1: 
            VISO_test = 1
            VISO_test = VISO_test/1000
            HP = ESP_head (QBEM, ESP_input, QL, 0.001, 1000, 0.001, 1000, 0.000018, 100, 0, 350*psi_to_pa, T, O_W_ST, None)
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
    ax.scatter([Q_Oil], [H_Oil], color = 'red', label=('现场 油%dcp (%dm$^3$/d,%dm)'%(523,Q_Oil,H_Oil)),  marker='*', linewidth=0.3)
    ax.plot([0, Q_Oil], [H_Oil, H_Oil], color = 'red', linewidth=0.75)
    ax.plot([Q_Oil, Q_Oil], [0, H_Oil], color = 'red', linewidth=0.75)
    ax.legend(frameon=False, fontsize=6)
    QL= np.array(Q_Oil*m3s_to_bpd/24/3600)
    VISG = gas_viscosity(0.7, T, P)
    HP = ESP_head (QBEM, ESP_input, QL, VISW_in, DENW_in, VISO_in, DENO_in, VISG_in, WC, GLR, P, T, O_W_ST, None)
    print (QL/(m3s_to_bpd/24/3600), HP*0.3048/0.433*ESP_input['SN'], GLR)
    fig.tight_layout()

def test_curve_match_example():
    global base_pump, pump_name
    
    try:
    
        '''P100 oil'''
        # match
        base_pump = 'DN1750'
        pump_name = 'Flex31'

        conn, c = connect_db('ESP.db')
        df_data = pd.read_sql_query("SELECT *"
                                + "FROM df_Viscosity "
                                + "ORDER BY RPM, TargetVISL_cp, QL_bpd"
                                + ";", conn)
        df_data = df_data[df_data.Pump == pump_name]
        df_data = df_data[df_data.Case == 'Flex31_CFD']
        df_data = df_data[df_data.RPM == 3600]
        df_data = df_data[df_data.QL_bpd != 0]
        df_data=df_data.reset_index(drop=True)
        disconnect_db(conn)

        Input = pd.DataFrame({'QL':df_data.QL_bpd, 'HP':df_data.DP_psi, 'VISO':df_data.TargetVISL_cp/1000, 'DENO':df_data.DENL_kgm3,
                                'VISG':0.000018, 'WC':df_data['TargetWC_%'], 'GLR':0, 
                                'P':100*psi_to_pa, 'TT':288, 'STOW':0.035})
        Target_HP = df_data.DP_psi
    
    except:
        print("SQL提取数据错误")
        return
    #SPSA参数设置
    try:
        noise_var=0.1
        a_par=1e-8
        max_iter=100
        report=10
        Train_parameter = np.ones(26)
        min_vals=np.ones(26)*0.1
        max_vals = np.ones(26)*10

        _, Train_parameter1, _, _ = SPSA_match(Train_parameter, Input, Target_HP, noise_var, a_par, min_vals, max_vals, max_iter, report)
    except:
        print('SPSA参数错误')
        return

    Oil_validation(Train_parameter1, pump_name, 3600)
    plt.show()

if __name__ == "__main__":
    # field_match_example()
    test_curve_match_example()
    print('计算结束')
    plt.show()
