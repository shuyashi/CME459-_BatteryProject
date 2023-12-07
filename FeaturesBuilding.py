import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
from scipy import io
from scipy.interpolate import CubicSpline
from scipy.stats import skew, kurtosis
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression as LR

# Load the preprocesses battery cycle data file
bat_dict = pickle.load(open(r"battery_cycle_data.pkl", "rb"))
print(bat_dict.keys())  # check the blocks (battery names) within the file
print(bat_dict['b1c0']['summary'].keys())  # check the variables under summary section
# plot for one single battery, the curve of temperature over time, the area under the curve is the integration
plt.plot(bat_dict['b1c0']['cycles']['2']['t'], bat_dict['b1c0']['cycles']['2']['T'])
plt.show()

# Plot the discharge capacity (Ah) vs cycle number curves for 124 different batteries
for i in bat_dict.keys():
    plt.plot(bat_dict[i]['summary']['cycle'], bat_dict[i]['summary']['QD'])
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')
plt.xlim([0,100])
plt.ylim([1.0,1.1])
plt.show()

# plot of one battery at cycle 100, V vs. Qd
plt.plot(bat_dict['b2c0']['cycles']['100']['V'], bat_dict['b2c0']['cycles']['100']['Qd'])
plt.xlabel('Discharge Capacity (Ah)')
plt.ylabel('Voltage')
plt.ylim([2.0, 3.6])
plt.ylim([0.0, 1.0])
plt.title('Cycle 100')
plt.show()

# plot of one sample battery at cycle 100, V vs. Qc
plt.plot(bat_dict['b2c0']['cycles']['100']['V'], bat_dict['b2c0']['cycles']['100']['Qd'])
plt.xlabel('Charge Capacity (Ah)')
plt.ylabel('Voltage')
plt.ylim([2.0, 3.6])
plt.ylim([0.0, 1.0])
plt.title('Cycle 100')
plt.show()

print(bat_dict['b2c0']['summary']['QD'][99])
print(bat_dict['b2c0'].keys())
print(bat_dict['b2c0']['summary'].keys())
print(bat_dict['b2c0']['cycle_life'])

# Function definitions required for interpolation of delta Q values
def get_sorted_data(np_array_v, np_array_Qd):
    v_max = np.argmax(np_array_v)
    v_min = np.argmin(np_array_v)
    v_mod = np_array_v[v_max:v_min]
    qmod = np_array_Qd[v_max:v_min]
    vidx = np.argsort(v_mod)
    return v_mod[vidx], qmod[vidx]

def get_interpolated_Qd(Vlin, Vmod, Qdmod):
    q_inter = []
    for vlin in Vlin:
        equals = np.argwhere(Vmod == vlin)
        if not len(equals) == 0:
            q_inter.append(Qdmod[equals][0][0])
            continue
        next_high = np.argwhere(Vmod > vlin)[0]
        x2 = Vmod[next_high]
        y2 = Qdmod[next_high]
        prev_low = np.argwhere(Vmod < vlin)[-1]
        x1 = Vmod[prev_low]
        y1 = Qdmod[prev_low]
        y = y1 + ((vlin - x1) * (y2 - y1) / (x2 - x1))
        q_inter.append(y[0])
    return q_inter

x_vec = []
y_vec = []
for tag in bat_dict.keys():
    battery_cycle = bat_dict[tag]
    V100_mod_asc, Qd100_mod_asc = get_sorted_data(battery_cycle['cycles']['100']['V'],
                                                  battery_cycle['cycles']['100']['Qd'])
    V10_mod_asc, Qd10_mod_asc = get_sorted_data(battery_cycle['cycles']['10']['V'], battery_cycle['cycles']['10']['Qd'])
    mins = np.max([np.min(V100_mod_asc), np.min(V10_mod_asc)])
    maxs = np.min([np.max(V100_mod_asc), np.max(V10_mod_asc)])
    V_lin = np.linspace(mins, maxs, num=1000)
    V_lin_mod = V_lin
    q100_inter = np.asarray(get_interpolated_Qd(V_lin_mod, V100_mod_asc, Qd100_mod_asc))
    q10_inter = np.asarray(get_interpolated_Qd(V_lin_mod, V10_mod_asc, Qd10_mod_asc))
    diffqd = np.asarray(q100_inter - q10_inter)

    ## delta Q based features
    logmin_deltaQ = np.log10(np.absolute(np.min(diffqd)))  # feature 1- log(|min(deltaQ)|)
    logmean_deltaQ = np.log10(np.absolute(np.mean(diffqd)))  # feature 2- log(|mean(deltaQ)|)
    logvar_deltaQ = np.log10(np.absolute(np.var(diffqd)))  # feature 3- log(|var(deltaQ)|)
    logskew_deltaQ = np.log10(np.absolute(skew(diffqd)))  # feature 4- log(|skew(deltaQ)|)
    logkurt_deltaQ = np.log10(np.absolute(kurtosis(diffqd)))  # feature 5 - log(|kurtosis(deltaQ)|)
    log2V_deltaQ = np.interp(2, V_lin_mod, diffqd)  # feature 6 - log(|deltaQ(V = 2)|), using linear interpolation of the curve, interpolate at 2

    ## Discharge capacity fade curve based features
    fit_cycles2_100 = LR().fit(battery_cycle['summary']['cycle'][1:100].reshape(-1, 1),
                               battery_cycle['summary']['QD'][1:100].reshape(-1,1))  # [1:100] means from the second value to the 100th.
    fit_cycles91_100 = LR().fit(battery_cycle['summary']['cycle'][90:100].reshape(-1, 1),
                                battery_cycle['summary']['QD'][90:100].reshape(-1,1))  # python will read the last index as [index-1]
    LR_slope_cycles2_100 = fit_cycles2_100.coef_[0][0]  # feature 7 - slope of linear fit to capacity fade curve - cycles 2-100
    LR_interc_cycles2_100 = fit_cycles2_100.intercept_[0]  # feature 8 - intercept of linear fit to capacity fade curve - cycles 2-100
    LR_slope_cycles91_100 = fit_cycles91_100.coef_[0][0]  # feature 9 - slope of linear fit to capacity fade curve - cycles 91-100
    LR_interc_cycles91_100 = fit_cycles2_100.intercept_[0]  # feature 10 - intercept of linear fit to capacity fade curve - cycles 91-100
    Qd_at_cycle2 = battery_cycle['summary']['QD'][1]  # feature 11 - QD capacity at cycle 2 (discharge value)
    maxQdminusQdatcycle2 = np.max(battery_cycle['summary']['QD']) - Qd_at_cycle2  # feature 12 Qdmax - Qd at cycle 2 (difference b/w max Qd and cycle 2)
    Qd_at_cycle100 = battery_cycle['summary']['QD'][99]  # feature 13 - Qd capacity at cycle 100

    ## Other features
    avg_chargetime_first5cycles = np.mean(battery_cycle['summary']['chargetime'][0:5])  # feature 14 - Avg chargetime (1-5)
    MaxTemp_cycles2_100 = np.max(battery_cycle['summary']['Tmax'][1:100])  # feature 15 - Maximum Temperature, cycles 2 to 100: [1:100]
    MinTemp_cycles2_100 = np.min(battery_cycle['summary']['Tmin'][1:100])  # feature 16 - Minimum Temperature, cycles 2 to 100: [1:100]
    Intg_T_t_cycles2_100 = np.trapz(battery_cycle['summary']['Tavg'][1:100])  # feature 17 - integral of temperature over time, cycles 2 to 100, the time means the cycle numbers
    intresis_2 = battery_cycle['summary']['IR'][1]  # feature 18 - Internal resistance at cycle 2
    MinIntResis_cycles2_100 = np.min(battery_cycle['summary']['IR'][1:100])  # feature 19 - minimum internal resistance cycles 2 to 100
    ir_100minus2 = battery_cycle['summary']['IR'][99] - battery_cycle['summary']['IR'][1]  # feature 20 - IR 100- IR 2

    ## Multivariate discharge curve features, for testing models, 7 features in total (12 features accounts of the duplicated)
    ## To do the linear fit, instead of using LR(), can also use the theta:
    # x_temp = np.concatenate((np.ones((10,1)),np.arange(290,300,1).reshape(10,1)),axis=1)
    # y_temp = battery_cycle['summary']['QD'][1:11].reshape(10, 1)
    # theta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_temp),x_temp)),np.transpose(x_temp)),y_temp)
    fit_cycles2_300 = LR().fit(battery_cycle['summary']['cycle'][1:300].reshape(-1, 1),
                                battery_cycle['summary']['QD'][1:300].reshape(-1, 1))
    fit_cycles291_300 = LR().fit(battery_cycle['summary']['cycle'][290:300].reshape(-1, 1),
                                 battery_cycle['summary']['QD'][290:300].reshape(-1, 1))
    LR_slope_cycles2_100 = fit_cycles2_100.coef_[0][0]  # feature 21 - slope of linear fit to capacity fade curve - cycles 2-100
    LR_slope_cycles2_300 = fit_cycles2_300.coef_[0][0] # feature 21 - slope of linear fit of capacity fade curve, cycles 2 to 300
    LR_interc_cycles2_100 = fit_cycles2_100.intercept_[0]  # feature 22 - intercept of linear fit to capacity fade curve - cycles 2-100
    LR_interc_cycles2_300 = fit_cycles2_300.intercept_[0] # feature 22 - inercept of the linear fit to capacity fade curve, cycles 2-300
    LR_slope_cycles91_100 = fit_cycles91_100.coef_[0][0]  # feature 23 - slope of linear fit to capacity fade curve - cycles 91-100
    LR_slope_cycles291_300 = fit_cycles291_300.coef_[0][0] # feature 23 - slope of linear fit to capacity fade curve - cycles 291-300
    LR_interc_cycles91_100 = fit_cycles91_100.intercept_[0]  # feature 24 - intercept of linear fit to capacity fade curve - cycles 91-100
    LR_interc_cycles291_300 = fit_cycles291_300.intercept_[0]  # feature 24 - intercept of linear fit to capacity fade curve - cycles 291-300
    Qd_at_cycle2 = battery_cycle['summary']['QD'][1]  # feature 25 - QD capacity at cycle 2 (discharge value)
    maxQdminusQdatcycle2 = np.max(battery_cycle['summary']['QD']) - Qd_at_cycle2  # feature 26 Qdmax - Qd at cycle 2 (difference b/w max Qd and cycle 2)
    Qd_at_cycle100 = battery_cycle['summary']['QD'][99]  # feature 27 - Qd capacity at cycle 100
    Qd_at_cycle300 = battery_cycle['summary']['QD'][299]  # feature 27 - Qd capacity at cycle 300

    ## Saving all features into x_vec as the matrix, and y_vec as vector contains cycle_life for modelling
    # x_vec.append([logmin_deltaQ, logmean_deltaQ, logvar_deltaQ, logskew_deltaQ, logkurt_deltaQ, log2V_deltaQ]) # features for set 1
    # x_vec.append([logmin_deltaQ, logmean_deltaQ, logvar_deltaQ, logskew_deltaQ, logkurt_deltaQ, log2V_deltaQ, LR_slope_cycles2_100, LR_interc_cycles2_100, LR_slope_cycles91_100, LR_interc_cycles91_100, Qd_at_cycle2, maxQdminusQdatcycle2, Qd_at_cycle100]) # for feature_set2
    # x_vec.append([logmin_deltaQ, logmean_deltaQ, logvar_deltaQ, logskew_deltaQ, logkurt_deltaQ, log2V_deltaQ, LR_slope_cycles2_100, LR_interc_cycles2_100, LR_slope_cycles91_100, LR_interc_cycles91_100, Qd_at_cycle2, maxQdminusQdatcycle2, Qd_at_cycle100, avg_chargetime_first5cycles, MaxTemp_cycles2_100, MinTemp_cycles2_100, Intg_T_t_cycles2_100, intresis_2, MinIntResis_cycles2_100, ir_100minus2]) # for feature_set3
    x_vec.append([LR_slope_cycles2_100, LR_slope_cycles2_300, LR_interc_cycles2_100, LR_interc_cycles2_300, LR_slope_cycles91_100, LR_slope_cycles291_300, LR_interc_cycles91_100, LR_interc_cycles291_300, Qd_at_cycle2, maxQdminusQdatcycle2, Qd_at_cycle100, Qd_at_cycle300]) # features for the secondary test use
    y_vec.append(bat_dict[tag]['cycle_life'])
    plt.plot(diffqd, V_lin_mod)
plt.xlabel('Qd_100-Qd_10')
plt.ylabel('Voltage V')
plt.show()

## using the first two points for V_lin_mod vs. diffQd curve, to get the delta Q at 2V
print(V_lin_mod)
print(diffqd)

x_vec1 = np.asarray(x_vec)
y_vec1 = np.asarray(y_vec)[:,0,:]
savedic = {'x_sec_matrix': x_vec1, 'y_sec_vector': y_vec1}
io.savemat('SecTest_FeatureSet.mat', savedic)
feature_number = 12
plt.scatter(x_vec1[:,feature_number-1],y_vec1)
plt.show()