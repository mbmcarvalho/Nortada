#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#               SÉRIES TEMPORAIS - wrfoutMPI - 2046 a 2065  e 2081-2100
#                    ssp245, ssp370 e ssp585

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

import pandas as pd
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
from scipy.stats import ttest_ind


#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                                   2046-2065
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

file_name_MPI_ssp245_2046_2065 = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\dados_wrfout_mpi_ssp\ficheiros_2046_2065\wrfout_MPI_ssp245_2046_2065_UV10m_12_18h.nc')
file_name_MPI_ssp370_2046_2065 = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\dados_wrfout_mpi_ssp\ficheiros_2046_2065\wrfout_MPI_ssp370_2046_2065_UV10m_12_18h.nc')
file_name_MPI_ssp585_2046_2065 = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\dados_wrfout_mpi_ssp\ficheiros_2046_2065\wrfout_MPI_ssp585_2046_2065_UV10m_12_18h.nc')

# %%
u10_MPI_ssp245_2046_2065 = file_name_MPI_ssp245_2046_2065.variables['U10'][:]
v10_MPI_ssp245_2046_2065 = file_name_MPI_ssp245_2046_2065.variables['V10'][:]
lon_MPI_ssp245_2046_2065 = file_name_MPI_ssp245_2046_2065.variables['XLONG'][:]
lat_MPI_ssp245_2046_2065 = file_name_MPI_ssp245_2046_2065.variables['XLAT'][:]
time_MPI_ssp245_2046_2065 = file_name_MPI_ssp245_2046_2065.variables['XTIME'][:]

u10_MPI_ssp370_2046_2065 = file_name_MPI_ssp370_2046_2065.variables['U10'][:]
v10_MPI_ssp370_2046_2065 = file_name_MPI_ssp370_2046_2065.variables['V10'][:]
lon_MPI_ssp370_2046_2065 = file_name_MPI_ssp370_2046_2065.variables['XLONG'][:]
lat_MPI_ssp370_2046_2065 = file_name_MPI_ssp370_2046_2065.variables['XLAT'][:]
time_MPI_ssp370_2046_2065 = file_name_MPI_ssp370_2046_2065.variables['XTIME'][:]

u10_MPI_ssp585_2046_2065 = file_name_MPI_ssp585_2046_2065.variables['U10'][:]
v10_MPI_ssp585_2046_2065 = file_name_MPI_ssp585_2046_2065.variables['V10'][:]
lon_MPI_ssp585_2046_2065 = file_name_MPI_ssp585_2046_2065.variables['XLONG'][:]
lat_MPI_ssp585_2046_2065 = file_name_MPI_ssp585_2046_2065.variables['XLAT'][:]
time_MPI_ssp585_2046_2065 = file_name_MPI_ssp585_2046_2065.variables['XTIME'][:]


# %%    tempo para data e hora

time_var_MPI_ssp245_2046_2065 = file_name_MPI_ssp245_2046_2065.variables['XTIME']  
time_units_MPI_ssp245_2046_2065 = time_var_MPI_ssp245_2046_2065.units  #hours since 1994-12-8 00:00:00
time_dates_MPI_ssp245_2046_2065 = nc.num2date(time_var_MPI_ssp245_2046_2065[:], units=time_units_MPI_ssp245_2046_2065)
time_dates_py_MPI_ssp245_2046_2065 = [np.datetime64(date) for date in time_dates_MPI_ssp245_2046_2065]
time_index_MPI_ssp245_2046_2065 = pd.DatetimeIndex(time_dates_py_MPI_ssp245_2046_2065) #formato pandas DateTimeIndex

time_var_MPI_ssp370_2046_2065 = file_name_MPI_ssp370_2046_2065.variables['XTIME']  
time_units_MPI_ssp370_2046_2065 = time_var_MPI_ssp370_2046_2065.units  #hours since 1994-12-8 00:00:00
time_dates_MPI_ssp370_2046_2065 = nc.num2date(time_var_MPI_ssp370_2046_2065[:], units=time_units_MPI_ssp370_2046_2065)
time_dates_py_MPI_ssp370_2046_2065 = [np.datetime64(date) for date in time_dates_MPI_ssp370_2046_2065]
time_index_MPI_ssp370_2046_2065 = pd.DatetimeIndex(time_dates_py_MPI_ssp370_2046_2065) #formato pandas DateTimeIndex

time_var_MPI_ssp585_2046_2065 = file_name_MPI_ssp585_2046_2065.variables['XTIME']  
time_units_MPI_ssp585_2046_2065 = time_var_MPI_ssp585_2046_2065.units  #hours since 1994-12-8 00:00:00
time_dates_MPI_ssp585_2046_2065 = nc.num2date(time_var_MPI_ssp585_2046_2065[:], units=time_units_MPI_ssp585_2046_2065)
time_dates_py_MPI_ssp585_2046_2065 = [np.datetime64(date) for date in time_dates_MPI_ssp585_2046_2065]
time_index_MPI_ssp585_2046_2065 = pd.DatetimeIndex(time_dates_py_MPI_ssp585_2046_2065) #formato pandas DateTimeIndex


# %%

# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#          COORDENADAS DO PONTO + PRÓXIMO DA PRAIA DA BARRA

# Coordenadas da Praia da Barra
lon_barra = np.array([-8.7578430, -8.7288208])
lat_barra = np.array([40.6077614, 40.6470909])

lon_flat = lon_MPI_ssp245_2046_2065.flatten() #Achatamento das matrizes lon e lat para vetores (os dados não desaparecem, só ficam unidimensionais)
lat_flat = lat_MPI_ssp245_2046_2065.flatten()
distances = np.sqrt((lon_flat - lon_barra[0])**2 + (lat_flat- lat_barra[0])**2) # Cálculo distância euclidiana para cada ponto
idx_nearest_point = np.argmin(distances) #índice do ponto mais próximo de Barra
lon_nearest = lon_flat[idx_nearest_point]
lat_nearest = lat_flat[idx_nearest_point]


# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           CÁLCULO DA INTENSIDADE DO VENTO (sqrt((u^2)+(v^2))


# MPI_ssp245

lon_index_MPI_ssp245_2046_2065, lat_index_MPI_ssp245 = np.unravel_index(idx_nearest_point, lon_MPI_ssp245_2046_2065.shape) # Desfazer o achatamento para obter os índices originais
u10_var_MPI_ssp245_2046_2065 = u10_MPI_ssp245_2046_2065[:, lon_index_MPI_ssp245_2046_2065, lat_index_MPI_ssp245]
v10_var_MPI_ssp245_2046_2065 = v10_MPI_ssp245_2046_2065[:, lon_index_MPI_ssp245_2046_2065, lat_index_MPI_ssp245]

u10_daily_mean_MPI_ssp245_2046_2065 = []
v10_daily_mean_MPI_ssp245_2046_2065 = []
daily_dates_MPI_ssp245_2046_2065 = []
unique_days_MPI_ssp245_2046_2065 = np.unique(time_index_MPI_ssp245_2046_2065.date) # Iterar sobre os dias únicos para calcular as médias diárias
for day_MPI_ssp245_2046_2065 in unique_days_MPI_ssp245_2046_2065:

    day_indices_MPI_ssp245_2046_2065 = np.where(time_index_MPI_ssp245_2046_2065.date == day_MPI_ssp245_2046_2065)[0]  # Seleção dos índices correspondentes ao dia específico
    u10_day_mean_MPI_ssp245_2046_2065 = np.mean(u10_var_MPI_ssp245_2046_2065[day_indices_MPI_ssp245_2046_2065]) #média diária para u10 
    v10_day_mean_MPI_ssp245_2046_2065 = np.mean(v10_var_MPI_ssp245_2046_2065[day_indices_MPI_ssp245_2046_2065]) #média diária para v10
    u10_daily_mean_MPI_ssp245_2046_2065.append(u10_day_mean_MPI_ssp245_2046_2065) # Adicionar à lista de médias diárias
    v10_daily_mean_MPI_ssp245_2046_2065.append(v10_day_mean_MPI_ssp245_2046_2065)
    daily_dates_MPI_ssp245_2046_2065.append(pd.Timestamp(day_MPI_ssp245_2046_2065)) # Adicionar a data correspondente
    
u10_daily_mean_MPI_ssp245_2046_2065 = np.array(u10_daily_mean_MPI_ssp245_2046_2065) # Conversão das listas em arrays numpy para facilitar a manipulação dos dados
v10_daily_mean_MPI_ssp245_2046_2065 = np.array(v10_daily_mean_MPI_ssp245_2046_2065)

# MPI_ssp370

lon_index_MPI_ssp370_2046_2065, lat_index_MPI_ssp370 = np.unravel_index(idx_nearest_point, lon_MPI_ssp370_2046_2065.shape) # Desfazer o achatamento para obter os índices originais
u10_var_MPI_ssp370_2046_2065 = u10_MPI_ssp370_2046_2065[:, lon_index_MPI_ssp370_2046_2065, lat_index_MPI_ssp370]
v10_var_MPI_ssp370_2046_2065 = v10_MPI_ssp370_2046_2065[:, lon_index_MPI_ssp370_2046_2065, lat_index_MPI_ssp370]

u10_daily_mean_MPI_ssp370_2046_2065 = []
v10_daily_mean_MPI_ssp370_2046_2065 = []
daily_dates_MPI_ssp370_2046_2065 = []
unique_days_MPI_ssp370_2046_2065 = np.unique(time_index_MPI_ssp370_2046_2065.date) # Iterar sobre os dias únicos para calcular as médias diárias
for day_MPI_ssp370_2046_2065 in unique_days_MPI_ssp370_2046_2065:

    day_indices_MPI_ssp370_2046_2065 = np.where(time_index_MPI_ssp370_2046_2065.date == day_MPI_ssp370_2046_2065)[0]  # Seleção dos índices correspondentes ao dia específico
    u10_day_mean_MPI_ssp370_2046_2065 = np.mean(u10_var_MPI_ssp370_2046_2065[day_indices_MPI_ssp370_2046_2065]) #média diária para u10 
    v10_day_mean_MPI_ssp370_2046_2065 = np.mean(v10_var_MPI_ssp370_2046_2065[day_indices_MPI_ssp370_2046_2065]) #média diária para v10
    u10_daily_mean_MPI_ssp370_2046_2065.append(u10_day_mean_MPI_ssp370_2046_2065) # Adicionar à lista de médias diárias
    v10_daily_mean_MPI_ssp370_2046_2065.append(v10_day_mean_MPI_ssp370_2046_2065)
    daily_dates_MPI_ssp370_2046_2065.append(pd.Timestamp(day_MPI_ssp370_2046_2065)) # Adicionar a data correspondente
    
u10_daily_mean_MPI_ssp370_2046_2065 = np.array(u10_daily_mean_MPI_ssp370_2046_2065) # Conversão das listas em arrays numpy para facilitar a manipulação dos dados
v10_daily_mean_MPI_ssp370_2046_2065 = np.array(v10_daily_mean_MPI_ssp370_2046_2065)


# MPI_ssp585

lon_index_MPI_ssp585_2046_2065, lat_index_MPI_ssp585 = np.unravel_index(idx_nearest_point, lon_MPI_ssp245_2046_2065.shape) # Desfazer o achatamento para obter os índices originais
u10_var_MPI_ssp585_2046_2065 = u10_MPI_ssp585_2046_2065[:, lon_index_MPI_ssp585_2046_2065, lat_index_MPI_ssp585]
v10_var_MPI_ssp585_2046_2065 = v10_MPI_ssp585_2046_2065[:, lon_index_MPI_ssp585_2046_2065, lat_index_MPI_ssp585]

u10_daily_mean_MPI_ssp585_2046_2065 = []
v10_daily_mean_MPI_ssp585_2046_2065 = []
daily_dates_MPI_ssp585_2046_2065 = []
unique_days_MPI_ssp585_2046_2065 = np.unique(time_index_MPI_ssp585_2046_2065.date) # Iterar sobre os dias únicos para calcular as médias diárias
for day_MPI_ssp585_2046_2065 in unique_days_MPI_ssp585_2046_2065:

    day_indices_MPI_ssp585_2046_2065 = np.where(time_index_MPI_ssp585_2046_2065.date == day_MPI_ssp585_2046_2065)[0]  # Seleção dos índices correspondentes ao dia específico
    u10_day_mean_MPI_ssp585_2046_2065 = np.mean(u10_var_MPI_ssp585_2046_2065[day_indices_MPI_ssp585_2046_2065]) #média diária para u10 
    v10_day_mean_MPI_ssp585_2046_2065 = np.mean(v10_var_MPI_ssp585_2046_2065[day_indices_MPI_ssp585_2046_2065]) #média diária para v10
    u10_daily_mean_MPI_ssp585_2046_2065.append(u10_day_mean_MPI_ssp585_2046_2065) # Adicionar à lista de médias diárias
    v10_daily_mean_MPI_ssp585_2046_2065.append(v10_day_mean_MPI_ssp585_2046_2065)
    daily_dates_MPI_ssp585_2046_2065.append(pd.Timestamp(day_MPI_ssp585_2046_2065)) # Adicionar a data correspondente
    
u10_daily_mean_MPI_ssp585_2046_2065 = np.array(u10_daily_mean_MPI_ssp585_2046_2065) # Conversão das listas em arrays numpy para facilitar a manipulação dos dados
v10_daily_mean_MPI_ssp585_2046_2065 = np.array(v10_daily_mean_MPI_ssp585_2046_2065)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                     CÁLCULO DA VELOCIDADE MÉDIA DIÁRIA

velocity_daily_mean_MPI_ssp245_2046_2065 = np.sqrt(u10_daily_mean_MPI_ssp245_2046_2065**2 + v10_daily_mean_MPI_ssp245_2046_2065**2)
velocity_daily_mean_MPI_ssp370_2046_2065 = np.sqrt(u10_daily_mean_MPI_ssp370_2046_2065**2 + v10_daily_mean_MPI_ssp370_2046_2065**2)
velocity_daily_mean_MPI_ssp585_2046_2065 = np.sqrt(u10_daily_mean_MPI_ssp585_2046_2065**2 + v10_daily_mean_MPI_ssp585_2046_2065**2)

#%%

# wrfout MPI_ssp245
data_MPI_ssp245_2046_2065 = {'Date': daily_dates_MPI_ssp245_2046_2065,'Velocity': velocity_daily_mean_MPI_ssp245_2046_2065} # DataFrame com as datas e as velocidades médias diárias
df_MPI_ssp245_2046_2065 = pd.DataFrame(data_MPI_ssp245_2046_2065)  #tabela
df_MPI_ssp245_2046_2065['Year'] = df_MPI_ssp245_2046_2065['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_MPI_ssp245_2046_2065 = df_MPI_ssp245_2046_2065.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade

# wrfout MPI_ssp370
data_MPI_ssp370_2046_2065 = {'Date': daily_dates_MPI_ssp370_2046_2065,'Velocity': velocity_daily_mean_MPI_ssp370_2046_2065} # DataFrame com as datas e as velocidades médias diárias
df_MPI_ssp370_2046_2065 = pd.DataFrame(data_MPI_ssp370_2046_2065)  #tabela
df_MPI_ssp370_2046_2065['Year'] = df_MPI_ssp370_2046_2065['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_MPI_ssp370_2046_2065 = df_MPI_ssp370_2046_2065.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade

# wrfout MPI_ssp585
data_MPI_ssp585_2046_2065 = {'Date': daily_dates_MPI_ssp585_2046_2065,'Velocity': velocity_daily_mean_MPI_ssp585_2046_2065} # DataFrame com as datas e as velocidades médias diárias
df_MPI_ssp585_2046_2065 = pd.DataFrame(data_MPI_ssp585_2046_2065)  #tabela
df_MPI_ssp585_2046_2065['Year'] = df_MPI_ssp585_2046_2065['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_MPI_ssp585_2046_2065 = df_MPI_ssp585_2046_2065.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                        FUNÇÕES CÁLCULO DIREÇÃO DO VENTO

import numpy as np
from scipy.stats import chi2

def circ_r_degrees(alpha, w=None, d=0, dim=0):
    if w is None:
        w = np.ones_like(alpha)
    else:
        if w.shape != alpha.shape:
            raise ValueError('Input dimensions do not match')

    r = np.sum(w * (np.cos(np.radians(alpha)) + 1j * np.sin(np.radians(alpha))), axis=dim)
    r = np.abs(r) / np.sum(w, axis=dim)

    if d != 0:
        c = d / (2 * np.sin(np.radians(d) / 2))
        r *= c
    
    return r

def circ_confmean_degrees(alpha, xi=0.05, w=None, d=0, dim=0):
    if w is None:
        w = np.ones_like(alpha)
    else:
        if w.shape != alpha.shape:
            raise ValueError('Input dimensions do not match')

    r = circ_r_degrees(alpha, w, d, dim)
    n = np.sum(w, axis=dim)
    R = n * r
    c2 = chi2.ppf(1 - xi, 1)
    
    t = np.zeros_like(r)
    for i in range(r.size):
        if r.flat[i] < 0.9 and r.flat[i] > np.sqrt(c2 / (2 * n.flat[i])):
            t.flat[i] = np.sqrt((2 * n.flat[i] * (2 * R.flat[i]**2 - n.flat[i] * c2)) / (4 * n.flat[i] - c2))
        elif r.flat[i] >= 0.9:
            t.flat[i] = np.sqrt(n.flat[i]**2 - (n.flat[i]**2 - R.flat[i]**2) * np.exp(c2 / n.flat[i]))
        else:
            t.flat[i] = np.nan
    
    t = np.degrees(np.arccos(t / R))
    return t

def circ_mean_degrees(alpha, w=None, dim=0):
    if w is None:
        w = np.ones_like(alpha)
    else:
        if w.shape != alpha.shape:
            raise ValueError('Input dimensions do not match')

    r = np.sum(w * (np.cos(np.radians(alpha)) + 1j * np.sin(np.radians(alpha))), axis=dim)
    mu = np.angle(r)
    mu = np.mod(np.degrees(mu), 360)
    
    t = circ_confmean_degrees(alpha, 0.05, w, dim=dim)
    ul = mu + t
    ll = mu - t
    
    return mu, ul, ll

# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                   CÁLCULO DAS DIREÇÕES DO VENTO

# Exemplo de cálculo das direções diárias para ERA5 e MPI
direction_daily_MPI_ssp245_2046_2065 = np.degrees(np.arctan2(v10_daily_mean_MPI_ssp245_2046_2065, u10_daily_mean_MPI_ssp245_2046_2065))
direction_daily_MPI_ssp370_2046_2065 = np.degrees(np.arctan2(v10_daily_mean_MPI_ssp370_2046_2065, u10_daily_mean_MPI_ssp370_2046_2065))
direction_daily_MPI_ssp585_2046_2065 = np.degrees(np.arctan2(v10_daily_mean_MPI_ssp585_2046_2065, u10_daily_mean_MPI_ssp585_2046_2065))

media_dir_v10_MPI_ssp245_2046_2065 = np.zeros_like(direction_daily_MPI_ssp245_2046_2065)
ul_MPI_ssp245_2046_2065 = np.zeros_like(direction_daily_MPI_ssp245_2046_2065)
ll_MPI_ssp245_2046_2065 = np.zeros_like(direction_daily_MPI_ssp245_2046_2065)

for i in range(direction_daily_MPI_ssp245_2046_2065.shape[0]):
    media_dir_v10_MPI_ssp245_2046_2065[i], ul_MPI_ssp245_2046_2065[i], ll_MPI_ssp245_2046_2065[i] = circ_mean_degrees(direction_daily_MPI_ssp245_2046_2065[i])

media_dir_v10_MPI_ssp370_2046_2065 = np.zeros_like(direction_daily_MPI_ssp370_2046_2065)
ul_MPI_ssp370_2046_2065 = np.zeros_like(direction_daily_MPI_ssp370_2046_2065)
ll_MPI_ssp370_2046_2065 = np.zeros_like(direction_daily_MPI_ssp370_2046_2065)

for i in range(direction_daily_MPI_ssp370_2046_2065.shape[0]):
    media_dir_v10_MPI_ssp370_2046_2065[i], ul_MPI_ssp370_2046_2065[i], ll_MPI_ssp370_2046_2065[i] = circ_mean_degrees(direction_daily_MPI_ssp370_2046_2065[i])

media_dir_v10_MPI_ssp585_2046_2065 = np.zeros_like(direction_daily_MPI_ssp585_2046_2065)
ul_MPI_ssp585_2046_2065 = np.zeros_like(direction_daily_MPI_ssp585_2046_2065)
ll_MPI_ssp585_2046_2065 = np.zeros_like(direction_daily_MPI_ssp585_2046_2065)

for i in range(direction_daily_MPI_ssp585_2046_2065.shape[0]):
    media_dir_v10_MPI_ssp585_2046_2065[i], ul_MPI_ssp585_2046_2065[i], ll_MPI_ssp585_2046_2065[i] = circ_mean_degrees(direction_daily_MPI_ssp585_2046_2065[i])



# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                                  NORTADA

nortada_MPI_ssp245_2046_2065 = (media_dir_v10_MPI_ssp245_2046_2065 >= 335) | (media_dir_v10_MPI_ssp245_2046_2065 <= 25)
nortada_MPI_ssp370_2046_2065 = (media_dir_v10_MPI_ssp370_2046_2065 >= 335) | (media_dir_v10_MPI_ssp370_2046_2065 <= 25)
nortada_MPI_ssp585_2046_2065 = (media_dir_v10_MPI_ssp585_2046_2065 >= 335) | (media_dir_v10_MPI_ssp585_2046_2065 <= 25)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           VELOCIDADE MÉDIA DOS DIAS COM NORTADA

velocity_nortada_MPI_ssp245_2046_2065 = velocity_daily_mean_MPI_ssp245_2046_2065[nortada_MPI_ssp245_2046_2065]
mean_velocity_nortada_MPI_ssp245_2046_2065 = np.mean(velocity_nortada_MPI_ssp245_2046_2065)

velocity_nortada_MPI_ssp370_2046_2065 = velocity_daily_mean_MPI_ssp370_2046_2065[nortada_MPI_ssp370_2046_2065]
mean_velocity_nortada_MPI_ssp370_2046_2065 = np.mean(velocity_nortada_MPI_ssp370_2046_2065)

velocity_nortada_MPI_ssp585_2046_2065= velocity_daily_mean_MPI_ssp585_2046_2065[nortada_MPI_ssp585_2046_2065]
mean_velocity_nortada_MPI_ssp585_2046_2065 = np.mean(velocity_nortada_MPI_ssp585_2046_2065)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   DATA FRAMES: vel_média|dir_média|nortada_diária|data|ano|mês|dia

# wrfout MPI_ssp245
df_filtered_MPI_ssp245_2046_2065 = pd.DataFrame({
    'velocity_daily_mean_MPI_ssp245_2046_2065': velocity_daily_mean_MPI_ssp245_2046_2065,
    'media_dir_v10_MPI_ssp245_2046_2065': media_dir_v10_MPI_ssp245_2046_2065,
    'nortada_MPI_ssp245_2046_2065': nortada_MPI_ssp245_2046_2065,
    'Date': daily_dates_MPI_ssp245_2046_2065
})
df_filtered_MPI_ssp245_2046_2065['ano'] = df_filtered_MPI_ssp245_2046_2065['Date'].dt.year
df_filtered_MPI_ssp245_2046_2065['mês'] = df_filtered_MPI_ssp245_2046_2065['Date'].dt.month
df_filtered_MPI_ssp245_2046_2065['dia'] = df_filtered_MPI_ssp245_2046_2065['Date'].dt.day

# wrfout MPI_ssp370
df_filtered_MPI_ssp370_2046_2065 = pd.DataFrame({
    'velocity_daily_mean_MPI_ssp370_2046_2065': velocity_daily_mean_MPI_ssp370_2046_2065,
    'media_dir_v10_MPI_ssp370_2046_2065': media_dir_v10_MPI_ssp370_2046_2065,
    'nortada_MPI_ssp370_2046_2065': nortada_MPI_ssp370_2046_2065,
    'Date': daily_dates_MPI_ssp370_2046_2065
})
df_filtered_MPI_ssp370_2046_2065['ano'] = df_filtered_MPI_ssp370_2046_2065['Date'].dt.year
df_filtered_MPI_ssp370_2046_2065['mês'] = df_filtered_MPI_ssp370_2046_2065['Date'].dt.month
df_filtered_MPI_ssp370_2046_2065['dia'] = df_filtered_MPI_ssp370_2046_2065['Date'].dt.day

# wrfout MPI_ssp585
df_filtered_MPI_ssp585_2046_2065 = pd.DataFrame({
    'velocity_daily_mean_MPI_ssp585_2046_2065': velocity_daily_mean_MPI_ssp585_2046_2065,
    'media_dir_v10_MPI_ssp585_2046_2065': media_dir_v10_MPI_ssp585_2046_2065,
    'nortada_MPI_ssp585_2046_2065': nortada_MPI_ssp585_2046_2065,
    'Date': daily_dates_MPI_ssp585_2046_2065
})
df_filtered_MPI_ssp585_2046_2065['ano'] = df_filtered_MPI_ssp585_2046_2065['Date'].dt.year
df_filtered_MPI_ssp585_2046_2065['mês'] = df_filtered_MPI_ssp585_2046_2065['Date'].dt.month
df_filtered_MPI_ssp585_2046_2065['dia'] = df_filtered_MPI_ssp585_2046_2065['Date'].dt.day



# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#        GROUPBY Nº DIAS C/ NORTADA E VELOCIDADE POR ANO

nortada_count_MPI_ssp245_2046_2065 = df_filtered_MPI_ssp245_2046_2065[df_filtered_MPI_ssp245_2046_2065['nortada_MPI_ssp245_2046_2065']].groupby('ano').size()
nortada_count_MPI_ssp370_2046_2065 = df_filtered_MPI_ssp370_2046_2065[df_filtered_MPI_ssp370_2046_2065['nortada_MPI_ssp370_2046_2065']].groupby('ano').size()
nortada_count_MPI_ssp585_2046_2065 = df_filtered_MPI_ssp585_2046_2065[df_filtered_MPI_ssp585_2046_2065['nortada_MPI_ssp585_2046_2065']].groupby('ano').size()

mean_velocity_nortada_MPI_ssp245_2046_2065 = df_filtered_MPI_ssp245_2046_2065[df_filtered_MPI_ssp245_2046_2065['nortada_MPI_ssp245_2046_2065']].groupby('ano')['velocity_daily_mean_MPI_ssp245_2046_2065'].mean()
mean_velocity_nortada_MPI_ssp370_2046_2065 = df_filtered_MPI_ssp370_2046_2065[df_filtered_MPI_ssp370_2046_2065['nortada_MPI_ssp370_2046_2065']].groupby('ano')['velocity_daily_mean_MPI_ssp370_2046_2065'].mean()
mean_velocity_nortada_MPI_ssp585_2046_2065 = df_filtered_MPI_ssp585_2046_2065[df_filtered_MPI_ssp585_2046_2065['nortada_MPI_ssp585_2046_2065']].groupby('ano')['velocity_daily_mean_MPI_ssp585_2046_2065'].mean()

# Criar DataFrame para os resultados
df_nortada_count_2046_2065 = pd.DataFrame({
    'nortada_count_MPI_ssp245_2046_2065': nortada_count_MPI_ssp245_2046_2065,
    'nortada_count_MPI_ssp370_2046_2065': nortada_count_MPI_ssp370_2046_2065,
    'nortada_count_MPI_ssp585_2046_2065': nortada_count_MPI_ssp585_2046_2065,
    'mean_velocity_nortada_MPI_ssp245_2046_2065': mean_velocity_nortada_MPI_ssp245_2046_2065,
    'mean_velocity_nortada_MPI_ssp370_2046_2065': mean_velocity_nortada_MPI_ssp370_2046_2065,
    'mean_velocity_nortada_MPI_ssp585_2046_2065': mean_velocity_nortada_MPI_ssp585_2046_2065
})

# Preencher valores NaN com zero (caso não haja nortadas em algum ano)
df_nortada_count_2046_2065 = df_nortada_count_2046_2065.fillna(0)

print("Número de dias com nortada e média da velocidade por ano:")
print(df_nortada_count_2046_2065)




#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                                    2081-2100
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

file_name_MPI_ssp245_2081_2100 = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\dados_wrfout_mpi_ssp\ficheiros_2081_2100\wrfout_MPI_ssp245_2081_2100_UV10m_12_18h.nc')
file_name_MPI_ssp370_2081_2100 = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\dados_wrfout_mpi_ssp\ficheiros_2081_2100\wrfout_MPI_ssp370_2081_2100_UV10m_12_18h.nc')
file_name_MPI_ssp585_2081_2100 = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\dados_wrfout_mpi_ssp\ficheiros_2081_2100\wrfout_MPI_ssp585_2081_2100_UV10m_12_18h.nc')

# %%
u10_MPI_ssp245_2081_2100 = file_name_MPI_ssp245_2081_2100.variables['U10'][:]
v10_MPI_ssp245_2081_2100 = file_name_MPI_ssp245_2081_2100.variables['V10'][:]
lon_MPI_ssp245_2081_2100 = file_name_MPI_ssp245_2081_2100.variables['XLONG'][:]
lat_MPI_ssp245_2081_2100 = file_name_MPI_ssp245_2081_2100.variables['XLAT'][:]
time_MPI_ssp245_2081_2100 = file_name_MPI_ssp245_2081_2100.variables['XTIME'][:]

u10_MPI_ssp370_2081_2100 = file_name_MPI_ssp370_2081_2100.variables['U10'][:]
v10_MPI_ssp370_2081_2100 = file_name_MPI_ssp370_2081_2100.variables['V10'][:]
lon_MPI_ssp370_2081_2100 = file_name_MPI_ssp370_2081_2100.variables['XLONG'][:]
lat_MPI_ssp370_2081_2100 = file_name_MPI_ssp370_2081_2100.variables['XLAT'][:]
time_MPI_ssp370_2081_2100 = file_name_MPI_ssp370_2081_2100.variables['XTIME'][:]

u10_MPI_ssp585_2081_2100 = file_name_MPI_ssp585_2081_2100.variables['U10'][:]
v10_MPI_ssp585_2081_2100 = file_name_MPI_ssp585_2081_2100.variables['V10'][:]
lon_MPI_ssp585_2081_2100 = file_name_MPI_ssp585_2081_2100.variables['XLONG'][:]
lat_MPI_ssp585_2081_2100 = file_name_MPI_ssp585_2081_2100.variables['XLAT'][:]
time_MPI_ssp585_2081_2100 = file_name_MPI_ssp585_2081_2100.variables['XTIME'][:]

# %%    tempo para data e hora

time_var_MPI_ssp245_2081_2100 = file_name_MPI_ssp245_2081_2100.variables['XTIME']  
time_units_MPI_ssp245_2081_2100 = time_var_MPI_ssp245_2081_2100.units  #hours since 1994-12-8 00:00:00
time_dates_MPI_ssp245_2081_2100 = nc.num2date(time_var_MPI_ssp245_2081_2100[:], units=time_units_MPI_ssp245_2081_2100)
time_dates_py_MPI_ssp245_2081_2100 = [np.datetime64(date) for date in time_dates_MPI_ssp245_2081_2100]
time_index_MPI_ssp245_2081_2100 = pd.DatetimeIndex(time_dates_py_MPI_ssp245_2081_2100) #formato pandas DateTimeIndex

time_var_MPI_ssp370_2081_2100 = file_name_MPI_ssp370_2081_2100.variables['XTIME']  
time_units_MPI_ssp370_2081_2100 = time_var_MPI_ssp370_2081_2100.units  #hours since 1994-12-8 00:00:00
time_dates_MPI_ssp370_2081_2100 = nc.num2date(time_var_MPI_ssp370_2081_2100[:], units=time_units_MPI_ssp370_2081_2100)
time_dates_py_MPI_ssp370_2081_2100 = [np.datetime64(date) for date in time_dates_MPI_ssp370_2081_2100]
time_index_MPI_ssp370_2081_2100 = pd.DatetimeIndex(time_dates_py_MPI_ssp370_2081_2100) #formato pandas DateTimeIndex

time_var_MPI_ssp585_2081_2100 = file_name_MPI_ssp585_2081_2100.variables['XTIME']  
time_units_MPI_ssp585_2081_2100 = time_var_MPI_ssp585_2081_2100.units  #hours since 1994-12-8 00:00:00
time_dates_MPI_ssp585_2081_2100 = nc.num2date(time_var_MPI_ssp585_2081_2100[:], units=time_units_MPI_ssp585_2081_2100)
time_dates_py_MPI_ssp585_2081_2100 = [np.datetime64(date) for date in time_dates_MPI_ssp585_2081_2100]
time_index_MPI_ssp585_2081_2100 = pd.DatetimeIndex(time_dates_py_MPI_ssp585_2081_2100) #formato pandas DateTimeIndex


# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#          COORDENADAS DO PONTO + PRÓXIMO DA PRAIA DA BARRA

# Coordenadas da Praia da Barra
lon_barra = np.array([-8.7578430, -8.7288208])
lat_barra = np.array([40.6077614, 40.6470909])

lon_flat = lon_MPI_ssp245_2081_2100.flatten() #Achatamento das matrizes lon e lat para vetores (os dados não desaparecem, só ficam unidimensionais)
lat_flat = lat_MPI_ssp245_2081_2100.flatten()
distances = np.sqrt((lon_flat - lon_barra[0])**2 + (lat_flat- lat_barra[0])**2) # Cálculo distância euclidiana para cada ponto
idx_nearest_point = np.argmin(distances) #índice do ponto mais próximo de Barra
lon_nearest = lon_flat[idx_nearest_point]
lat_nearest = lat_flat[idx_nearest_point]


# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           CÁLCULO DA INTENSIDADE DO VENTO (sqrt((u^2)+(v^2))


# MPI_ssp245

lon_index_MPI_ssp245_2081_2100, lat_index_MPI_ssp245_2081_2100 = np.unravel_index(idx_nearest_point, lon_MPI_ssp245_2081_2100.shape) # Desfazer o achatamento para obter os índices originais
u10_var_MPI_ssp245_2081_2100 = u10_MPI_ssp245_2081_2100[:, lon_index_MPI_ssp245_2081_2100, lat_index_MPI_ssp245_2081_2100]
v10_var_MPI_ssp245_2081_2100 = v10_MPI_ssp245_2081_2100[:, lon_index_MPI_ssp245_2081_2100, lat_index_MPI_ssp245_2081_2100]

u10_daily_mean_MPI_ssp245_2081_2100 = []
v10_daily_mean_MPI_ssp245_2081_2100 = []
daily_dates_MPI_ssp245_2081_2100 = []
unique_days_MPI_ssp245_2081_2100 = np.unique(time_index_MPI_ssp245_2081_2100.date) # Iterar sobre os dias únicos para calcular as médias diárias
for day_MPI_ssp245_2081_2100 in unique_days_MPI_ssp245_2081_2100:

    day_indices_MPI_ssp245_2081_2100 = np.where(time_index_MPI_ssp245_2081_2100.date == day_MPI_ssp245_2081_2100)[0]  # Seleção dos índices correspondentes ao dia específico
    u10_day_mean_MPI_ssp245_2081_2100 = np.mean(u10_var_MPI_ssp245_2081_2100[day_indices_MPI_ssp245_2081_2100]) #média diária para u10 
    v10_day_mean_MPI_ssp245_2081_2100 = np.mean(v10_var_MPI_ssp245_2081_2100[day_indices_MPI_ssp245_2081_2100]) #média diária para v10
    u10_daily_mean_MPI_ssp245_2081_2100.append(u10_day_mean_MPI_ssp245_2081_2100) # Adicionar à lista de médias diárias
    v10_daily_mean_MPI_ssp245_2081_2100.append(v10_day_mean_MPI_ssp245_2081_2100)
    daily_dates_MPI_ssp245_2081_2100.append(pd.Timestamp(day_MPI_ssp245_2081_2100)) # Adicionar a data correspondente
    
u10_daily_mean_MPI_ssp245_2081_2100 = np.array(u10_daily_mean_MPI_ssp245_2081_2100) # Conversão das listas em arrays numpy para facilitar a manipulação dos dados
v10_daily_mean_MPI_ssp245_2081_2100 = np.array(v10_daily_mean_MPI_ssp245_2081_2100)

# MPI_ssp370

lon_index_MPI_ssp370_2081_2100, lat_index_MPI_ssp370_2081_2100 = np.unravel_index(idx_nearest_point, lon_MPI_ssp370_2081_2100.shape) # Desfazer o achatamento para obter os índices originais
u10_var_MPI_ssp370_2081_2100 = u10_MPI_ssp370_2081_2100[:, lon_index_MPI_ssp370_2081_2100, lat_index_MPI_ssp370_2081_2100]
v10_var_MPI_ssp370_2081_2100 = v10_MPI_ssp370_2081_2100[:, lon_index_MPI_ssp370_2081_2100, lat_index_MPI_ssp370_2081_2100]

u10_daily_mean_MPI_ssp370_2081_2100 = []
v10_daily_mean_MPI_ssp370_2081_2100 = []
daily_dates_MPI_ssp370_2081_2100 = []
unique_days_MPI_ssp370_2081_2100 = np.unique(time_index_MPI_ssp370_2081_2100.date) # Iterar sobre os dias únicos para calcular as médias diárias
for day_MPI_ssp370_2081_2100 in unique_days_MPI_ssp370_2081_2100:

    day_indices_MPI_ssp370_2081_2100 = np.where(time_index_MPI_ssp370_2081_2100.date == day_MPI_ssp370_2081_2100)[0]  # Seleção dos índices correspondentes ao dia específico
    u10_day_mean_MPI_ssp370_2081_2100 = np.mean(u10_var_MPI_ssp370_2081_2100[day_indices_MPI_ssp370_2081_2100]) #média diária para u10 
    v10_day_mean_MPI_ssp370_2081_2100 = np.mean(v10_var_MPI_ssp370_2081_2100[day_indices_MPI_ssp370_2081_2100]) #média diária para v10
    u10_daily_mean_MPI_ssp370_2081_2100.append(u10_day_mean_MPI_ssp370_2081_2100) # Adicionar à lista de médias diárias
    v10_daily_mean_MPI_ssp370_2081_2100.append(v10_day_mean_MPI_ssp370_2081_2100)
    daily_dates_MPI_ssp370_2081_2100.append(pd.Timestamp(day_MPI_ssp370_2081_2100)) # Adicionar a data correspondente
    
u10_daily_mean_MPI_ssp370_2081_2100 = np.array(u10_daily_mean_MPI_ssp370_2081_2100) # Conversão das listas em arrays numpy para facilitar a manipulação dos dados
v10_daily_mean_MPI_ssp370_2081_2100 = np.array(v10_daily_mean_MPI_ssp370_2081_2100)


# MPI_ssp585

lon_index_MPI_ssp585_2081_2100, lat_index_MPI_ssp585_2081_2100 = np.unravel_index(idx_nearest_point, lon_MPI_ssp245_2081_2100.shape) # Desfazer o achatamento para obter os índices originais
u10_var_MPI_ssp585_2081_2100 = u10_MPI_ssp585_2081_2100[:, lon_index_MPI_ssp585_2081_2100, lat_index_MPI_ssp585_2081_2100]
v10_var_MPI_ssp585_2081_2100 = v10_MPI_ssp585_2081_2100[:, lon_index_MPI_ssp585_2081_2100, lat_index_MPI_ssp585_2081_2100]

u10_daily_mean_MPI_ssp585_2081_2100 = []
v10_daily_mean_MPI_ssp585_2081_2100 = []
daily_dates_MPI_ssp585_2081_2100 = []
unique_days_MPI_ssp585_2081_2100 = np.unique(time_index_MPI_ssp585_2081_2100.date) # Iterar sobre os dias únicos para calcular as médias diárias
for day_MPI_ssp585_2081_2100 in unique_days_MPI_ssp585_2081_2100:

    day_indices_MPI_ssp585_2081_2100 = np.where(time_index_MPI_ssp585_2081_2100.date == day_MPI_ssp585_2081_2100)[0]  # Seleção dos índices correspondentes ao dia específico
    u10_day_mean_MPI_ssp585_2081_2100 = np.mean(u10_var_MPI_ssp585_2081_2100[day_indices_MPI_ssp585_2081_2100]) #média diária para u10 
    v10_day_mean_MPI_ssp585_2081_2100 = np.mean(v10_var_MPI_ssp585_2081_2100[day_indices_MPI_ssp585_2081_2100]) #média diária para v10
    u10_daily_mean_MPI_ssp585_2081_2100.append(u10_day_mean_MPI_ssp585_2081_2100) # Adicionar à lista de médias diárias
    v10_daily_mean_MPI_ssp585_2081_2100.append(v10_day_mean_MPI_ssp585_2081_2100)
    daily_dates_MPI_ssp585_2081_2100.append(pd.Timestamp(day_MPI_ssp585_2081_2100)) # Adicionar a data correspondente
    
u10_daily_mean_MPI_ssp585_2081_2100 = np.array(u10_daily_mean_MPI_ssp585_2081_2100) # Conversão das listas em arrays numpy para facilitar a manipulação dos dados
v10_daily_mean_MPI_ssp585_2081_2100 = np.array(v10_daily_mean_MPI_ssp585_2081_2100)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#          CÁLCULO DA VELOCIDADE MÉDIA DIÁRIA

velocity_daily_mean_MPI_ssp245_2081_2100 = np.sqrt(u10_daily_mean_MPI_ssp245_2081_2100**2 + v10_daily_mean_MPI_ssp245_2081_2100**2)
velocity_daily_mean_MPI_ssp370_2081_2100 = np.sqrt(u10_daily_mean_MPI_ssp370_2081_2100**2 + v10_daily_mean_MPI_ssp370_2081_2100**2)
velocity_daily_mean_MPI_ssp585_2081_2100 = np.sqrt(u10_daily_mean_MPI_ssp585_2081_2100**2 + v10_daily_mean_MPI_ssp585_2081_2100**2)

#%%

# wrfout MPI_ssp245
data_MPI_ssp245_2081_2100 = {'Date': daily_dates_MPI_ssp245_2081_2100,'Velocity': velocity_daily_mean_MPI_ssp245_2081_2100} # DataFrame com as datas e as velocidades médias diárias
df_MPI_ssp245_2081_2100 = pd.DataFrame(data_MPI_ssp245_2081_2100)  #tabela
df_MPI_ssp245_2081_2100['Year'] = df_MPI_ssp245_2081_2100['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_MPI_ssp245_2081_2100 = df_MPI_ssp245_2081_2100.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade

# wrfout MPI_ssp370
data_MPI_ssp370_2081_2100 = {'Date': daily_dates_MPI_ssp370_2081_2100,'Velocity': velocity_daily_mean_MPI_ssp370_2081_2100} # DataFrame com as datas e as velocidades médias diárias
df_MPI_ssp370_2081_2100 = pd.DataFrame(data_MPI_ssp370_2081_2100)  #tabela
df_MPI_ssp370_2081_2100['Year'] = df_MPI_ssp370_2081_2100['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_MPI_ssp370 = df_MPI_ssp370_2081_2100.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade

# wrfout MPI_ssp585
data_MPI_ssp585_2081_2100 = {'Date': daily_dates_MPI_ssp585_2081_2100,'Velocity': velocity_daily_mean_MPI_ssp585_2081_2100} # DataFrame com as datas e as velocidades médias diárias
df_MPI_ssp585_2081_2100 = pd.DataFrame(data_MPI_ssp585_2081_2100)  #tabela
df_MPI_ssp585_2081_2100['Year'] = df_MPI_ssp585_2081_2100['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_MPI_ssp585_2081_2100 = df_MPI_ssp585_2081_2100.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                   CÁLCULO DAS DIREÇÕES DO VENTO

# Exemplo de cálculo das direções diárias para ERA5 e MPI
direction_daily_MPI_ssp245_2081_2100 = np.degrees(np.arctan2(v10_daily_mean_MPI_ssp245_2081_2100, u10_daily_mean_MPI_ssp245_2081_2100))
direction_daily_MPI_ssp370_2081_2100 = np.degrees(np.arctan2(v10_daily_mean_MPI_ssp370_2081_2100, u10_daily_mean_MPI_ssp370_2081_2100))
direction_daily_MPI_ssp585_2081_2100 = np.degrees(np.arctan2(v10_daily_mean_MPI_ssp585_2081_2100, u10_daily_mean_MPI_ssp585_2081_2100))

media_dir_v10_MPI_ssp245_2081_2100 = np.zeros_like(direction_daily_MPI_ssp245_2081_2100)
ul_MPI_ssp245_2081_2100 = np.zeros_like(direction_daily_MPI_ssp245_2081_2100)
ll_MPI_ssp245_2081_2100 = np.zeros_like(direction_daily_MPI_ssp245_2081_2100)

for i in range(direction_daily_MPI_ssp245_2081_2100.shape[0]):
    media_dir_v10_MPI_ssp245_2081_2100[i], ul_MPI_ssp245_2081_2100[i], ll_MPI_ssp245_2081_2100[i] = circ_mean_degrees(direction_daily_MPI_ssp245_2081_2100[i])

media_dir_v10_MPI_ssp370_2081_2100 = np.zeros_like(direction_daily_MPI_ssp370_2081_2100)
ul_MPI_ssp370_2081_2100 = np.zeros_like(direction_daily_MPI_ssp370_2081_2100)
ll_MPI_ssp370_2081_2100 = np.zeros_like(direction_daily_MPI_ssp370_2081_2100)

for i in range(direction_daily_MPI_ssp370_2081_2100.shape[0]):
    media_dir_v10_MPI_ssp370_2081_2100[i], ul_MPI_ssp370_2081_2100[i], ll_MPI_ssp370_2081_2100[i] = circ_mean_degrees(direction_daily_MPI_ssp370_2081_2100[i])

media_dir_v10_MPI_ssp585_2081_2100 = np.zeros_like(direction_daily_MPI_ssp585_2081_2100)
ul_MPI_ssp585_2081_2100 = np.zeros_like(direction_daily_MPI_ssp585_2081_2100)
ll_MPI_ssp585_2081_2100 = np.zeros_like(direction_daily_MPI_ssp585_2081_2100)

for i in range(direction_daily_MPI_ssp585_2081_2100.shape[0]):
    media_dir_v10_MPI_ssp585_2081_2100[i], ul_MPI_ssp585_2081_2100[i], ll_MPI_ssp585_2081_2100[i] = circ_mean_degrees(direction_daily_MPI_ssp585_2081_2100[i])



# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                                  NORTADA

nortada_MPI_ssp245_2081_2100 = (media_dir_v10_MPI_ssp245_2081_2100 >= 335) | (media_dir_v10_MPI_ssp245_2081_2100 <= 25)
nortada_MPI_ssp370_2081_2100 = (media_dir_v10_MPI_ssp370_2081_2100 >= 335) | (media_dir_v10_MPI_ssp370_2081_2100 <= 25)
nortada_MPI_ssp585_2081_2100 = (media_dir_v10_MPI_ssp585_2081_2100 >= 335) | (media_dir_v10_MPI_ssp585_2081_2100 <= 25)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           VELOCIDADE MÉDIA DOS DIAS COM NORTADA

velocity_nortada_MPI_ssp245_2081_2100 = velocity_daily_mean_MPI_ssp245_2081_2100[nortada_MPI_ssp245_2081_2100]
mean_velocity_nortada_MPI_ssp245_2081_2100 = np.mean(velocity_nortada_MPI_ssp245_2081_2100)

velocity_nortada_MPI_ssp370_2081_2100 = velocity_daily_mean_MPI_ssp370_2081_2100[nortada_MPI_ssp370_2081_2100]
mean_velocity_nortada_MPI_ssp370_2081_2100 = np.mean(velocity_nortada_MPI_ssp370_2081_2100)

velocity_nortada_MPI_ssp585_2081_2100 = velocity_daily_mean_MPI_ssp585_2081_2100[nortada_MPI_ssp585_2081_2100]
mean_velocity_nortada_MPI_ssp585_2081_2100 = np.mean(velocity_nortada_MPI_ssp585_2081_2100)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   DATA FRAMES: vel_média|dir_média|nortada_diária|data|ano|mês|dia

# wrfout MPI_ssp245
df_filtered_MPI_ssp245_2081_2100 = pd.DataFrame({
    'velocity_daily_mean_MPI_ssp245_2081_2100': velocity_daily_mean_MPI_ssp245_2081_2100,
    'media_dir_v10_MPI_ssp245_2081_2100': media_dir_v10_MPI_ssp245_2081_2100,
    'nortada_MPI_ssp245_2081_2100': nortada_MPI_ssp245_2081_2100,
    'Date': daily_dates_MPI_ssp245_2081_2100
})
df_filtered_MPI_ssp245_2081_2100['ano'] = df_filtered_MPI_ssp245_2081_2100['Date'].dt.year
df_filtered_MPI_ssp245_2081_2100['mês'] = df_filtered_MPI_ssp245_2081_2100['Date'].dt.month
df_filtered_MPI_ssp245_2081_2100['dia'] = df_filtered_MPI_ssp245_2081_2100['Date'].dt.day

# wrfout MPI_ssp370
df_filtered_MPI_ssp370_2081_2100 = pd.DataFrame({
    'velocity_daily_mean_MPI_ssp370_2081_2100': velocity_daily_mean_MPI_ssp370_2081_2100,
    'media_dir_v10_MPI_ssp370_2081_2100': media_dir_v10_MPI_ssp370_2081_2100,
    'nortada_MPI_ssp370_2081_2100': nortada_MPI_ssp370_2081_2100,
    'Date': daily_dates_MPI_ssp370_2081_2100
})
df_filtered_MPI_ssp370_2081_2100['ano'] = df_filtered_MPI_ssp370_2081_2100['Date'].dt.year
df_filtered_MPI_ssp370_2081_2100['mês'] = df_filtered_MPI_ssp370_2081_2100['Date'].dt.month
df_filtered_MPI_ssp370_2081_2100['dia'] = df_filtered_MPI_ssp370_2081_2100['Date'].dt.day

# wrfout MPI_ssp585
df_filtered_MPI_ssp585_2081_2100 = pd.DataFrame({
    'velocity_daily_mean_MPI_ssp585_2081_2100': velocity_daily_mean_MPI_ssp585_2081_2100,
    'media_dir_v10_MPI_ssp585_2081_2100': media_dir_v10_MPI_ssp585_2081_2100,
    'nortada_MPI_ssp585_2081_2100': nortada_MPI_ssp585_2081_2100,
    'Date': daily_dates_MPI_ssp585_2081_2100
})
df_filtered_MPI_ssp585_2081_2100['ano'] = df_filtered_MPI_ssp585_2081_2100['Date'].dt.year
df_filtered_MPI_ssp585_2081_2100['mês'] = df_filtered_MPI_ssp585_2081_2100['Date'].dt.month
df_filtered_MPI_ssp585_2081_2100['dia'] = df_filtered_MPI_ssp585_2081_2100['Date'].dt.day



# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#        GROUPBY Nº DIAS C/ NORTADA E VELOCIDADE POR ANO

nortada_count_MPI_ssp245_2081_2100 = df_filtered_MPI_ssp245_2081_2100[df_filtered_MPI_ssp245_2081_2100['nortada_MPI_ssp245_2081_2100']].groupby('ano').size()
nortada_count_MPI_ssp370_2081_2100 = df_filtered_MPI_ssp370_2081_2100[df_filtered_MPI_ssp370_2081_2100['nortada_MPI_ssp370_2081_2100']].groupby('ano').size()
nortada_count_MPI_ssp585_2081_2100 = df_filtered_MPI_ssp585_2081_2100[df_filtered_MPI_ssp585_2081_2100['nortada_MPI_ssp585_2081_2100']].groupby('ano').size()

mean_velocity_nortada_MPI_ssp245_2081_2100 = df_filtered_MPI_ssp245_2081_2100[df_filtered_MPI_ssp245_2081_2100['nortada_MPI_ssp245_2081_2100']].groupby('ano')['velocity_daily_mean_MPI_ssp245_2081_2100'].mean()
mean_velocity_nortada_MPI_ssp370_2081_2100 = df_filtered_MPI_ssp370_2081_2100[df_filtered_MPI_ssp370_2081_2100['nortada_MPI_ssp370_2081_2100']].groupby('ano')['velocity_daily_mean_MPI_ssp370_2081_2100'].mean()
mean_velocity_nortada_MPI_ssp585_2081_2100 = df_filtered_MPI_ssp585_2081_2100[df_filtered_MPI_ssp585_2081_2100['nortada_MPI_ssp585_2081_2100']].groupby('ano')['velocity_daily_mean_MPI_ssp585_2081_2100'].mean()

# Criar DataFrame para os resultados
df_nortada_count_2081_2100 = pd.DataFrame({
    'nortada_count_MPI_ssp245_2081_2100': nortada_count_MPI_ssp245_2081_2100,
    'nortada_count_MPI_ssp370_2081_2100': nortada_count_MPI_ssp370_2081_2100,
    'nortada_count_MPI_ssp585_2081_2100': nortada_count_MPI_ssp585_2081_2100,
    'mean_velocity_nortada_MPI_ssp245_2081_2100': mean_velocity_nortada_MPI_ssp245_2081_2100,
    'mean_velocity_nortada_MPI_ssp370_2081_2100': mean_velocity_nortada_MPI_ssp370_2081_2100,
    'mean_velocity_nortada_MPI_ssp585_2081_2100': mean_velocity_nortada_MPI_ssp585_2081_2100
})

# Preencher valores NaN com zero (caso não haja nortadas em algum ano)
df_nortada_count_2081_2100 = df_nortada_count_2081_2100.fillna(0)

print("Número de dias com nortada e média da velocidade por ano:")
print(df_nortada_count_2081_2100)



#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#               NÚMERO DE DIAS COM NORTADA POR ANO (BARRA)
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind
from matplotlib.lines import Line2D

#<<<<< 2046-2065 >>>>>>

anos_2046_2065 = df_nortada_count_2046_2065.index  # Anos são os índices do DataFrame
nortada_count_MPI_ssp245_2046_2065 = df_nortada_count_2046_2065['nortada_count_MPI_ssp245_2046_2065']
nortada_count_MPI_ssp370_2046_2065 = df_nortada_count_2046_2065['nortada_count_MPI_ssp370_2046_2065']
nortada_count_MPI_ssp585_2046_2065 = df_nortada_count_2046_2065['nortada_count_MPI_ssp585_2046_2065']

# Calcular tendências e coeficientes de determinação (R^2)
nortada_coefficients_MPI_ssp245_2046_2065 = np.polyfit(anos_2046_2065, nortada_count_MPI_ssp245_2046_2065, 1)
nortada_polynomial_MPI_ssp245_2046_2065 = np.poly1d(nortada_coefficients_MPI_ssp245_2046_2065)
nortada_trend_line_MPI_ssp245_2046_2065 = nortada_polynomial_MPI_ssp245_2046_2065(anos_2046_2065)
nortada_a_MPI_ssp245_2046_2065 = nortada_coefficients_MPI_ssp245_2046_2065[0]
nortada_b_MPI_ssp245_2046_2065 = nortada_coefficients_MPI_ssp245_2046_2065[1]
nortada_r_squared_MPI_ssp245_2046_2065 = r2_score(nortada_count_MPI_ssp245_2046_2065, nortada_trend_line_MPI_ssp245_2046_2065)

nortada_coefficients_MPI_ssp370_2046_2065 = np.polyfit(anos_2046_2065, nortada_count_MPI_ssp370_2046_2065, 1)
nortada_polynomial_MPI_ssp370_2046_2065 = np.poly1d(nortada_coefficients_MPI_ssp370_2046_2065)
nortada_trend_line_MPI_ssp370_2046_2065 = nortada_polynomial_MPI_ssp370_2046_2065(anos_2046_2065)
nortada_a_MPI_ssp370_2046_2065 = nortada_coefficients_MPI_ssp370_2046_2065[0]
nortada_b_MPI_ssp370_2046_2065 = nortada_coefficients_MPI_ssp370_2046_2065[1]
nortada_r_squared_MPI_ssp370_2046_2065 = r2_score(nortada_count_MPI_ssp370_2046_2065, nortada_trend_line_MPI_ssp370_2046_2065)

nortada_coefficients_MPI_ssp585_2046_2065 = np.polyfit(anos_2046_2065, nortada_count_MPI_ssp585_2046_2065, 1)
nortada_polynomial_MPI_ssp585_2046_2065 = np.poly1d(nortada_coefficients_MPI_ssp585_2046_2065)
nortada_trend_line_MPI_ssp585_2046_2065 = nortada_polynomial_MPI_ssp585_2046_2065(anos_2046_2065)
nortada_a_MPI_ssp585_2046_2065 = nortada_coefficients_MPI_ssp585_2046_2065[0]
nortada_b_MPI_ssp585_2046_2065 = nortada_coefficients_MPI_ssp585_2046_2065[1]
nortada_r_squared_MPI_ssp585_2046_2065 = r2_score(nortada_count_MPI_ssp585_2046_2065, nortada_trend_line_MPI_ssp585_2046_2065)


#<<<<< 2081-2100 >>>>>>

anos_2081_2100 = df_nortada_count_2081_2100.index  # Anos são os índices do DataFrame
nortada_count_MPI_ssp245_2081_2100 = df_nortada_count_2081_2100['nortada_count_MPI_ssp245_2081_2100']
nortada_count_MPI_ssp370_2081_2100 = df_nortada_count_2081_2100['nortada_count_MPI_ssp370_2081_2100']
nortada_count_MPI_ssp585_2081_2100 = df_nortada_count_2081_2100['nortada_count_MPI_ssp585_2081_2100']

# Calcular tendências e coeficientes de determinação (R^2)
nortada_coefficients_MPI_ssp245_2081_2100 = np.polyfit(anos_2081_2100, nortada_count_MPI_ssp245_2081_2100, 1)
nortada_polynomial_MPI_ssp245_2081_2100 = np.poly1d(nortada_coefficients_MPI_ssp245_2081_2100)
nortada_trend_line_MPI_ssp245_2081_2100 = nortada_polynomial_MPI_ssp245_2081_2100(anos_2081_2100)
nortada_a_MPI_ssp245_2081_2100 = nortada_coefficients_MPI_ssp245_2081_2100[0]
nortada_b_MPI_ssp245_2081_2100 = nortada_coefficients_MPI_ssp245_2081_2100[1]
nortada_r_squared_MPI_ssp245_2081_2100 = r2_score(nortada_count_MPI_ssp245_2081_2100, nortada_trend_line_MPI_ssp245_2081_2100)

nortada_coefficients_MPI_ssp370_2081_2100 = np.polyfit(anos_2081_2100, nortada_count_MPI_ssp370_2081_2100, 1)
nortada_polynomial_MPI_ssp370_2081_2100 = np.poly1d(nortada_coefficients_MPI_ssp370_2081_2100)
nortada_trend_line_MPI_ssp370_2081_2100 = nortada_polynomial_MPI_ssp370_2081_2100(anos_2081_2100)
nortada_a_MPI_ssp370_2081_2100 = nortada_coefficients_MPI_ssp370_2081_2100[0]
nortada_b_MPI_ssp370_2081_2100 = nortada_coefficients_MPI_ssp370_2081_2100[1]
nortada_r_squared_MPI_ssp370_2081_2100 = r2_score(nortada_count_MPI_ssp370_2081_2100, nortada_trend_line_MPI_ssp370_2081_2100)

nortada_coefficients_MPI_ssp585_2081_2100 = np.polyfit(anos_2081_2100, nortada_count_MPI_ssp585_2081_2100, 1)
nortada_polynomial_MPI_ssp585_2081_2100 = np.poly1d(nortada_coefficients_MPI_ssp585_2081_2100)
nortada_trend_line_MPI_ssp585_2081_2100 = nortada_polynomial_MPI_ssp585_2081_2100(anos_2081_2100)
nortada_a_MPI_ssp585_2081_2100 = nortada_coefficients_MPI_ssp585_2081_2100[0]
nortada_b_MPI_ssp585_2081_2100 = nortada_coefficients_MPI_ssp585_2081_2100[1]
nortada_r_squared_MPI_ssp585_2081_2100 = r2_score(nortada_count_MPI_ssp585_2081_2100, nortada_trend_line_MPI_ssp585_2081_2100)

#%%
#<<<<< 2046-2065 >>>>>>

anos_2046_2065 = df_nortada_count_2046_2065.index  # Anos são os índices do DataFrame
vel_count_MPI_ssp245_2046_2065 = df_nortada_count_2046_2065['mean_velocity_nortada_MPI_ssp245_2046_2065']
vel_count_MPI_ssp370_2046_2065 = df_nortada_count_2046_2065['mean_velocity_nortada_MPI_ssp370_2046_2065']
vel_count_MPI_ssp585_2046_2065 = df_nortada_count_2046_2065['mean_velocity_nortada_MPI_ssp585_2046_2065']

# Calcular tendências e coeficientes de determinação (R^2)
vel_coefficients_MPI_ssp245_2046_2065 = np.polyfit(anos_2046_2065, vel_count_MPI_ssp245_2046_2065, 1)
vel_polynomial_MPI_ssp245_2046_2065 = np.poly1d(vel_coefficients_MPI_ssp245_2046_2065)
vel_trend_line_MPI_ssp245_2046_2065 = vel_polynomial_MPI_ssp245_2046_2065(anos_2046_2065)
vel_a_MPI_ssp245_2046_2065 = vel_coefficients_MPI_ssp245_2046_2065[0]
vel_b_MPI_ssp245_2046_2065 = vel_coefficients_MPI_ssp245_2046_2065[1]
vel_r_squared_MPI_ssp245_2046_2065 = r2_score(vel_count_MPI_ssp245_2046_2065, vel_trend_line_MPI_ssp245_2046_2065)

vel_coefficients_MPI_ssp370_2046_2065 = np.polyfit(anos_2046_2065, vel_count_MPI_ssp370_2046_2065, 1)
vel_polynomial_MPI_ssp370_2046_2065 = np.poly1d(vel_coefficients_MPI_ssp370_2046_2065)
vel_trend_line_MPI_ssp370_2046_2065 = vel_polynomial_MPI_ssp370_2046_2065(anos_2046_2065)
vel_a_MPI_ssp370_2046_2065 = vel_coefficients_MPI_ssp370_2046_2065[0]
vel_b_MPI_ssp370_2046_2065 = vel_coefficients_MPI_ssp370_2046_2065[1]
vel_r_squared_MPI_ssp370_2046_2065 = r2_score(vel_count_MPI_ssp370_2046_2065, vel_trend_line_MPI_ssp370_2046_2065)

vel_coefficients_MPI_ssp585_2046_2065 = np.polyfit(anos_2046_2065, vel_count_MPI_ssp585_2046_2065, 1)
vel_polynomial_MPI_ssp585_2046_2065 = np.poly1d(vel_coefficients_MPI_ssp585_2046_2065)
vel_trend_line_MPI_ssp585_2046_2065 = vel_polynomial_MPI_ssp585_2046_2065(anos_2046_2065)
vel_a_MPI_ssp585_2046_2065 = vel_coefficients_MPI_ssp585_2046_2065[0]
vel_b_MPI_ssp585_2046_2065 = vel_coefficients_MPI_ssp585_2046_2065[1]
vel_r_squared_MPI_ssp585_2046_2065 = r2_score(vel_count_MPI_ssp585_2046_2065, vel_trend_line_MPI_ssp585_2046_2065)


#<<<<< 2081-2100 >>>>>>

anos_2081_2100 = df_nortada_count_2081_2100.index  # Anos são os índices do DataFrame
vel_count_MPI_ssp245_2081_2100 = df_nortada_count_2081_2100['mean_velocity_nortada_MPI_ssp245_2081_2100']
vel_count_MPI_ssp370_2081_2100 = df_nortada_count_2081_2100['mean_velocity_nortada_MPI_ssp370_2081_2100']
vel_count_MPI_ssp585_2081_2100 = df_nortada_count_2081_2100['mean_velocity_nortada_MPI_ssp585_2081_2100']

# Calcular tendências e coeficientes de determinação (R^2)
vel_coefficients_MPI_ssp245_2081_2100 = np.polyfit(anos_2081_2100, vel_count_MPI_ssp245_2081_2100, 1)
vel_polynomial_MPI_ssp245_2081_2100 = np.poly1d(vel_coefficients_MPI_ssp245_2081_2100)
vel_trend_line_MPI_ssp245_2081_2100 =vel_polynomial_MPI_ssp245_2081_2100(anos_2081_2100)
vel_a_MPI_ssp245_2081_2100 = vel_coefficients_MPI_ssp245_2081_2100[0]
vel_b_MPI_ssp245_2081_2100 = vel_coefficients_MPI_ssp245_2081_2100[1]
vel_r_squared_MPI_ssp245_2081_2100 = r2_score(vel_count_MPI_ssp245_2081_2100, vel_trend_line_MPI_ssp245_2081_2100)

vel_coefficients_MPI_ssp370_2081_2100 = np.polyfit(anos_2081_2100, vel_count_MPI_ssp370_2081_2100, 1)
vel_polynomial_MPI_ssp370_2081_2100 = np.poly1d(vel_coefficients_MPI_ssp370_2081_2100)
vel_trend_line_MPI_ssp370_2081_2100 = vel_polynomial_MPI_ssp370_2081_2100(anos_2081_2100)
vel_a_MPI_ssp370_2081_2100 = vel_coefficients_MPI_ssp370_2081_2100[0]
vel_b_MPI_ssp370_2081_2100 = vel_coefficients_MPI_ssp370_2081_2100[1]
vel_r_squared_MPI_ssp370_2081_2100 = r2_score(vel_count_MPI_ssp370_2081_2100, vel_trend_line_MPI_ssp370_2081_2100)

vel_coefficients_MPI_ssp585_2081_2100 = np.polyfit(anos_2081_2100, vel_count_MPI_ssp585_2081_2100, 1)
vel_polynomial_MPI_ssp585_2081_2100 = np.poly1d(vel_coefficients_MPI_ssp585_2081_2100)
vel_trend_line_MPI_ssp585_2081_2100 = vel_polynomial_MPI_ssp585_2081_2100(anos_2081_2100)
vel_a_MPI_ssp585_2081_2100 = vel_coefficients_MPI_ssp585_2081_2100[0]
vel_b_MPI_ssp585_2081_2100 = vel_coefficients_MPI_ssp585_2081_2100[1]
vel_r_squared_MPI_ssp585_2081_2100 = r2_score(vel_count_MPI_ssp585_2081_2100, vel_trend_line_MPI_ssp585_2081_2100)


#%% GRÁFICOS Nº DIAS COM NORTADA POR ANO
import seaborn as sns
# ssp245 - 2046-2065
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp245_2046_2065', marker="s",  markersize=12, color="#03A9F4", label="wrfout MPI_ssp245_2046_2065")
plt.plot(anos_2046_2065, nortada_trend_line_MPI_ssp245_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2046_2065: y={nortada_a_MPI_ssp245_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp245_2046_2065 > 0 else ""}{nortada_b_MPI_ssp245_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2046_2065:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp245_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2046_2065: y={nortada_a_MPI_ssp245_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp245_2046_2065 > 0 else ""}{nortada_b_MPI_ssp245_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2046_2065:.3f}'),
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp245_2046_2065",f'Regressão linear: \n y={nortada_a_MPI_ssp245_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp245_2046_2065 > 0 else ""}{nortada_b_MPI_ssp245_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2046_2065:.3f}'],
           loc='upper left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2046, 2065, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com Nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_ssp245_2046_2065.jpeg", bbox_inches='tight')

#%%
# ssp245 - 2081-2100
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp245_2081_2100', marker="s",  markersize=12, color="#03A9F4", label="wrfout MPI_ssp245_2081_2100")
plt.plot(anos_2081_2100, nortada_trend_line_MPI_ssp245_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2081_2100: y={nortada_a_MPI_ssp245_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp245_2081_2100 > 0 else ""}{nortada_b_MPI_ssp245_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2081_2100:.3f}')


handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp245_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2081_2100: y={nortada_a_MPI_ssp245_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp245_2081_2100 > 0 else ""}{nortada_b_MPI_ssp245_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2081_2100:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp245_2081_2100",f'Regressão linear: \n y={nortada_a_MPI_ssp245_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp245_2081_2100 > 0 else ""}{nortada_b_MPI_ssp245_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2081_2100:.3f}'],
           loc='upper center',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com Nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_ssp245_2081_2100.jpeg", bbox_inches='tight')

#%%
# ssp370 - 2046-2065
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp370_2046_2065', marker="s",  markersize=12, color="#03A9F4", label="wrfout MPI_ssp370_2046_2065")
plt.plot(anos_2046_2065, nortada_trend_line_MPI_ssp370_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2046_2065: y={nortada_a_MPI_ssp370_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp370_2046_2065 > 0 else ""}{nortada_b_MPI_ssp370_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2046_2065:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, label="wrfout MPI_ssp370_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2046_2065: y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp370_2046_2065 > 0 else ""}{nortada_b_MPI_ssp370_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2046_2065:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp370_2046_2065",f'Regressão linear: \n y={nortada_a_MPI_ssp370_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp370_2046_2065 > 0 else ""}{nortada_b_MPI_ssp370_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2046_2065:.3f}'],
           loc='lower left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2046, 2065, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com Nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_ssp370_2046_2065.jpeg", bbox_inches='tight')


#%%
# ssp370 - 2081-2100
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp370_2081_2100', marker="s", markersize=12, color="#03A9F4", label="wrfout MPI_ssp370_2081_2100")
plt.plot(anos_2081_2100, nortada_trend_line_MPI_ssp370_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2081_2100: y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp370_2081_2100 > 0 else ""}{nortada_b_MPI_ssp370_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2081_2100:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp370_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2081_2100: y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp370_2081_2100 > 0 else ""}{nortada_b_MPI_ssp370_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2081_2100:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp370_2081_2100",f'Regressão linear: \n y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp370_2081_2100 > 0 else ""}{nortada_b_MPI_ssp370_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2081_2100:.3f}'],
           loc='lower left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com Nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_ssp370_2081_2100.jpeg", bbox_inches='tight')

#%%
# ssp585 - 2046-2065
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp585_2046_2065', marker="s",  markersize=12, color="#03A9F4", label="wrfout MPI_ssp585_2046_2065")
plt.plot(anos_2046_2065, nortada_trend_line_MPI_ssp585_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp585_2046_2065: y={nortada_a_MPI_ssp585_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp585_2046_2065 > 0 else ""}{nortada_b_MPI_ssp585_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2046_2065:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp585_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp585_2046_2065: y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp585_2046_2065 > 0 else ""}{nortada_b_MPI_ssp585_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2046_2065:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp585_2046_2065",f'Regressão linear: \n y={nortada_a_MPI_ssp585_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp585_2046_2065 > 0 else ""}{nortada_b_MPI_ssp585_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2046_2065:.3f}'],
           loc='upper left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2046, 2065, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com Nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_ssp585_2046_2065.jpeg", bbox_inches='tight')

#%%
# ssp585 - 2081-2100
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp585_2081_2100', marker="s",  markersize=12, color="#03A9F4", label="wrfout MPI_ssp585_2081_2100")
plt.plot(anos_2081_2100, nortada_trend_line_MPI_ssp585_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp585_2081_2100: y={nortada_a_MPI_ssp585_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp585_2081_2100 > 0 else ""}{nortada_b_MPI_ssp585_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2081_2100:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp585_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp585_2081_2100: y={nortada_a_MPI_ssp585_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp585_2081_2100 > 0 else ""}{nortada_b_MPI_ssp585_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2081_2100:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp585_2081_2100",f'Regressão linear: \n y={nortada_a_MPI_ssp585_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp585_2081_2100 > 0 else ""}{nortada_b_MPI_ssp585_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2081_2100:.3f}'],
           loc='upper left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com Nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_ssp585_2081_2100.jpeg", bbox_inches='tight')






#%% Número de dias com nortada por ano sob os três cenários de clima futuro do projeto CMIP6: SSP2-4.5, SSP3-7.0 e SSP5-8.5, abrangendo o período 2046-2065

plt.figure(figsize=(18, 8))
sns.set(style="darkgrid")

# Plot ssp245
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp245_2046_2065', marker="o",  linewidth=2.5, markersize=14, color="#03A9F4", label="SSP2-4.5")
plt.plot(anos_2046_2065, nortada_trend_line_MPI_ssp245_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP2-4.5: y={nortada_a_MPI_ssp245_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp245_2046_2065 > 0 else ""}{nortada_b_MPI_ssp245_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2046_2065:.3f}')

# Plot ssp370
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp370_2046_2065', marker="s", linewidth=2.5,  markersize=14, color="#FF9800", label="SSP3-7.0")
plt.plot(anos_2046_2065, nortada_trend_line_MPI_ssp370_2046_2065, color='#FF9800', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP3-7.0: y={nortada_a_MPI_ssp370_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp370_2046_2065 > 0 else ""}{nortada_b_MPI_ssp370_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2046_2065:.3f}')

# Plot ssp585
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp585_2046_2065', marker="^", linewidth=2.5,  markersize=14, color="#4CAF50", label="SSP5-8.5")
plt.plot(anos_2046_2065, nortada_trend_line_MPI_ssp585_2046_2065, color='#4CAF50', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP5-8.5: y={nortada_a_MPI_ssp585_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp585_2046_2065 > 0 else ""}{nortada_b_MPI_ssp585_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2046_2065:.3f}')


# Criação dos handles para a legenda
handles = [
    Line2D([0], [0], marker="o",  linewidth=2.5, markersize=14, color="#03A9F4", label="SSP2-4.5"),
    Line2D([0], [0], marker="s", linewidth=2.5,  markersize=14, color="#FF9800", label="SSP3-7.0"),
    Line2D([0], [0], marker="^", linewidth=2.5,  markersize=14, color="#4CAF50", label="SSP5-8.5"),
    Line2D([0], [0], color='#03A9F4', linestyle='--', linewidth=3, label=f'Regressão linear SSP2-4.5: \n y={nortada_a_MPI_ssp245_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp245_2046_2065 > 0 else ""}{nortada_b_MPI_ssp245_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2046_2065:.3f}'),
    Line2D([0], [0], color='#FF9800', linestyle='--', linewidth=3, label=f'Regressão linear SSP3-7.0: \n y={nortada_a_MPI_ssp370_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp370_2046_2065 > 0 else ""}{nortada_b_MPI_ssp370_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2046_2065:.3f}'),
    Line2D([0], [0], color='#4CAF50', linestyle='--', linewidth=3, label=f'Regressão linear SSP5-8.5: \n y={nortada_a_MPI_ssp585_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp585_2046_2065 > 0 else ""}{nortada_b_MPI_ssp585_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2046_2065:.3f}'),
      
]

# Adição da legenda com o texto de significância estatística
plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=[handle.get_label() for handle in handles],
           loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)


plt.xticks(np.arange(2046, 2066, 1), rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Número de dias com nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()

plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_SSPS_2046_2065.jpeg", bbox_inches='tight')




#%% Número de dias com nortada por ano sob os três cenários de clima futuro do projeto CMIP6: SSP2-4.5, SSP3-7.0 e SSP5-8.5, abrangendo o período 2081-2100

plt.figure(figsize=(18, 8))
sns.set(style="darkgrid")

# Plot ssp245
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp245_2081_2100', marker="o",  linewidth=2.5, markersize=14, color="#03A9F4", label="SSP2-4.5")
plt.plot(anos_2081_2100, nortada_trend_line_MPI_ssp245_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP2-4.5: y={nortada_a_MPI_ssp245_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp245_2081_2100 > 0 else ""}{nortada_b_MPI_ssp245_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2081_2100:.3f}')

# Plot ssp370
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp370_2081_2100', marker="s", linewidth=2.5, markersize=14, color="#FF9800", label="SSP3-7.0")
plt.plot(anos_2081_2100, nortada_trend_line_MPI_ssp370_2081_2100, color='#FF9800', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP3-7.0: y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp370_2081_2100 > 0 else ""}{nortada_b_MPI_ssp370_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2081_2100:.3f}')

# Plot ssp585
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp585_2081_2100', marker="^", linewidth=2.5,  markersize=14, color="#4CAF50", label="SSP5-8.5")
plt.plot(anos_2081_2100, nortada_trend_line_MPI_ssp585_2081_2100, color='#4CAF50', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP5-8.5: y={nortada_a_MPI_ssp585_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp585_2081_2100 > 0 else ""}{nortada_b_MPI_ssp585_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2081_2100:.3f}')


# Criação dos handles para a legenda
handles = [
    Line2D([0], [0], marker="s",  linewidth=2.5, markersize=14, color="#03A9F4", label="SSP2-4.5"),
    Line2D([0], [0], marker="s", linewidth=2.5, markersize=14, color="#FF9800", label="SSP3-7.0"),
    Line2D([0], [0], marker="s", linewidth=2.5,  markersize=14, color="#4CAF50", label="SSP5-8.5"),
    Line2D([0], [0], color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP2-4.5: \n y={nortada_a_MPI_ssp245_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp245_2081_2100 > 0 else ""}{nortada_b_MPI_ssp245_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2081_2100:.3f}'),
    Line2D([0], [0], color='#FF9800', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP3-7.0: \n y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp370_2081_2100 > 0 else ""}{nortada_b_MPI_ssp370_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2081_2100:.3f}'),
    Line2D([0], [0], color='#4CAF50', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP5-8.5: \n y={nortada_a_MPI_ssp585_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp585_2081_2100> 0 else ""}{nortada_b_MPI_ssp585_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2081_2100:.3f}'),
      
]

# Adição da legenda com o texto de significância estatística
plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=[handle.get_label() for handle in handles],
           loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)


plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Número de dias com nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()

plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_SSPS_2081_2100.jpeg", bbox_inches='tight')







#%% MÉDIA DO NÚMERO DE DIAS COM NORTADA PARA OS PERÍODOS 2046-2065 E 2081-2100

# Média do número de dias com nortada para SSP245 no período 2046-2065
media_nortada_SSP245_2046_2065 = df_nortada_count_2046_2065['nortada_count_MPI_ssp245_2046_2065'].mean()
print(f"Média de dias com nortada para SSP245 no período 2046-2065: {media_nortada_SSP245_2046_2065:.0f} dias")

# Média do número de dias com nortada para SSP370 no período 2046-2065
media_nortada_SSP370_2046_2065 = df_nortada_count_2046_2065['nortada_count_MPI_ssp370_2046_2065'].mean()
print(f"Média de dias com nortada para SSP370 no período 2046-2065: {media_nortada_SSP370_2046_2065:.0f} dias")

# Média do número de dias com nortada para SSP585 no período 2046-2065
media_nortada_SSP585_2046_2065 = df_nortada_count_2046_2065['nortada_count_MPI_ssp585_2046_2065'].mean()
print(f"Média de dias com nortada para SSP585 no período 2046-2065: {media_nortada_SSP585_2046_2065:.0f} dias")

# Média do número de dias com nortada para SSP245 no período 2081-2100
media_nortada_SSP245_2081_2100 = df_nortada_count_2081_2100['nortada_count_MPI_ssp245_2081_2100'].mean()
print(f"Média de dias com nortada para SSP245 no período 2081-2100: {media_nortada_SSP245_2081_2100:.0f} dias")

# Média do número de dias com nortada para SSP370 no período 2081-2100
media_nortada_SSP370_2081_2100 = df_nortada_count_2081_2100['nortada_count_MPI_ssp370_2081_2100'].mean()
print(f"Média de dias com nortada para SSP370 no período 2081-2100: {media_nortada_SSP370_2081_2100:.0f} dias")

# Média do número de dias com nortada para SSP585 no período 2081-2100
media_nortada_SSP585_2081_2100 = df_nortada_count_2081_2100['nortada_count_MPI_ssp585_2081_2100'].mean()
print(f"Média de dias com nortada para SSP585 no período 2081-2100: {media_nortada_SSP585_2081_2100:.0f} dias")


#%% MÉDIA DA VELOCIDADE COM NORTADA PARA OS PERÍODOS 2046-2065 E 2081-2100

# Média da velocidade média com nortada para SSP245 no período 2046-2065
media_velocidade_SSP245_2046_2065 = df_nortada_count_2046_2065['mean_velocity_nortada_MPI_ssp245_2046_2065'].mean()
print(f"Média de velocidade média com nortada para SSP245 no período 2046-2065: {media_velocidade_SSP245_2046_2065:.2f} m/s")

# Média da velocidade média com nortada para SSP370 no período 2046-2065
media_velocidade_SSP370_2046_2065 = df_nortada_count_2046_2065['mean_velocity_nortada_MPI_ssp370_2046_2065'].mean()
print(f"Média de velocidade média com nortada para SSP370 no período 2046-2065: {media_velocidade_SSP370_2046_2065:.2f} m/s")

# Média da velocidade média com nortada para SSP585 no período 2046-2065
media_velocidade_SSP585_2046_2065 = df_nortada_count_2046_2065['mean_velocity_nortada_MPI_ssp585_2046_2065'].mean()
print(f"Média de velocidade média com nortada para SSP585 no período 2046-2065: {media_velocidade_SSP585_2046_2065:.2f} m/s")

# Média da velocidade média com nortada para SSP245 no período 2081-2100
media_velocidade_SSP245_2081_2100 = df_nortada_count_2081_2100['mean_velocity_nortada_MPI_ssp245_2081_2100'].mean()
print(f"Média de velocidade média com nortada para SSP245 no período 2081-2100: {media_velocidade_SSP245_2081_2100:.2f} m/s")

# Calculando a média da velocidade média com nortada para SSP370 no período 2081-2100
media_velocidade_SSP370_2081_2100 = df_nortada_count_2081_2100['mean_velocity_nortada_MPI_ssp370_2081_2100'].mean()
print(f"Média de velocidade média com nortada para SSP370 no período 2081-2100: {media_velocidade_SSP370_2081_2100:.2f} m/s")

# Média da velocidade média com nortada para SSP585 no período 2081-2100
media_velocidade_SSP585_2081_2100 = df_nortada_count_2081_2100['mean_velocity_nortada_MPI_ssp585_2081_2100'].mean()
print(f"Média de velocidade média com nortada para SSP585 no período 2081-2100: {media_velocidade_SSP585_2081_2100:.2f} m/s")




#%%
# Suponha que você já tenha os dados armazenados em df_nortada_count_2046_2065, anos_2046_2065,
# nortada_trend_line_MPI_ssp245_2046_2065, nortada_a_MPI_ssp245_2046_2065, nortada_b_MPI_ssp245_2046_2065, nortada_r_squared_MPI_ssp245_2046_2065,
# nortada_trend_line_MPI_ssp370_2046_2065, nortada_a_MPI_ssp370_2046_2065, nortada_b_MPI_ssp370_2046_2065, nortada_r_squared_MPI_ssp370_2046_2065,
# nortada_trend_line_MPI_ssp585_2046_2065, nortada_a_MPI_ssp585_2046_2065, nortada_b_MPI_ssp585_2046_2065, nortada_r_squared_MPI_ssp585_2046_2065

# Configurar estilo seaborn e matplotlib
sns.set(style="darkgrid")

# Criar figura e subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 24), sharex=True)

# Plotar para ssp245 - 2046-2065
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp245_2046_2065', marker="s",  markersize=12, linewidth=2.5, color="#03A9F4", label="wrfout MPI_ssp245_2046_2065", ax=axs[0])
axs[0].plot(anos_2046_2065, nortada_trend_line_MPI_ssp245_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2046_2065: y={nortada_a_MPI_ssp245_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp245_2046_2065 > 0 else ""}{nortada_b_MPI_ssp245_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2046_2065:.3f}')
handles_ssp245 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, label="wrfout MPI_ssp245_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Regressão linear: y={nortada_a_MPI_ssp245_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp245_2046_2065 > 0 else ""}{nortada_b_MPI_ssp245_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2046_2065:.3f}')
]
axs[0].legend(handles=handles_ssp245, loc='upper left', facecolor='white', edgecolor='black', fontsize=25)
axs[0].set_ylabel('Número de dias com nortada', fontsize=30)
axs[0].tick_params(axis='both', which='major', labelsize=25)

# Plotar para ssp370 - 2046-2065
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp370_2046_2065', marker="s",  markersize=12, linewidth=2.5, color="#03A9F4", label="wrfout MPI_ssp370_2046_2065", ax=axs[1])
axs[1].plot(anos_2046_2065, nortada_trend_line_MPI_ssp370_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2046_2065: y={nortada_a_MPI_ssp370_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp370_2046_2065 > 0 else ""}{nortada_b_MPI_ssp370_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2046_2065:.3f}')
handles_ssp370 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp370_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Regressão linear: y={nortada_a_MPI_ssp370_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp370_2046_2065 > 0 else ""}{nortada_b_MPI_ssp370_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2046_2065:.3f}')
]
axs[1].legend(handles=handles_ssp370, loc='upper left', facecolor='white', edgecolor='black', fontsize=25)
axs[1].set_ylabel('Número de dias com nortada', fontsize=30)
axs[1].tick_params(axis='both', which='major', labelsize=25)

# Plotar para ssp585 - 2046-2065
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='nortada_count_MPI_ssp585_2046_2065', marker="s", markersize=12, linewidth=2.5, color="#03A9F4", label="wrfout MPI_ssp585_2046_2065", ax=axs[2])
axs[2].plot(anos_2046_2065, nortada_trend_line_MPI_ssp585_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp585_2046_2065: y={nortada_a_MPI_ssp585_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp585_2046_2065 > 0 else ""}{nortada_b_MPI_ssp585_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2046_2065:.3f}')
handles_ssp585 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, markeredgecolor='black', label="wrfout MPI_ssp585_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Regressão linear: y={nortada_a_MPI_ssp585_2046_2065:.3f}x{"+" if nortada_b_MPI_ssp585_2046_2065 > 0 else ""}{nortada_b_MPI_ssp585_2046_2065:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2046_2065:.3f}')
]
axs[2].legend(handles=handles_ssp585, loc='upper left', facecolor='white', edgecolor='black', fontsize=25)
axs[2].set_ylabel('Número de dias com nortada', fontsize=30)
axs[2].tick_params(axis='both', which='major', labelsize=25)

# Configurações gerais dos eixos e labels
plt.xticks(np.arange(2046, 2065, 1), rotation=45, fontsize=25)
plt.xlabel('Ano', fontsize=30)
plt.tight_layout()

plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_ssps_2046_2065_subplots.jpeg", bbox_inches='tight')


#%% NÚMERO DE DIAS COM NORTADA SPPs (2046-2065)


# Criar figura e subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 24), sharex=True)

# ssp245 - 2081-2100
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp245_2081_2100', linewidth=2.5, 
             marker="s",  markersize=12, color="#03A9F4", 
             label="wrfout MPI_ssp245_2081_2100", ax=axs[0])

axs[0].plot(anos_2081_2100, nortada_trend_line_MPI_ssp245_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, 
            label=f'Tendência MPI_ssp245_2081_2100: y={nortada_a_MPI_ssp245_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp245_2081_2100 > 0 else ""}{nortada_b_MPI_ssp245_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2081_2100:.3f}')

handles_ssp245 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, 
           label="wrfout MPI_ssp245_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, 
           label=f'Regressão linear: y={nortada_a_MPI_ssp245_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp245_2081_2100 > 0 else ""}{nortada_b_MPI_ssp245_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp245_2081_2100:.3f}')
]

axs[0].legend(handles=handles_ssp245, loc='upper right', facecolor='white', edgecolor='black', fontsize=25)
axs[0].set_ylabel('Número de dias com nortada', fontsize=30)
axs[0].tick_params(axis='both', which='major', labelsize=25)

# ssp370 - 2081-2100
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp370_2081_2100', linewidth=2.5, 
             marker="s",  markersize=12, color="#03A9F4", 
             label="wrfout MPI_ssp370_2081_2100", ax=axs[1])

axs[1].plot(anos_2081_2100, nortada_trend_line_MPI_ssp370_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, 
            label=f'Tendência MPI_ssp370_2081_2100: y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp370_2081_2100 > 0 else ""}{nortada_b_MPI_ssp370_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2081_2100:.3f}')

handles_ssp370 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  
           label="wrfout MPI_ssp370_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, 
           label=f'Regressão linear: y={nortada_a_MPI_ssp370_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp370_2081_2100 > 0 else ""}{nortada_b_MPI_ssp370_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp370_2081_2100:.3f}')
]

axs[1].legend(handles=handles_ssp370, loc='upper left', facecolor='white', edgecolor='black', fontsize=25)
axs[1].set_ylabel('Número de dias com nortada', fontsize=30)
axs[1].tick_params(axis='both', which='major', labelsize=25)

# ssp585 - 2081-2100
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='nortada_count_MPI_ssp585_2081_2100', linewidth=2.5, 
             marker="s",  markersize=12, color="#03A9F4", 
             label="wrfout MPI_ssp585_2081_2100", ax=axs[2])

axs[2].plot(anos_2081_2100, nortada_trend_line_MPI_ssp585_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, 
            label=f'Tendência MPI_ssp585_2081_2100: y={nortada_a_MPI_ssp585_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp585_2081_2100 > 0 else ""}{nortada_b_MPI_ssp585_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2081_2100:.3f}')

handles_ssp585 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=10,  
           label="wrfout MPI_ssp585_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2, 
           label=f'Regressão linear: y={nortada_a_MPI_ssp585_2081_2100:.3f}x{"+" if nortada_b_MPI_ssp585_2081_2100 > 0 else ""}{nortada_b_MPI_ssp585_2081_2100:.3f}, $R^2$={nortada_r_squared_MPI_ssp585_2081_2100:.3f}')
]

axs[2].legend(handles=handles_ssp585, loc='upper left', facecolor='white', edgecolor='black', fontsize=25)
axs[2].set_ylabel('Número de dias com nortada', fontsize=30)
axs[2].tick_params(axis='both', which='major', labelsize=25)

# Configuração dos eixos x compartilhados e rótulos gerais
plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=25)
plt.xlabel('Ano', fontsize=30)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_por_ano_wrfout_MPI_ssps_2081_2100_subplots.jpeg", bbox_inches='tight')



#%% GRÁFICOS VELOCIDADE MÉDIA COM NORTADA POR ANO

import seaborn as sns
# ssp245 - 2046-2065
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp245_2046_2065', linewidth=2.5, marker="s",  markersize=12, color="#03A9F4", label="wrfout MPI_ssp245_2046_2065")
plt.plot(anos_2046_2065, vel_trend_line_MPI_ssp245_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2046_2065: y={vel_a_MPI_ssp245_2046_2065:.3f}x{"+" if vel_b_MPI_ssp245_2046_2065 > 0 else ""}{vel_b_MPI_ssp245_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2046_2065:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp245_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2046_2065: y={vel_a_MPI_ssp245_2046_2065:.3f}x{"+" if vel_b_MPI_ssp245_2046_2065 > 0 else ""}{vel_b_MPI_ssp245_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2046_2065:.3f}'),
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp245_2046_2065",f'Regressão linear: \n y={vel_a_MPI_ssp245_2046_2065:.3f}x{"+" if vel_b_MPI_ssp245_2046_2065 > 0 else ""}{vel_b_MPI_ssp245_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2046_2065:.3f}'],
           loc='upper left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2046, 2065, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Velocidade média da Nortada (m/s)', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_ssp245_2046_2065.jpeg", bbox_inches='tight')

#%%
# ssp245 - 2081-2100
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp245_2081_2100', linewidth=2.5, marker="s", markersize=12, color="#03A9F4", label="wrfout MPI_ssp245_2081_2100")
plt.plot(anos_2081_2100, vel_trend_line_MPI_ssp245_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2081_2100: y={vel_a_MPI_ssp245_2081_2100:.3f}x{"+" if vel_b_MPI_ssp245_2081_2100 > 0 else ""}{vel_b_MPI_ssp245_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2081_2100:.3f}')


handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp245_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp245_2081_2100: y={vel_a_MPI_ssp245_2081_2100:.3f}x{"+" if vel_b_MPI_ssp245_2081_2100 > 0 else ""}{vel_b_MPI_ssp245_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2081_2100:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp245_2081_2100",f'Regressão linear: \n y={vel_a_MPI_ssp245_2081_2100:.3f}x{"+" if vel_b_MPI_ssp245_2081_2100 > 0 else ""}{vel_b_MPI_ssp245_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2081_2100:.3f}'],
           loc='upper left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',bbox_to_anchor=(0.15, 0.95),
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Velocidade média da Nortada (m/s)', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_ssp245_2081_2100.jpeg", bbox_inches='tight')

#%%
# ssp370 - 2046-2065
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp370_2046_2065', linewidth=2.5, marker="s",  markersize=12, color="#03A9F4", label="wrfout MPI_ssp370_2046_2065")
plt.plot(anos_2046_2065, vel_trend_line_MPI_ssp370_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2046_2065: y={vel_a_MPI_ssp370_2046_2065:.3f}x{"+" if vel_b_MPI_ssp370_2046_2065 > 0 else ""}{vel_b_MPI_ssp370_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2046_2065:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, label="wrfout MPI_ssp370_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2046_2065: y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp370_2046_2065 > 0 else ""}{vel_b_MPI_ssp370_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2046_2065:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp370_2046_2065",f'Regressão linear: \n y={vel_a_MPI_ssp370_2046_2065:.3f}x{"+" if vel_b_MPI_ssp370_2046_2065 > 0 else ""}{vel_b_MPI_ssp370_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2046_2065:.3f}'],
            loc='upper center',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2046, 2065, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Velocidade média da Nortada (m/s)', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_ssp370_2046_2065.jpeg", bbox_inches='tight')


#%%
# ssp370 - 2081-2100
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp370_2081_2100', linewidth=2.5, marker="s", markersize=12, color="#03A9F4", label="wrfout MPI_ssp370_2081_2100")
plt.plot(anos_2081_2100, vel_trend_line_MPI_ssp370_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2081_2100: y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp370_2081_2100 > 0 else ""}{vel_b_MPI_ssp370_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2081_2100:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, label="wrfout MPI_ssp370_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp370_2081_2100: y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp370_2081_2100 > 0 else ""}{vel_b_MPI_ssp370_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2081_2100:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp370_2081_2100",f'Regressão linear: \n y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp370_2081_2100 > 0 else ""}{vel_b_MPI_ssp370_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2081_2100:.3f}'],
                      loc='upper center',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white',
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Velocidade média da Nortada (m/s)', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_ssp370_2081_2100.jpeg", bbox_inches='tight')

#%%
# ssp585 - 2046-2065
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp585_2046_2065', linewidth=2.5, marker="s",  markersize=10, color="#03A9F4", label="wrfout MPI_ssp585_2046_2065")
plt.plot(anos_2046_2065, vel_trend_line_MPI_ssp585_2046_2065, color='#03A9F4', linestyle='--', linewidth=2, label=f'Tendência MPI_ssp585_2046_2065: y={vel_a_MPI_ssp585_2046_2065:.3f}x{"+" if vel_b_MPI_ssp585_2046_2065 > 0 else ""}{vel_b_MPI_ssp585_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2046_2065:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, label="wrfout MPI_ssp585_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp585_2046_2065: y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp585_2046_2065 > 0 else ""}{vel_b_MPI_ssp585_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2046_2065:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp585_2046_2065",f'Regressão linear: \n y={vel_a_MPI_ssp585_2046_2065:.3f}x{"+" if vel_b_MPI_ssp585_2046_2065 > 0 else ""}{vel_b_MPI_ssp585_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2046_2065:.3f}'],
           loc='lower left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white', bbox_to_anchor=(0.2, 0.05),
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2046, 2065, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Velocidade média da Nortada (m/s)', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_ssp585_2046_2065.jpeg", bbox_inches='tight')

#%%
# ssp585 - 2081-2100
sns.set(style="darkgrid")
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp585_2081_2100', linewidth=2.5, marker="s",  markersize=12, color="#03A9F4", label="wrfout MPI_ssp585_2081_2100")
plt.plot(anos_2081_2100, vel_trend_line_MPI_ssp585_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp585_2081_2100: y={vel_a_MPI_ssp585_2081_2100:.3f}x{"+" if vel_b_MPI_ssp585_2081_2100 > 0 else ""}{vel_b_MPI_ssp585_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2081_2100:.3f}')

handles = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="wrfout MPI_ssp585_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI_ssp585_2081_2100: y={nortada_a_MPI_ssp585_2081_2100:.3f}x{"+" if vel_b_MPI_ssp585_2081_2100 > 0 else ""}{vel_b_MPI_ssp585_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2081_2100:.3f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=["wrfout MPI_ssp585_2081_2100",f'Regressão linear: \n y={vel_a_MPI_ssp585_2081_2100:.3f}x{"+" if vel_b_MPI_ssp585_2081_2100 > 0 else ""}{vel_b_MPI_ssp585_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2081_2100:.3f}'],
            loc='lower left',  # Posiciona a legenda na parte superior esquerda do gráfico
           facecolor='white', 
           edgecolor='black', 
           fontsize=20)

plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Velocidade média da Nortada (m/s)', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_ssp585_2081_2100.jpeg", bbox_inches='tight')


#%% Velocidade com nortada por ano sob os três cenários de clima futuro do projeto CMIP6: SSP2-4.5, SSP3-7.0 e SSP5-8.5, abrangendo o período 2046-2065

plt.figure(figsize=(18, 8))
sns.set(style="darkgrid")

# Plot ssp245
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp245_2046_2065', marker="o",  linewidth=2.5, markersize=14, color="#03A9F4", label="SSP2-4.5")
plt.plot(anos_2046_2065, vel_trend_line_MPI_ssp245_2046_2065, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP2-4.5: y={vel_a_MPI_ssp245_2046_2065:.3f}x{"+" if vel_b_MPI_ssp245_2046_2065 > 0 else ""}{vel_b_MPI_ssp245_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2046_2065:.3f}')

# Plot ssp370
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp370_2046_2065', marker="s", linewidth=2.5,  markersize=14, color="#FF9800", label="SSP3-7.0")
plt.plot(anos_2046_2065, vel_trend_line_MPI_ssp370_2046_2065, color='#FF9800', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP3-7.0: y={vel_a_MPI_ssp370_2046_2065:.3f}x{"+" if vel_b_MPI_ssp370_2046_2065 > 0 else ""}{vel_b_MPI_ssp370_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2046_2065:.3f}')

# Plot ssp585
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp585_2046_2065', marker="^", linewidth=2.5,  markersize=14, color="#4CAF50", label="SSP5-8.5")
plt.plot(anos_2046_2065, vel_trend_line_MPI_ssp585_2046_2065, color='#4CAF50', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP5-8.5: y={vel_a_MPI_ssp585_2046_2065:.3f}x{"+" if vel_b_MPI_ssp585_2046_2065 > 0 else ""}{vel_b_MPI_ssp585_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2046_2065:.3f}')


# Criação dos handles para a legenda
handles = [
    Line2D([0], [0], marker="o",  linewidth=2.5, markersize=14, color="#03A9F4", label="SSP2-4.5"),
    Line2D([0], [0], marker="s", linewidth=2.5,  markersize=14, color="#FF9800", label="SSP3-7.0"),
    Line2D([0], [0], marker="^", linewidth=2.5,  markersize=14, color="#4CAF50", label="SSP5-8.5"),
    Line2D([0], [0], color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP2-4.5: \n y={vel_a_MPI_ssp245_2046_2065:.3f}x{"+" if vel_b_MPI_ssp245_2046_2065 > 0 else ""}{vel_b_MPI_ssp245_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2046_2065:.3f}'),
    Line2D([0], [0], color='#FF9800', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP3-7.0: \n y={vel_a_MPI_ssp370_2046_2065:.3f}x{"+" if vel_b_MPI_ssp370_2046_2065 > 0 else ""}{vel_b_MPI_ssp370_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2046_2065:.3f}'),
    Line2D([0], [0], color='#4CAF50', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP5-8.5: \n y={vel_a_MPI_ssp585_2046_2065:.3f}x{"+" if vel_b_MPI_ssp585_2046_2065 > 0 else ""}{vel_b_MPI_ssp585_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2046_2065:.3f}'),
      
]

# Adição da legenda com o texto de significância estatística
plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=[handle.get_label() for handle in handles],
           loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)


plt.xticks(np.arange(2046, 2066, 1), rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Velocidade média da nortada (m/s)', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()

plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_SSPs_2046_2065.jpeg", bbox_inches='tight')



#%% Velocidade com nortada por ano sob os três cenários de clima futuro do projeto CMIP6: SSP2-4.5, SSP3-7.0 e SSP5-8.5, abrangendo o período 2081-2100

plt.figure(figsize=(18, 8))
sns.set(style="darkgrid")

# Plot ssp245
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp245_2081_2100', marker="o",  linewidth=2.5, markersize=14, color="#03A9F4", label="SSP2-4.5")
plt.plot(anos_2081_2100, vel_trend_line_MPI_ssp245_2081_2100, color='#03A9F4', linestyle='--', linewidth=2, label=f'Regressão linear SSP2-4.5: y={vel_a_MPI_ssp245_2081_2100:.3f}x{"+" if vel_b_MPI_ssp245_2081_2100 > 0 else ""}{vel_b_MPI_ssp245_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2081_2100:.3f}')

# Plot ssp370
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp370_2081_2100', marker="s", linewidth=2.5,  markersize=14, color="#FF9800", label="SSP3-7.0")
plt.plot(anos_2081_2100, vel_trend_line_MPI_ssp370_2081_2100, color='#FF9800', linestyle='--', linewidth=2, label=f'Regressão linear SSP3-7.0: y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp370_2081_2100 > 0 else ""}{vel_b_MPI_ssp370_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2081_2100:.3f}')

# Plot ssp585
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp585_2081_2100', marker="^", linewidth=2.5,  markersize=14, color="#4CAF50", label="SSP5-8.5")
plt.plot(anos_2081_2100, vel_trend_line_MPI_ssp585_2081_2100, color='#4CAF50', linestyle='--', linewidth=2, label=f'Regressão linear SSP5-8.5: y={vel_a_MPI_ssp585_2081_2100:.3f}x{"+" if vel_b_MPI_ssp585_2081_2100 > 0 else ""}{vel_b_MPI_ssp585_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2081_2100:.3f}')


# Criação dos handles para a legenda
handles = [
    Line2D([0], [0], marker="o",  linewidth=2.5, markersize=14, color="#03A9F4", label="SSP2-4.5"),
    Line2D([0], [0], marker="s", linewidth=2.5,  markersize=14, color="#FF9800", label="SSP3-7.0"),
    Line2D([0], [0], marker="^", linewidth=2.5,  markersize=14, color="#4CAF50", label="SSP5-8.5"),
    Line2D([0], [0], color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP2-4.5: \n y={vel_a_MPI_ssp245_2081_2100:.3f}x{"+" if vel_b_MPI_ssp245_2081_2100 > 0 else ""}{vel_b_MPI_ssp245_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2081_2100:.3f}'),
    Line2D([0], [0], color='#FF9800', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP3-7.0: \n y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp370_2081_2100 > 0 else ""}{vel_b_MPI_ssp370_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2081_2100:.3f}'),
    Line2D([0], [0], color='#4CAF50', linestyle='--', linewidth=2.5, label=f'Regressão linear SSP5-8.5: \n y={vel_a_MPI_ssp585_2081_2100:.3f}x{"+" if vel_b_MPI_ssp585_2081_2100 > 0 else ""}{vel_b_MPI_ssp585_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2081_2100:.3f}'),
      
]

# Adição da legenda com o texto de significância estatística
plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=[handle.get_label() for handle in handles],
           loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)


plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Velocidade média da nortada (m/s)', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()

plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_SSPs_2081_2100.jpeg", bbox_inches='tight')

#%%


# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 24), sharex=True)

# ssp245 - 2046-2065
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp245_2046_2065', 
             marker="s", linewidth=2.5, markersize=12, color="#03A9F4", 
             label="wrfout MPI_ssp245_2046_2065", ax=axs[0])

axs[0].plot(anos_2046_2065, vel_trend_line_MPI_ssp245_2046_2065, color='#03A9F4', linestyle='--', linewidth=2, 
            label=f'Tendência MPI_ssp245_2046_2065: y={vel_a_MPI_ssp245_2046_2065:.3f}x{"+" if vel_b_MPI_ssp245_2046_2065 > 0 else ""}{vel_b_MPI_ssp245_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2046_2065:.3f}')

handles_ssp245 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, 
           label="wrfout MPI_ssp245_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, 
           label=f'Regressão linear: y={vel_a_MPI_ssp245_2046_2065:.3f}x{"+" if vel_b_MPI_ssp245_2046_2065 > 0 else ""}{vel_b_MPI_ssp245_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2046_2065:.3f}')
]

axs[0].legend(handles=handles_ssp245, loc='upper left', facecolor='white', edgecolor='black', fontsize=25)
axs[0].set_ylabel('Velocidade média com nortada (m/s)', fontsize=30)
axs[0].tick_params(axis='both', which='major', labelsize=25)

# ssp370 - 2046-2065
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp370_2046_2065', 
             marker="s", linewidth=2.5,  markersize=10, color="#03A9F4", 
             label="wrfout MPI_ssp370_2046_2065", ax=axs[1])

axs[1].plot(anos_2046_2065, vel_trend_line_MPI_ssp370_2046_2065, color='#03A9F4', linestyle='--', linewidth=2, 
            label=f'Tendência MPI_ssp370_2046_2065: y={vel_a_MPI_ssp370_2046_2065:.3f}x{"+" if vel_b_MPI_ssp370_2046_2065 > 0 else ""}{vel_b_MPI_ssp370_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2046_2065:.3f}')

handles_ssp370 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, 
           label="wrfout MPI_ssp370_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, 
           label=f'Regressão linear: y={vel_a_MPI_ssp370_2046_2065:.3f}x{"+" if vel_b_MPI_ssp370_2046_2065 > 0 else ""}{vel_b_MPI_ssp370_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2046_2065:.3f}')
]

axs[1].legend(handles=handles_ssp370, loc='upper left', facecolor='white', edgecolor='black', fontsize=25)
axs[1].set_ylabel('Velocidade média com nortada (m/s)', fontsize=30)
axs[1].tick_params(axis='both', which='major', labelsize=25)

# ssp585 - 2046-2065
sns.lineplot(data=df_nortada_count_2046_2065, x=anos_2046_2065, y='mean_velocity_nortada_MPI_ssp585_2046_2065', 
             marker="s", linewidth=2.5,  markersize=10, color="#03A9F4", 
             label="wrfout MPI_ssp585_2046_2065", ax=axs[2])

axs[2].plot(anos_2046_2065, vel_trend_line_MPI_ssp585_2046_2065, color='#03A9F4', linestyle='--', linewidth=2, 
            label=f'Tendência MPI_ssp585_2046_2065: y={vel_a_MPI_ssp585_2046_2065:.3f}x{"+" if vel_b_MPI_ssp585_2046_2065 > 0 else ""}{vel_b_MPI_ssp585_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2046_2065:.3f}')

handles_ssp585 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12, 
           label="wrfout MPI_ssp585_2046_2065"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, 
           label=f'Regressão linear: y={vel_a_MPI_ssp585_2046_2065:.3f}x{"+" if vel_b_MPI_ssp585_2046_2065 > 0 else ""}{vel_b_MPI_ssp585_2046_2065:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2046_2065:.3f}')
]

axs[2].legend(handles=handles_ssp585, loc='lower left', facecolor='white', edgecolor='black', fontsize=25)
axs[2].set_ylabel('Velocidade média da nortada (m/s)', fontsize=30)
axs[2].tick_params(axis='both', which='major', labelsize=25)

# Configuração dos eixos x compartilhados e rótulos gerais
plt.xticks(np.arange(2046, 2065, 1), rotation=45, fontsize=25)
plt.xlabel('Ano', fontsize=30)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_ssps_2046_2065_subplots.jpeg", bbox_inches='tight')


#%% VELOCIDADE MÉDIA COM NORTADA SSPs (2046-2065)


# Configurações globais
sns.set(style="darkgrid")
plt.rcParams.update({'font.size': 22})

# Criar figura e subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 24), sharex=True)

# ssp245 - 2081-2100
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp245_2081_2100', 
             marker="s",linewidth=2.5, markersize=12, color="#03A9F4", 
             label="wrfout MPI_ssp245_2081_2100", ax=axs[0])
axs[0].plot(anos_2081_2100, vel_trend_line_MPI_ssp245_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, 
            label=f'Tendência MPI_ssp245_2081_2100: y={vel_a_MPI_ssp245_2081_2100:.3f}x{"+" if vel_b_MPI_ssp245_2081_2100 > 0 else ""}{vel_b_MPI_ssp245_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2081_2100:.3f}')
handles_ssp245 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=10, 
           label="wrfout MPI_ssp245_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, 
           label=f'Regressão linear: y={vel_a_MPI_ssp245_2081_2100:.3f}x{"+" if vel_b_MPI_ssp245_2081_2100 > 0 else ""}{vel_b_MPI_ssp245_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp245_2081_2100:.3f}')
]
axs[0].legend(handles=handles_ssp245, loc='upper right', facecolor='white', edgecolor='black', fontsize=25)
axs[0].set_ylabel('Velocidade média com nortada (m/s)', fontsize=30)
axs[0].tick_params(axis='both', which='major', labelsize=25)


# ssp370 - 2081-2100
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp370_2081_2100', 
             marker="s",linewidth=2.5,  markersize=12, color="#03A9F4", 
             label="wrfout MPI_ssp370_2081_2100", ax=axs[1])
axs[1].plot(anos_2081_2100, vel_trend_line_MPI_ssp370_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, 
            label=f'Tendência MPI_ssp370_2081_2100: y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp370_2081_2100 > 0 else ""}{vel_b_MPI_ssp370_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2081_2100:.3f}')
handles_ssp370 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=10, 
           label="wrfout MPI_ssp370_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, 
           label=f'Regressão linear: y={vel_a_MPI_ssp370_2081_2100:.3f}x{"+" if vel_b_MPI_ssp370_2081_2100 > 0 else ""}{vel_b_MPI_ssp370_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp370_2081_2100:.3f}')
]
axs[1].legend(handles=handles_ssp370, loc='upper right', facecolor='white', edgecolor='black', fontsize=25)
axs[1].set_ylabel('Velocidade média com nortada (m/s)', fontsize=30)
axs[1].tick_params(axis='both', which='major', labelsize=25)

# ssp585 - 2081-2100
sns.lineplot(data=df_nortada_count_2081_2100, x=anos_2081_2100, y='mean_velocity_nortada_MPI_ssp585_2081_2100', 
             marker="s",linewidth=2.5, markersize=12, color="#03A9F4", 
             label="wrfout MPI_ssp585_2081_2100", ax=axs[2])
axs[2].plot(anos_2081_2100, vel_trend_line_MPI_ssp585_2081_2100, color='#03A9F4', linestyle='--', linewidth=2.5, 
            label=f'Tendência MPI_ssp585_2081_2100: y={vel_a_MPI_ssp585_2081_2100:.3f}x{"+" if vel_b_MPI_ssp585_2081_2100 > 0 else ""}{vel_b_MPI_ssp585_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2081_2100:.3f}')
handles_ssp585 = [
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=10, 
           label="wrfout MPI_ssp585_2081_2100"),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, 
           label=f'Regressão linear: y={vel_a_MPI_ssp585_2081_2100:.3f}x{"+" if vel_b_MPI_ssp585_2081_2100 > 0 else ""}{vel_b_MPI_ssp585_2081_2100:.3f}, $R^2$={vel_r_squared_MPI_ssp585_2081_2100:.3f}')
]
axs[2].legend(handles=handles_ssp585, loc='lower left', facecolor='white', edgecolor='black', fontsize=25)
axs[2].set_ylabel('Velocidade média da nortada (m/s)', fontsize=30)
axs[2].tick_params(axis='both', which='major', labelsize=25)

# Ajustes gerais dos subplots
plt.xticks(np.arange(2081, 2100, 1), rotation=45, fontsize=25)
plt.xlabel('Ano', fontsize=30)
plt.tight_layout()

plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_por_ano_wrfout_MPI_ssps_2081_2100_subplots.jpeg", bbox_inches='tight')

# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#        GROUPBY Nº DIAS C/ NORTADA E VELOCIDADE POR ANO

nortada_count_MPI_ssp245_2046_2065 = df_filtered_MPI_ssp245_2046_2065[df_filtered_MPI_ssp245_2046_2065['nortada_MPI_ssp245_2046_2065']].groupby('ano').size()
nortada_count_MPI_ssp370_2046_2065 = df_filtered_MPI_ssp370_2046_2065[df_filtered_MPI_ssp370_2046_2065['nortada_MPI_ssp370_2046_2065']].groupby('ano').size()
nortada_count_MPI_ssp585_2046_2065 = df_filtered_MPI_ssp585_2046_2065[df_filtered_MPI_ssp585_2046_2065['nortada_MPI_ssp585_2046_2065']].groupby('ano').size()

mean_velocity_nortada_MPI_ssp245_2046_2065 = df_filtered_MPI_ssp245_2046_2065[df_filtered_MPI_ssp245_2046_2065['nortada_MPI_ssp245_2046_2065']].groupby('ano')['velocity_daily_mean_MPI_ssp245_2046_2065'].mean()
mean_velocity_nortada_MPI_ssp370_2046_2065 = df_filtered_MPI_ssp370_2046_2065[df_filtered_MPI_ssp370_2046_2065['nortada_MPI_ssp370_2046_2065']].groupby('ano')['velocity_daily_mean_MPI_ssp370_2046_2065'].mean()
mean_velocity_nortada_MPI_ssp585_2046_2065 = df_filtered_MPI_ssp585_2046_2065[df_filtered_MPI_ssp585_2046_2065['nortada_MPI_ssp585_2046_2065']].groupby('ano')['velocity_daily_mean_MPI_ssp585_2046_2065'].mean()

# Criar DataFrame para os resultados
df_nortada_count_2046_2065 = pd.DataFrame({
    'nortada_count_MPI_ssp245_2046_2065': nortada_count_MPI_ssp245_2046_2065,
    'nortada_count_MPI_ssp370_2046_2065': nortada_count_MPI_ssp370_2046_2065,
    'nortada_count_MPI_ssp585_2046_2065': nortada_count_MPI_ssp585_2046_2065,
    'mean_velocity_nortada_MPI_ssp245_2046_2065': mean_velocity_nortada_MPI_ssp245_2046_2065,
    'mean_velocity_nortada_MPI_ssp370_2046_2065': mean_velocity_nortada_MPI_ssp370_2046_2065,
    'mean_velocity_nortada_MPI_ssp585_2046_2065': mean_velocity_nortada_MPI_ssp585_2046_2065
})

# Preencher valores NaN com zero (caso não haja nortadas em algum ano)
df_nortada_count_2046_2065 = df_nortada_count_2046_2065.fillna(0)

print("Número de dias com nortada e média da velocidade por ano:")
print(df_nortada_count_2046_2065)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         NÚMERO DE DIAS COM NORTADA MENSAL (2046-2065)


# Função para contar o número de dias com nortada por ano e mês
def count_nortada_per_month(df, df_filtered_MPI_ssp245_2046_2065):
    df_nortada = df[df[f'nortada_MPI_{df_filtered_MPI_ssp245_2046_2065}_2046_2065']]
    count_per_month = df_nortada.groupby(['ano', 'mês']).size()

    # Preencher valores NaN com zero (caso não haja nortadas em algum mês)
    count_per_month = count_per_month.fillna(0)
    
    return count_per_month

# Função para calcular a média de dias com nortada por mês
def average_nortada_per_month(count_per_month):
    # Agrupar por mês e calcular a média
    average_per_month = count_per_month.groupby('mês').mean()
    
    return average_per_month

# Contagem de dias com nortada por ano e mês para cada cenário
nortada_count_per_month_MPI_ssp245_2046_2065 = count_nortada_per_month(df_filtered_MPI_ssp245_2046_2065, 'ssp245')
nortada_count_per_month_MPI_ssp370_2046_2065 = count_nortada_per_month(df_filtered_MPI_ssp370_2046_2065, 'ssp370')
nortada_count_per_month_MPI_ssp585_2046_2065 = count_nortada_per_month(df_filtered_MPI_ssp585_2046_2065, 'ssp585')

# Calcular a média de dias com nortada por mês para cada cenário
average_nortada_MPI_ssp245_2046_2065 = average_nortada_per_month(nortada_count_per_month_MPI_ssp245_2046_2065)
average_nortada_MPI_ssp370_2046_2065 = average_nortada_per_month(nortada_count_per_month_MPI_ssp370_2046_2065)
average_nortada_MPI_ssp585_2046_2065 = average_nortada_per_month(nortada_count_per_month_MPI_ssp585_2046_2065)

# Criar DataFrame para os resultados
df_average_nortada_per_month_2046_2065 = pd.DataFrame({
    'average_nortada_MPI_ssp245_2046_2065': average_nortada_MPI_ssp245_2046_2065,
    'average_nortada_MPI_ssp370_2046_2065': average_nortada_MPI_ssp370_2046_2065,
    'average_nortada_MPI_ssp585_2046_2065': average_nortada_MPI_ssp585_2046_2065
})

print("Média de dias com nortada por mês:")
print(df_average_nortada_per_month_2046_2065)


# Contagem de dias com nortada por ano e mês para cada cenário
nortada_count_per_month_MPI_ssp245_2046_2065 = count_nortada_per_month(df_filtered_MPI_ssp245_2046_2065, 'ssp245')
nortada_count_per_month_MPI_ssp370_2046_2065 = count_nortada_per_month(df_filtered_MPI_ssp370_2046_2065, 'ssp370')
nortada_count_per_month_MPI_ssp585_2046_2065 = count_nortada_per_month(df_filtered_MPI_ssp585_2046_2065, 'ssp585')

# Calcular a média de dias com nortada por mês para cada cenário
average_nortada_MPI_ssp245_2046_2065 = average_nortada_per_month(nortada_count_per_month_MPI_ssp245_2046_2065)
average_nortada_MPI_ssp370_2046_2065 = average_nortada_per_month(nortada_count_per_month_MPI_ssp370_2046_2065)
average_nortada_MPI_ssp585_2046_2065 = average_nortada_per_month(nortada_count_per_month_MPI_ssp585_2046_2065)

# Criar DataFrame para os resultados
df_average_nortada_per_month_2046_2065 = pd.DataFrame({
    'ssp245': average_nortada_MPI_ssp245_2046_2065,
    'ssp370': average_nortada_MPI_ssp370_2046_2065,
    'ssp585': average_nortada_MPI_ssp585_2046_2065
})

# Dados de exemplo (substitua pelos seus dados)
x = np.arange(1, 13)
y_ssp245 = df_average_nortada_per_month_2046_2065['ssp245']
y_ssp370 = df_average_nortada_per_month_2046_2065['ssp370']
y_ssp585 = df_average_nortada_per_month_2046_2065['ssp585']


plt.figure(figsize=(10, 6))

# Plot para ssp245 com tendência
z_ssp245 = np.polyfit(x, y_ssp245, 1)
p_ssp245 = np.poly1d(z_ssp245)
corr_ssp245 = np.corrcoef(y_ssp245, p_ssp245(x))[0, 1]
corr_ssp245_formatted = f'{corr_ssp245:.2f}'
plt.plot(x, y_ssp245, marker='o', linewidth=2,  markersize=8, linestyle='-', color="#03A9F4", label='SSP2-4.5')
plt.plot(x, p_ssp245(x), linestyle='--', linewidth=2, color="#03A9F4", label= f'Regressão linear SSP2-4.5:\n y={p_ssp245.coefficients[1]:.3f}x{"+" if p_ssp245.coefficients[0] > 0 else ""}{p_ssp245.coefficients[0]:.3f}, $R^2$={p_ssp245.coefficients[0]:.3f}')  # Linha de tendência

# Plot para ssp370 com tendência
z_ssp370 = np.polyfit(x, y_ssp370, 1)
p_ssp370 = np.poly1d(z_ssp370)
corr_ssp370 = np.corrcoef(y_ssp370, p_ssp370(x))[0, 1]
corr_ssp370_formatted = f'{corr_ssp370:.2f}'
plt.plot(x, y_ssp370, marker='s', linewidth=2,  markersize=8, linestyle='-', color="#FF9800", label='SSP3-7.0')
plt.plot(x, p_ssp370(x), linestyle='--', linewidth=2, color="#FF9800", label= f'Regressão linear SSP2-4.5:\n y={p_ssp370.coefficients[1]:.3f}x{"+" if p_ssp370.coefficients[0] > 0 else ""}{p_ssp370.coefficients[0]:.3f}, $R^2$={p_ssp370.coefficients[0]:.3f}')  # Linha de tendência

# Plot para ssp585 com tendência
z_ssp585 = np.polyfit(x, y_ssp585, 1)
p_ssp585 = np.poly1d(z_ssp585)
corr_ssp585 = np.corrcoef(y_ssp585, p_ssp585(x))[0, 1]
corr_ssp585_formatted = f'{corr_ssp585:.2f}'
plt.plot(x, y_ssp585, marker='^', linewidth=2,  markersize=8,  linestyle='-', color='#4CAF50', label='SSP5-8.5')
plt.plot(x, p_ssp585(x), linestyle='--', linewidth=2, color='#4CAF50', label= f'Regressão linear SSP2-4.5:\n y={p_ssp585.coefficients[1]:.3f}x{"+" if p_ssp585.coefficients[0] > 0 else ""}{p_ssp585.coefficients[0]:.3f}, $R^2$={p_ssp585.coefficients[0]:.3f}')  # Linha de tendência


plt.xlabel('Mês')
plt.ylabel('Número de dias com Nortada')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)

plt.legend(loc='upper right', facecolor='white', fontsize=10)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_médias_mensais_regressões_lineares_2046_2065.jpeg", bbox_inches='tight')

# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         NÚMERO DE DIAS COM NORTADA MENSAL (2081-2100)

# Função para contar o número de dias com nortada por ano e mês
def count_nortada_per_month(df, df_filtered_MPI_ssp245_2081_2100):
    df_nortada = df[df[f'nortada_MPI_{df_filtered_MPI_ssp245_2081_2100}_2081_2100']]
    count_per_month = df_nortada.groupby(['ano', 'mês']).size()

    # Preencher valores NaN com zero (caso não haja nortadas em algum mês)
    count_per_month = count_per_month.fillna(0)
    
    return count_per_month

# Função para calcular a média de dias com nortada por mês
def average_nortada_per_month(count_per_month):
    # Agrupar por mês e calcular a média
    average_per_month = count_per_month.groupby('mês').mean()
    
    return average_per_month

# Contagem de dias com nortada por ano e mês para cada cenário
nortada_count_per_month_MPI_ssp245_2081_2100 = count_nortada_per_month(df_filtered_MPI_ssp245_2081_2100, 'ssp245')
nortada_count_per_month_MPI_ssp370_2081_2100 = count_nortada_per_month(df_filtered_MPI_ssp370_2081_2100, 'ssp370')
nortada_count_per_month_MPI_ssp585_2081_2100 = count_nortada_per_month(df_filtered_MPI_ssp585_2081_2100, 'ssp585')

# Calcular a média de dias com nortada por mês para cada cenário
average_nortada_MPI_ssp245_2081_2100 = average_nortada_per_month(nortada_count_per_month_MPI_ssp245_2081_2100)
average_nortada_MPI_ssp370_2081_2100 = average_nortada_per_month(nortada_count_per_month_MPI_ssp370_2081_2100)
average_nortada_MPI_ssp585_2081_2100 = average_nortada_per_month(nortada_count_per_month_MPI_ssp585_2081_2100)

# Criar DataFrame para os resultados
df_average_nortada_per_month_2081_2100 = pd.DataFrame({
    'average_nortada_MPI_ssp245_2081_2100': average_nortada_MPI_ssp245_2081_2100,
    'average_nortada_MPI_ssp370_2081_2100': average_nortada_MPI_ssp370_2081_2100,
    'average_nortada_MPI_ssp585_2081_2100': average_nortada_MPI_ssp585_2081_2100
})

print("Média de dias com nortada por mês:")
print(df_average_nortada_per_month_2081_2100)


# Contagem de dias com nortada por ano e mês para cada cenário
nortada_count_per_month_MPI_ssp245_2081_2100 = count_nortada_per_month(df_filtered_MPI_ssp245_2081_2100, 'ssp245')
nortada_count_per_month_MPI_ssp370_2081_2100 = count_nortada_per_month(df_filtered_MPI_ssp370_2081_2100, 'ssp370')
nortada_count_per_month_MPI_ssp585_2081_2100 = count_nortada_per_month(df_filtered_MPI_ssp585_2081_2100, 'ssp585')

# Calcular a média de dias com nortada por mês para cada cenário
average_nortada_MPI_ssp245_2081_2100 = average_nortada_per_month(nortada_count_per_month_MPI_ssp245_2081_2100)
average_nortada_MPI_ssp370_2081_2100 = average_nortada_per_month(nortada_count_per_month_MPI_ssp370_2081_2100)
average_nortada_MPI_ssp585_2081_2100 = average_nortada_per_month(nortada_count_per_month_MPI_ssp585_2081_2100)

# Criar DataFrame para os resultados
df_average_nortada_per_month_2081_2100 = pd.DataFrame({
    'ssp245': average_nortada_MPI_ssp245_2081_2100,
    'ssp370': average_nortada_MPI_ssp370_2081_2100,
    'ssp585': average_nortada_MPI_ssp585_2081_2100
})

# Dados de exemplo (substitua pelos seus dados)
x = np.arange(1, 13)
y_ssp245 = df_average_nortada_per_month_2081_2100['ssp245']
y_ssp370 = df_average_nortada_per_month_2081_2100['ssp370']
y_ssp585 = df_average_nortada_per_month_2081_2100['ssp585']


plt.figure(figsize=(10, 6))

# Plot para ssp245 com tendência
z_ssp245 = np.polyfit(x, y_ssp245, 1)
p_ssp245 = np.poly1d(z_ssp245)
corr_ssp245 = np.corrcoef(y_ssp245, p_ssp245(x))[0, 1]
corr_ssp245_formatted = f'{corr_ssp245:.2f}'
plt.plot(x, y_ssp245, marker='o', linewidth=2, markersize=8, linestyle='-', color="#03A9F4", label='SSP2-4.5')
plt.plot(x, p_ssp245(x), linestyle='--',linewidth=2,  color="#03A9F4", label= f'Regressão linear SSP2-4.5:\n y={p_ssp245.coefficients[1]:.3f}x{"+" if p_ssp245.coefficients[0] > 0 else ""}{p_ssp245.coefficients[0]:.3f}, $R^2$={p_ssp245.coefficients[0]:.3f}')  # Linha de tendência

# Plot para ssp370 com tendência
z_ssp370 = np.polyfit(x, y_ssp370, 1)
p_ssp370 = np.poly1d(z_ssp370)
corr_ssp370 = np.corrcoef(y_ssp370, p_ssp370(x))[0, 1]
corr_ssp370_formatted = f'{corr_ssp370:.2f}'
plt.plot(x, y_ssp370, marker='s', linewidth=2, markersize=8, linestyle='-', color="#FF9800", label='SSP3-7.0')
plt.plot(x, p_ssp370(x), linestyle='--', linewidth=2, color="#FF9800", label= f'Regressão linear SSP2-4.5:\n y={p_ssp370.coefficients[1]:.3f}x{"+" if p_ssp370.coefficients[0] > 0 else ""}{p_ssp370.coefficients[0]:.3f}, $R^2$={p_ssp370.coefficients[0]:.3f}')  # Linha de tendência

# Plot para ssp585 com tendência
z_ssp585 = np.polyfit(x, y_ssp585, 1)
p_ssp585 = np.poly1d(z_ssp585)
corr_ssp585 = np.corrcoef(y_ssp585, p_ssp585(x))[0, 1]
corr_ssp585_formatted = f'{corr_ssp585:.2f}'
plt.plot(x, y_ssp585, marker='^', linewidth=2, markersize=8, linestyle='-', color='#4CAF50', label='SSP5-8.5')
plt.plot(x, p_ssp585(x), linestyle='--', linewidth=2, color='#4CAF50', label= f'Regressão linear SSP2-4.5:\n y={p_ssp585.coefficients[1]:.3f}x{"+" if p_ssp585.coefficients[0] > 0 else ""}{p_ssp585.coefficients[0]:.3f}, $R^2$={p_ssp585.coefficients[0]:.3f}')  # Linha de tendência


plt.xlabel('Mês')
plt.ylabel('Número de dias com Nortada')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)

plt.legend(loc='upper right', facecolor='white', fontsize=10)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_médias_mensais_regressões_lineares_2081_2100.jpeg", bbox_inches='tight')



# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         VELOCIDADE COM NORTADA MENSAL (2046-2065)

# Função para calcular a média da velocidade diária média por mês
def average_velocity_per_month(df, scenario_name):
    average_per_month = df.groupby(['mês']).mean()[f'velocity_daily_mean_MPI_{scenario_name}_2046_2065']
    return average_per_month

# Calcular a média da velocidade diária média por mês para SSP245, SSP370 e SSP585
average_velocity_MPI_ssp245_2046_2065 = average_velocity_per_month(df_filtered_MPI_ssp245_2046_2065, 'ssp245')
average_velocity_MPI_ssp370_2046_2065 = average_velocity_per_month(df_filtered_MPI_ssp370_2046_2065, 'ssp370')
average_velocity_MPI_ssp585_2046_2065 = average_velocity_per_month(df_filtered_MPI_ssp585_2046_2065, 'ssp585')


# Dados de exemplo (substitua pelos seus dados)
x = np.arange(1, 13)
y_ssp245 = average_velocity_MPI_ssp245_2046_2065
y_ssp370 = average_velocity_MPI_ssp370_2046_2065
y_ssp585 = average_velocity_MPI_ssp585_2046_2065

# Criar o plot de linha
plt.figure(figsize=(10, 6))

# Plot para SSP245 com tendência
z_ssp245 = np.polyfit(x, y_ssp245, 1)
p_ssp245 = np.poly1d(z_ssp245)
corr_ssp245 = np.corrcoef(y_ssp245, p_ssp245(x))[0, 1]
corr_ssp245_formatted = f'{corr_ssp245:.2f}'
plt.plot(x, y_ssp245, marker='o', linewidth=2, markersize=8, linestyle='-', color="#03A9F4", label='SSP2-4.5')
plt.plot(x, p_ssp245(x), linestyle='--', linewidth=2, color="#03A9F4", label=f'Regressão linear SSP2-4.5:\n y={p_ssp245.coefficients[1]:.3f}x{"+" if p_ssp245.coefficients[0] > 0 else ""}{p_ssp245.coefficients[0]:.3f}, $R^2$={p_ssp245.coefficients[0]:.3f}')  # Linha de tendência

# Plot para SSP370 com tendência
z_ssp370 = np.polyfit(x, y_ssp370, 1)
p_ssp370 = np.poly1d(z_ssp370)
corr_ssp370 = np.corrcoef(y_ssp370, p_ssp370(x))[0, 1]
corr_ssp370_formatted = f'{corr_ssp370:.2f}'
plt.plot(x, y_ssp370, marker='s', linewidth=2, markersize=8,  linestyle='-', color="#FF9800", label='SSP3-7.0')
plt.plot(x, p_ssp370(x), linestyle='--', linewidth=2, color="#FF9800", label=f'Regressão linear SSP3-7.0:\n y={p_ssp370.coefficients[1]:.3f}x{"+" if p_ssp370.coefficients[0] > 0 else ""}{p_ssp370.coefficients[0]:.3f}, $R^2$={p_ssp370.coefficients[0]:.3f}')  # Linha de tendência

# Plot para SSP585 com tendência
z_ssp585 = np.polyfit(x, y_ssp585, 1)
p_ssp585 = np.poly1d(z_ssp585)
corr_ssp585 = np.corrcoef(y_ssp585, p_ssp585(x))[0, 1]
corr_ssp585_formatted = f'{corr_ssp585:.2f}'
plt.plot(x, y_ssp585, marker='^', linewidth=2, markersize=8, linestyle='-', color='#4CAF50', label='SSP5-8.5')
plt.plot(x, p_ssp585(x), linestyle='--', linewidth=2, color='#4CAF50', label=f'Regressão linear SSP5-8.5:\n y={p_ssp585.coefficients[1]:.3f}x{"+" if p_ssp585.coefficients[0] > 0 else ""}{p_ssp585.coefficients[0]:.3f}, $R^2$={p_ssp585.coefficients[0]:.3f}')  # Linha de tendência

plt.xlabel('Mês')
plt.ylabel('Velocidade Média (m/s))')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)

plt.legend(loc='upper right', facecolor='white', fontsize=10)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_médias_mensais_regressões_lineares_2046_2065.jpeg", bbox_inches='tight')


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         VELOCIDADE COM NORTADA MENSAL (2081-2100)

# Função para calcular a média da velocidade diária média por mês
def average_velocity_per_month(df, scenario_name):
    average_per_month = df.groupby(['mês']).mean()[f'velocity_daily_mean_MPI_{scenario_name}_2081_2100']
    return average_per_month

# Calcular a média da velocidade diária média por mês para SSP245, SSP370 e SSP585
average_velocity_MPI_ssp245_2081_2100 = average_velocity_per_month(df_filtered_MPI_ssp245_2081_2100, 'ssp245')
average_velocity_MPI_ssp370_2081_2100 = average_velocity_per_month(df_filtered_MPI_ssp370_2081_2100, 'ssp370')
average_velocity_MPI_ssp585_2081_2100 = average_velocity_per_month(df_filtered_MPI_ssp585_2081_2100, 'ssp585')

# Dados de exemplo (substitua pelos seus dados)
x = np.arange(1, 13)
y_ssp245 = average_velocity_MPI_ssp245_2081_2100
y_ssp370 = average_velocity_MPI_ssp370_2081_2100
y_ssp585 = average_velocity_MPI_ssp585_2081_2100

# Criar o plot de linha
plt.figure(figsize=(10, 6))

# Plot para SSP245 com tendência
z_ssp245 = np.polyfit(x, y_ssp245, 1)
p_ssp245 = np.poly1d(z_ssp245)
corr_ssp245 = np.corrcoef(y_ssp245, p_ssp245(x))[0, 1]
corr_ssp245_formatted = f'{corr_ssp245:.2f}'
plt.plot(x, y_ssp245, marker='o', linewidth=2, markersize=8,linestyle='-', color="#03A9F4", label='SSP2-4.5')
plt.plot(x, p_ssp245(x), linestyle='--', linewidth=2, color="#03A9F4", label=f'Regressão linear SSP2-4.5:\n y={p_ssp245.coefficients[1]:.3f}x{"+" if p_ssp245.coefficients[0] > 0 else ""}{p_ssp245.coefficients[0]:.3f}, $R^2$={p_ssp245.coefficients[0]:.3f}')  # Linha de tendência

# Plot para SSP370 com tendência
z_ssp370 = np.polyfit(x, y_ssp370, 1)
p_ssp370 = np.poly1d(z_ssp370)
corr_ssp370 = np.corrcoef(y_ssp370, p_ssp370(x))[0, 1]
corr_ssp370_formatted = f'{corr_ssp370:.2f}'
plt.plot(x, y_ssp370, marker='s', linewidth=2, markersize=8,linestyle='-', color="#FF9800", label='SSP3-7.0')
plt.plot(x, p_ssp370(x), linestyle='--', linewidth=2, color="#FF9800", label=f'Regressão linear SSP3-7.0:\n y={p_ssp370.coefficients[1]:.3f}x{"+" if p_ssp370.coefficients[0] > 0 else ""}{p_ssp370.coefficients[0]:.3f}, $R^2$={p_ssp370.coefficients[0]:.3f}')  # Linha de tendência

# Plot para SSP585 com tendência
z_ssp585 = np.polyfit(x, y_ssp585, 1)
p_ssp585 = np.poly1d(z_ssp585)
corr_ssp585 = np.corrcoef(y_ssp585, p_ssp585(x))[0, 1]
corr_ssp585_formatted = f'{corr_ssp585:.2f}'
plt.plot(x, y_ssp585, marker='^', linewidth=2, markersize=8, linestyle='-', color='#4CAF50', label='SSP5-8.5')
plt.plot(x, p_ssp585(x), linestyle='--', linewidth=2, color='#4CAF50', label=f'Regressão linear SSP5-8.5:\n y={p_ssp585.coefficients[1]:.3f}x{"+" if p_ssp585.coefficients[0] > 0 else ""}{p_ssp585.coefficients[0]:.3f}, $R^2$={p_ssp585.coefficients[0]:.3f}')  # Linha de tendência

plt.xlabel('Mês')
plt.ylabel('Velocidade Média (m/s)')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)

plt.legend(loc='upper right', facecolor='white', fontsize=10)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_médias_mensais_regressões_lineares_2081_2100.jpeg", bbox_inches='tight')




# %%
