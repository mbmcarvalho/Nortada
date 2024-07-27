#%%
#
#
#
#                               CLIMA PRESENTE  (1995-2014)
#
#
# 
# 
# #         
#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#               SÉRIES TEMPORAIS - wrfoutERA5 e wrfoutMPI - 1995 a 2014

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

import pandas as pd
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
from scipy.stats import ttest_ind

#%%

file_name_MPI = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\Dados_2\ficheiros\wrfout_MPI_hist_UV10m_12_18h.nc')



# Extração variáveis wrfout MPI
u10_MPI = file_name_MPI.variables['U10'][:]
v10_MPI = file_name_MPI.variables['V10'][:]
lon_MPI = file_name_MPI.variables['XLONG'][:]
lat_MPI = file_name_MPI.variables['XLAT'][:]
time_MPI = file_name_MPI.variables['XTIME'][:]


# %%

# Conversão de valores de tempo para data e hora 

# wrfout MPI
time_var_MPI = file_name_MPI.variables['XTIME']  
time_units_MPI = time_var_MPI.units  #hours since 1994-12-8 00:00:00
time_dates_MPI = nc.num2date(time_var_MPI[:], units=time_units_MPI)
time_dates_py_MPI = [np.datetime64(date) for date in time_dates_MPI]
time_index_MPI = pd.DatetimeIndex(time_dates_py_MPI) #formato pandas DateTimeIndex


# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#          COORDENADAS DO PONTO + PRÓXIMO DA PRAIA DA BARRA

# Coordenadas da Praia da Barra
lon_barra = np.array([-8.7578430, -8.7288208])
lat_barra = np.array([40.6077614, 40.6470909])


# wrfout MPI
lon_flat_MPI = lon_MPI.flatten() #Achatamento das matrizes lon e lat para vetores
lat_flat_MPI = lat_MPI.flatten()
distances_MPI = np.sqrt((lon_flat_MPI - lon_barra[0])**2 + (lat_flat_MPI - lat_barra[0])**2) # Cálculo distância euclidiana para cada ponto
idx_nearest_point_MPI = np.argmin(distances_MPI) #índice do ponto mais próximo de Barra
lon_nearest_MPI = lon_flat_MPI[idx_nearest_point_MPI]
lat_nearest_MPI = lat_flat_MPI[idx_nearest_point_MPI]



# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   CÁLCULO DA DISTÂNCIA DO PONTO + PRÓXIMO DA PRAIA DA BARRA À COSTA
#              

from geopy.distance import distance

coords_nearest = (lat_nearest_MPI, lon_nearest_MPI) # Coordenadas do ponto mais próximo
coords_barra = [(lat_barra[0], lon_barra[0]), (lat_barra[1], lon_barra[1])]  # Coordenadas da Praia da Barra

# Cálculo da distância entre o ponto mais próximo e a costa (Praia da Barra)
dist_to_coast_km = min(distance(coords_nearest, coords_barra[0]).km,
                       distance(coords_nearest, coords_barra[1]).km)
print(f"A distância do ponto mais próximo da Praia da Barra à costa é aproximadamente {dist_to_coast_km:.2f} km.")



# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           CÁLCULO DA INTENSIDADE DO VENTO (sqrt((u^2)+(v^2))


# wrfout MPI

lon_index_MPI, lat_index_MPI = np.unravel_index(idx_nearest_point_MPI, lon_MPI.shape) # Desfazer o achatamento para obter os índices originais
u10_var_MPI = u10_MPI[:, lon_index_MPI, lat_index_MPI]
v10_var_MPI = v10_MPI[:, lon_index_MPI, lat_index_MPI]

u10_daily_mean_MPI = []
v10_daily_mean_MPI = []
daily_dates_MPI = []
unique_days_MPI = np.unique(time_index_MPI.date) # Iterar sobre os dias únicos para calcular as médias diárias
for day_MPI in unique_days_MPI:

    day_indices_MPI = np.where(time_index_MPI.date == day_MPI)[0]  # Seleção dos índices correspondentes ao dia específico
    u10_day_mean_MPI = np.mean(u10_var_MPI[day_indices_MPI]) #média diária para u10 
    v10_day_mean_MPI = np.mean(v10_var_MPI[day_indices_MPI]) #média diária para v10
    u10_daily_mean_MPI.append(u10_day_mean_MPI) # Adicionar à lista de médias diárias
    v10_daily_mean_MPI.append(v10_day_mean_MPI)
    daily_dates_MPI.append(pd.Timestamp(day_MPI)) # Adicionar a data correspondente
    
u10_daily_mean_MPI = np.array(u10_daily_mean_MPI) # Conversão das listas em arrays numpy para facilitar a manipulação dos dados
v10_daily_mean_MPI = np.array(v10_daily_mean_MPI)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#          CÁLCULO DA VELOCIDADE MÉDIA DIÁRIA

# wrfout MPI

velocity_daily_mean_MPI = np.sqrt(u10_daily_mean_MPI**2 + v10_daily_mean_MPI**2)
media_velocidade_total_MPI = np.mean(velocity_daily_mean_MPI)  # Calcular a média da velocidade média diária

print(f"A média da velocidade média diária ao longo do período (wrfout MPI) é: {media_velocidade_total_MPI} m/s")

#%% 


# wrfout MPI
data_MPI = {'Date': daily_dates_MPI,'Velocity': velocity_daily_mean_MPI} # DataFrame com as datas e as velocidades médias diárias
df_MPI = pd.DataFrame(data_MPI)  #tabela
df_MPI['Year'] = df_MPI['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_MPI = df_MPI.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade




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
direction_daily_MPI = np.degrees(np.arctan2(v10_daily_mean_MPI, u10_daily_mean_MPI))



# Calcular as direções corrigidas para cada dia individualmente para MPI
media_dir_v10_MPI = np.zeros_like(direction_daily_MPI)
ul_MPI = np.zeros_like(direction_daily_MPI)
ll_MPI = np.zeros_like(direction_daily_MPI)

for i in range(direction_daily_MPI.shape[0]):
    media_dir_v10_MPI[i], ul_MPI[i], ll_MPI[i] = circ_mean_degrees(direction_daily_MPI[i])

# Imprimir os resultados de um exemplo qualquer de dia
dia_exemplo = 0
print(f'Direção média corrigida para o dia {dia_exemplo + 1} (MPI): {media_dir_v10_MPI[dia_exemplo]:.2f} graus')
print(f'Intervalo de confiança para o dia {dia_exemplo + 1} (MPI): [{ll_MPI[dia_exemplo]:.2f}, {ul_MPI[dia_exemplo]:.2f}] graus')


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                                  NORTADA

nortada_MPI = (media_dir_v10_MPI >= 335) | (media_dir_v10_MPI <= 25)

# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           VELOCIDADE MÉDIA DOS DIAS COM NORTADA

velocity_nortada_MPI = velocity_daily_mean_MPI[nortada_MPI]
mean_velocity_nortada_MPI = np.mean(velocity_nortada_MPI)

print(f"A velocidade média nos dias com nortada (MPI) é: {mean_velocity_nortada_MPI} m/s")


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                             DATA FRAMES


# Criando o DataFrame para wrfout MPI
data_MPI = {
    'Date': daily_dates_MPI,
    'U10_mean': u10_daily_mean_MPI,
    'V10_mean': v10_daily_mean_MPI,
    'Velocity_mean': velocity_daily_mean_MPI,
    'Direction_mean': media_dir_v10_MPI,
    'CI_lower': ll_MPI,
    'CI_upper': ul_MPI,
    'Nortada': nortada_MPI
}
df_MPI = pd.DataFrame(data_MPI)


print("\nDados do wrfout MPI:")
print(df_MPI.head())


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   DATA FRAMES: vel_média|dir_média|nortada_diária|data|ano|mês|dia


df_filtered_MPI = pd.DataFrame({
    'velocity_daily_mean_MPI': velocity_daily_mean_MPI,
    'media_dir_v10_MPI': media_dir_v10_MPI,
    'nortada_MPI': nortada_MPI,
    'Date': daily_dates_MPI
})
df_filtered_MPI['ano'] = df_filtered_MPI['Date'].dt.year
df_filtered_MPI['mês'] = df_filtered_MPI['Date'].dt.month
df_filtered_MPI['dia'] = df_filtered_MPI['Date'].dt.day


print("\nDataFrame df_filtered_MPI:")
print(df_filtered_MPI)




# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#        GROUPBY Nº DIAS C/ NORTADA E VELOCIDADE POR ANO


nortada_count_MPI = df_filtered_MPI[df_filtered_MPI['nortada_MPI']].groupby('ano').size()
mean_velocity_nortada_MPI = df_filtered_MPI[df_filtered_MPI['nortada_MPI']].groupby('ano')['velocity_daily_mean_MPI'].mean()

# Criar DataFrame para os resultados
df_nortada_count = pd.DataFrame({
    'nortada_count_MPI': nortada_count_MPI,
    'mean_velocity_nortada_MPI': mean_velocity_nortada_MPI
})

# Preencher valores NaN com zero (caso não haja nortadas em algum ano)
df_nortada_count = df_nortada_count.fillna(0)

print("Número de dias com nortada e média da velocidade por ano:")
print(df_nortada_count)



# %%
# Function to count the number of days with "nortada" per year and month
def count_nortada_per_month(df):
    df_nortada = df[df['nortada_MPI']]  # Assuming 'nortada_MPI' column exists indicating days with 'nortada'
    count_per_month = df_nortada.groupby(['ano', 'mês']).size()

    # Preencher valores NaN com zero (caso não haja nortadas em algum mês)
    count_per_month = count_per_month.fillna(0)
    
    return count_per_month
# %%

# Function to calculate the average number of days with "nortada" per month
def average_nortada_per_month(count_per_month):
    # Agrupar por mês e calcular a média
    average_per_month = count_per_month.groupby('mês').mean()
    
    return average_per_month
# %%
# Contagem de dias com nortada por ano e mês para o período 1995-2014
nortada_count_per_month_MPI = count_nortada_per_month(df_filtered_MPI)

# Calcular a média de dias com nortada por mês para o período 1995-2014
average_nortada_MPI = average_nortada_per_month(nortada_count_per_month_MPI)

# Criar DataFrame para os resultados
df_average_nortada_per_month_1995_2014 = pd.DataFrame({
    'average_nortada_MPI': average_nortada_MPI
})

print("Média de dias com nortada por mês (1995-2014):")
print(df_average_nortada_per_month_1995_2014)


# %%
import matplotlib.pyplot as plt

months = df_average_nortada_per_month_1995_2014.index  # Meses de 1 a 12
average_nortada = df_average_nortada_per_month_1995_2014['average_nortada_MPI'].values

plt.figure(figsize=(10, 6))
plt.plot(months, average_nortada, marker='s', color='#03A9F4', linestyle='-', linewidth=2, markersize=8, label='WRF MPI')
plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Mês')
plt.ylabel('Número de dias com nortada')
plt.grid(True)
plt.legend(facecolor='white', fontsize=15)
plt.tight_layout()
plt.title('1995-2014', fontsize=17)
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Validação/num_dias_nortada_medias_mensais.jpeg", bbox_inches='tight')


#%% VELOCIDADE MÉDIA MENSAL (1995-2014)

# Função para calcular a média da velocidade dos ventos 'nortada' por mês
def average_velocity_nortada_per_month(df):
    df_nortada = df[df['nortada_MPI']]  # Filtrar apenas os dias com 'nortada'
    average_velocity_per_month = df_nortada.groupby(['ano', 'mês'])['velocity_daily_mean_MPI'].mean()
    
    # Calcular a média por mês
    average_velocity_per_month = average_velocity_per_month.groupby('mês').mean()
    
    return average_velocity_per_month


#%%
# Calcular a média da velocidade por mês para o período 1995-2014
average_velocity_MPI = average_velocity_nortada_per_month(df_filtered_MPI)

# Criar DataFrame para os resultados
df_average_velocity_per_month_1995_2014 = pd.DataFrame({
    'average_velocity_MPI': average_velocity_MPI
})

print("Média de velocidade dos ventos 'nortada' por mês (1995-2014):")
print(df_average_velocity_per_month_1995_2014)

# Plotar o gráfico de linha da média de velocidade por mês
import matplotlib.pyplot as plt

months = df_average_velocity_per_month_1995_2014.index  # Índices dos meses
average_velocity = df_average_velocity_per_month_1995_2014['average_velocity_MPI'].values  # Valores da velocidade média por mês

plt.figure(figsize=(10, 6))
plt.plot(months, average_velocity, marker='o', color='#03A9F4', linestyle='-', linewidth=2, markersize=8, label='WRF-MPI')

# Configurar o gráfico
plt.xticks(months, ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.xlabel('Mês')
plt.ylabel('Velocidade Média (m/s)')
plt.grid(True)
plt.legend(facecolor='white', fontsize=15)
plt.title('1995-2014', fontsize=17)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Validação/vel_nortada_medias_mensais.jpeg", bbox_inches='tight')



#%%
#
#
#
#
#
#                            CLIMA FUTURO (2046-2065 e 2081-2100)
#
#
# 
# 
# #
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
#        GROUPBY Nº DIAS C/ NORTADA E VELOCIDADE POR ANO (2081-2100)

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

# %%

# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#        GROUPBY Nº DIAS C/ NORTADA E VELOCIDADE POR ANO (2046-2065)

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
plt.plot(x, y_ssp245, marker='o', linewidth=2,  linestyle='-',  color='#03A9F4', label='SSP2-4.5')
plt.plot(x, y_ssp370, marker='s', linewidth=2,  linestyle='-',  color='#FF9800', label='SSP3-7.0')
plt.plot(x, y_ssp585, marker='^', linewidth=2,  linestyle='-',  color='#4CAF50', label='SSP5-8.5')


plt.xlabel('Mês')
plt.ylabel('Número de dias com nortada')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)
plt.title('2046-2065')
plt.legend(loc='upper right', facecolor='white', fontsize=10)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_médias_mensais_2046_2065.jpeg", bbox_inches='tight')



# %% NÚMERO DE DIAS COM NORTADA MÉDIAS MENSAIS (PERÍODO HISTÓRICO E 2046-2065)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dados históricos
months = df_average_nortada_per_month_1995_2014.index
average_nortada = df_average_nortada_per_month_1995_2014['average_nortada_MPI'].values

# Médias mensais dos cenários futuros (2046-2065)
y_ssp245 = df_average_nortada_per_month_2046_2065['ssp245'].values
y_ssp370 = df_average_nortada_per_month_2046_2065['ssp370'].values
y_ssp585 = df_average_nortada_per_month_2046_2065['ssp585'].values

# Calcular as diferenças
diff_ssp245 = y_ssp245 - average_nortada
diff_ssp370 = y_ssp370 - average_nortada
diff_ssp585 = y_ssp585 - average_nortada

# Organizar as diferenças em um DataFrame para facilitar a plotagem
df_differences = pd.DataFrame({
    'Month': months,
    'SSP2-4.5': diff_ssp245,
    'SSP3-7.0': diff_ssp370,
    'SSP5-8.5': diff_ssp585
})


sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
plt.plot(months, diff_ssp245, marker='o', linestyle='-', linewidth=2, markersize=8, label='SSP2-4.5', color='#03A9F4')
plt.plot(months, diff_ssp370, marker='s', linestyle='-', linewidth=2, markersize=8, label='SSP3-7.0', color='#FF9800')
plt.plot(months, diff_ssp585, marker='^', linestyle='-', linewidth=2, markersize=8, label='SSP5-8.5', color='#4CAF50')

# Configurações do gráfico
plt.yticks(fontsize=15)
plt.xticks(months, ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'], rotation=45, fontsize=15)
plt.xlabel('Mês', fontsize=18)
plt.ylabel('Diferença do número de dias com nortada', fontsize=18)
plt.legend(loc='upper right', facecolor='white', fontsize=15)
plt.grid(True)

# Exibir o gráfico
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Futuro_vs_Presente/num_dias_nortada_diferenças_médias_mensais_2046_2065.jpeg", bbox_inches='tight')




# Imprimir as diferenças para verificação
print(df_differences)


# MÉDIA POR ESTAÇÃO DIFERENÇAS NÚMERO MÉDIO DE DIAS COM NORTADA (1995-2014 e 2081-2100)

# Definir os meses e as estações
months = np.arange(1, 13)
estacoes = {
    'Inverno': [12, 1, 2],
    'Primavera': [3, 4, 5],
    'Verão': [6, 7, 8],
    'Outono': [9, 10, 11]
    
}

# Organizar as diferenças em um DataFrame
df_differences = pd.DataFrame({
    'Month': months,
    'SSP2-4.5': diff_ssp245,
    'SSP3-7.0': diff_ssp370,
    'SSP5-8.5': diff_ssp585
})

# Função para calcular a média por estação
def calcular_media_estacao(df, estacao):
    meses = estacoes[estacao]
    media_ssp245 = df.loc[df['Month'].isin(meses), 'SSP2-4.5'].mean()
    media_ssp370 = df.loc[df['Month'].isin(meses), 'SSP3-7.0'].mean()
    media_ssp585 = df.loc[df['Month'].isin(meses), 'SSP5-8.5'].mean()
    return round(media_ssp245, 2), round(media_ssp370, 2), round(media_ssp585, 2)

# Calcular as médias por estação
medias_estacoes = {estacao: calcular_media_estacao(df_differences, estacao) for estacao in estacoes}

# Exibir os resultados
df_medias_estacoes = pd.DataFrame(medias_estacoes, index=['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']).T
print(df_medias_estacoes)




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
y_ssp245_2081_2100 = df_average_nortada_per_month_2081_2100['ssp245']
y_ssp370_2081_2100 = df_average_nortada_per_month_2081_2100['ssp370']
y_ssp585_2081_2100 = df_average_nortada_per_month_2081_2100['ssp585']

plt.figure(figsize=(10, 6))
plt.plot(x, y_ssp245_2081_2100, marker='o', linewidth=2, linestyle='-', color="#03A9F4", label='SSP2-4.5')
plt.plot(x, y_ssp370_2081_2100, marker='s', linewidth=2, linestyle='-', color="#FF9800", label='SSP3-7.0')
plt.plot(x, y_ssp585_2081_2100, marker='^', linewidth=2, linestyle='-', color='#4CAF50', label='SSP5-8.5')


plt.xlabel('Mês')
plt.ylabel('Número de dias com Nortada')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)
plt.title('2081-2100')
plt.legend(loc='upper right', facecolor='white', fontsize=10)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/num_dias_nortada_médias_mensais_2081_2100.jpeg", bbox_inches='tight')




# %% DIFERENÇAS MÉDIAS MENSAIS - NÚMERO DE DIAS COM NORTADA (PERÍODO HISTÓRICO E 2081-2100)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Médias mensais dos cenários futuros (2081-2100)
y_ssp245_2081_2100 = df_average_nortada_per_month_2081_2100['ssp245'].values
y_ssp370_2081_2100 = df_average_nortada_per_month_2081_2100['ssp370'].values
y_ssp585_2081_2100 = df_average_nortada_per_month_2046_2065['ssp585'].values

# Calcular as diferenças
diff_ssp245_2081_2100 = y_ssp245_2081_2100 - average_nortada
diff_ssp370_2081_2100 = y_ssp370_2081_2100 - average_nortada
diff_ssp585_2081_2100 = y_ssp585_2081_2100 - average_nortada

# Organizar as diferenças em um DataFrame para facilitar a plotagem
df_differences = pd.DataFrame({
    'Month': months,
    'SSP2-4.5': diff_ssp245_2081_2100,
    'SSP3-7.0': diff_ssp370_2081_2100,
    'SSP5-8.5': diff_ssp585_2081_2100
})


sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
plt.plot(months, diff_ssp245_2081_2100, marker='o', linestyle='-', linewidth=2, markersize=8, label='SSP2-4.5', color='#03A9F4')
plt.plot(months, diff_ssp370_2081_2100, marker='s', linestyle='-', linewidth=2, markersize=8, label='SSP3-7.0', color='#FF9800')
plt.plot(months, diff_ssp585_2081_2100, marker='^', linestyle='-', linewidth=2, markersize=8, label='SSP5-8.5', color='#4CAF50')

# Configurações do gráfico
plt.xticks(months, ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'], rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Mês', fontsize=18)
plt.ylabel('Diferença do número de dias com nortada', fontsize=18)
plt.legend(loc='upper right', fontsize=15, facecolor='white')
plt.grid(True)

# Exibir o gráfico
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Futuro_vs_Presente/num_dias_nortada_diferenças_médias_mensais_2081_2100.jpeg", bbox_inches='tight')


# Imprimir as diferenças para verificação
print(df_differences)


# MÉDIA POR ESTAÇÃO DIFERENÇAS NÚMERO MÉDIO DE DIAS COM NORTADA (1995-2014 e 2081-2100)

# Definir os meses e as estações
months = np.arange(1, 13)
estacoes = {
    'Inverno': [12, 1, 2],
    'Primavera': [3, 4, 5],
    'Verão': [6, 7, 8],
    'Outono': [9, 10, 11]
    
}

# Organizar as diferenças em um DataFrame
df_differences = pd.DataFrame({
    'Month': months,
    'SSP2-4.5': diff_ssp245_2081_2100,
    'SSP3-7.0': diff_ssp370_2081_2100,
    'SSP5-8.5': diff_ssp585_2081_2100
})

# Função para calcular a média por estação
def calcular_media_estacao(df, estacao):
    meses = estacoes[estacao]
    media_ssp245 = df.loc[df['Month'].isin(meses), 'SSP2-4.5'].mean()
    media_ssp370 = df.loc[df['Month'].isin(meses), 'SSP3-7.0'].mean()
    media_ssp585 = df.loc[df['Month'].isin(meses), 'SSP5-8.5'].mean()
    return round(media_ssp245, 2), round(media_ssp370, 2), round(media_ssp585, 2)

# Calcular as médias por estação
medias_estacoes = {estacao: calcular_media_estacao(df_differences, estacao) for estacao in estacoes}

# Exibir os resultados
df_medias_estacoes = pd.DataFrame(medias_estacoes, index=['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']).T
print(df_medias_estacoes)



#%%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# # FUNÇÕES PARA CALCULAR A VELOCIDADE MÉDIA COM NORTADA 


# Função para calcular a média da velocidade dos ventos 'nortada' por mês (1995-2014)
def average_velocity_nortada_per_month(df):
    df_nortada = df[df['nortada_MPI']]  # Filtrar apenas os dias com 'nortada'
    average_velocity_per_month = df_nortada.groupby(['ano', 'mês'])['velocity_daily_mean_MPI'].mean()
    average_velocity_per_month = average_velocity_per_month.groupby('mês').mean()
    return average_velocity_per_month

# Calcular a média da velocidade por mês para o período 1995-2014
average_velocity_MPI_1995_2014 = average_velocity_nortada_per_month(df_filtered_MPI)

#%%

# Função para calcular a média da velocidade diária média por mês para dias com nortada (cenários futuros)
def average_velocity_nortada_per_month_future(df, scenario_name, period):
    nortada_column = f'nortada_MPI_{scenario_name}_{period}'
    velocity_column = f'velocity_daily_mean_MPI_{scenario_name}_{period}'
    df_nortada = df[df[nortada_column]]  # Filtrar apenas os dias com 'nortada'
    average_per_month = df_nortada.groupby(['mês'])[velocity_column].mean()
    return average_per_month

# Calcular a média da velocidade diária média por mês para SSP245, SSP370 e SSP585 (2046-2065)
average_velocity_MPI_ssp245_2046_2065 = average_velocity_nortada_per_month_future(df_filtered_MPI_ssp245_2046_2065, 'ssp245', '2046_2065')
average_velocity_MPI_ssp370_2046_2065 = average_velocity_nortada_per_month_future(df_filtered_MPI_ssp370_2046_2065, 'ssp370', '2046_2065')
average_velocity_MPI_ssp585_2046_2065 = average_velocity_nortada_per_month_future(df_filtered_MPI_ssp585_2046_2065, 'ssp585', '2046_2065')

# Calcular a média da velocidade diária média por mês para SSP245, SSP370 e SSP585 (2081-2100)
average_velocity_MPI_ssp245_2081_2100 = average_velocity_nortada_per_month_future(df_filtered_MPI_ssp245_2081_2100, 'ssp245', '2081_2100')
average_velocity_MPI_ssp370_2081_2100 = average_velocity_nortada_per_month_future(df_filtered_MPI_ssp370_2081_2100, 'ssp370', '2081_2100')
average_velocity_MPI_ssp585_2081_2100 = average_velocity_nortada_per_month_future(df_filtered_MPI_ssp585_2081_2100, 'ssp585', '2081_2100')


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         VELOCIDADE COM NORTADA MENSAL (2046-2065)


# Dados de exemplo (substitua pelos seus dados)
x = np.arange(1, 13)
y_ssp245 = average_velocity_MPI_ssp245_2046_2065
y_ssp370 = average_velocity_MPI_ssp370_2046_2065
y_ssp585 = average_velocity_MPI_ssp585_2046_2065

# Criar o plot de linha
plt.figure(figsize=(10, 6))

plt.plot(x, y_ssp245, marker='s', linewidth=2,  linestyle='-', color="#03A9F4", label='SSP2-4.5')
plt.plot(x, y_ssp370, marker='s', linewidth=2,  linestyle='-', color="#FF9800", label='SSP3-7.0')
plt.plot(x, y_ssp585, marker='s', linewidth=2,  linestyle='-', color='#4CAF50', label='SSP5-8.5')

plt.xlabel('Mês')
plt.ylabel('Velocidade média da nortada (m/s))')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)

plt.legend(loc='upper right', facecolor='white', fontsize=10)
plt.title('2046-2065')
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_médias_mensais_2046_2065.jpeg", bbox_inches='tight')


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         VELOCIDADE COM NORTADA MENSAL (2081-2100)


# Dados de exemplo (substitua pelos seus dados)
x = np.arange(1, 13)
y_ssp245 = average_velocity_MPI_ssp245_2081_2100
y_ssp370 = average_velocity_MPI_ssp370_2081_2100
y_ssp585 = average_velocity_MPI_ssp585_2081_2100

# Criar o plot de linha
plt.figure(figsize=(10, 6))

plt.plot(x, y_ssp245, marker='s', linewidth=2,  linestyle='-', color="#03A9F4", label='SSP2-4.5')
plt.plot(x, y_ssp370, marker='s', linewidth=2,  linestyle='-', color="#FF9800", label='SSP3-7.0')
plt.plot(x, y_ssp585, marker='s', linewidth=2,  linestyle='-', color='#4CAF50', label='SSP5-8.5')

plt.xlabel('Mês')
plt.ylabel('Velocidade média da nortada (m/s)')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)
plt.title('2081-2100')
plt.legend(loc='upper right', facecolor='white', fontsize=10)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Alterações_futuras/vel_nortada_médias_mensais_2081_2100.jpeg", bbox_inches='tight')



# %%  DIFERENÇAS VELOCIDADES MÉDIAS MENSAIS (1995-2014 e 2064-2065)

# Calcular as diferenças para cada cenário entre 2046-2065 e 1995-2014
diff_ssp245 = average_velocity_MPI_ssp245_2046_2065 - average_velocity_MPI_1995_2014.values
diff_ssp370 = average_velocity_MPI_ssp370_2046_2065 - average_velocity_MPI_1995_2014.values
diff_ssp585 = average_velocity_MPI_ssp585_2046_2065 - average_velocity_MPI_1995_2014.values


# Organizar as diferenças em um DataFrame para facilitar a plotagem
df_differences = pd.DataFrame({
    'Month': months,
    'SSP2-4.5': diff_ssp245,
    'SSP3-7.0': diff_ssp370,
    
    'SSP5-8.5': diff_ssp585
})


sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
plt.plot(months, diff_ssp245, marker='o', linestyle='-', linewidth=2, markersize=8, label='SSP2-4.5', color='#03A9F4')
plt.plot(months, diff_ssp370, marker='s', linestyle='-', linewidth=2, markersize=8, label='SSP3-7.0', color='#FF9800')
plt.plot(months, diff_ssp585, marker='^', linestyle='-', linewidth=2, markersize=8, label='SSP5-8.5', color='#4CAF50')


plt.xticks(months, ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'], rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Mês', fontsize=18)
plt.ylabel('Diferença da velocidade média da nortada (m/s)', fontsize=18)
plt.legend(loc='upper right', fontsize=15, facecolor='white')
plt.grid(True)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Futuro_vs_Presente/vel_nortada_diferenças_médias_mensais_2046_2065.jpeg", bbox_inches='tight')
print(df_differences)


# MÉDIA POR ESTAÇÃO DIFERENÇAS VELOCIDADES MÉDIAS MENSAIS (1995-2014 e 2046-2065)

# Definir os meses e as estações
months = np.arange(1, 13)
estacoes = {
    'Inverno': [12, 1, 2],
    'Primavera': [3, 4, 5],
    'Verão': [6, 7, 8],
    'Outono': [9, 10, 11]
    
}
# Organizar as diferenças em um DataFrame
df_differences = pd.DataFrame({
    'Month': months,
    'SSP2-4.5': diff_ssp245,
    'SSP3-7.0': diff_ssp370,
    'SSP5-8.5': diff_ssp585
})

# Função para calcular a média por estação
def calcular_media_estacao(df, estacao):
    meses = estacoes[estacao]
    media_ssp245 = df.loc[df['Month'].isin(meses), 'SSP2-4.5'].mean()
    media_ssp370 = df.loc[df['Month'].isin(meses), 'SSP3-7.0'].mean()
    media_ssp585 = df.loc[df['Month'].isin(meses), 'SSP5-8.5'].mean()
    return round(media_ssp245, 2), round(media_ssp370, 2), round(media_ssp585, 2)

# Calcular as médias por estação
medias_estacoes = {estacao: calcular_media_estacao(df_differences, estacao) for estacao in estacoes}

# Exibir os resultados
df_medias_estacoes = pd.DataFrame(medias_estacoes, index=['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']).T
print(df_medias_estacoes)

# %%  DIFERENÇAS VELOCIDADES MÉDIAS MENSAIS (1995-2014 e 2081-2100)


# Calcular as diferenças para cada cenário entre 2081-2100 e 1995-2014
diff_ssp245_2081_2100 = average_velocity_MPI_ssp245_2081_2100 - average_velocity_MPI_1995_2014.values
diff_ssp370_2081_2100 = average_velocity_MPI_ssp370_2081_2100 - average_velocity_MPI_1995_2014.values
diff_ssp585_2081_2100 = average_velocity_MPI_ssp585_2081_2100 - average_velocity_MPI_1995_2014.values


# Organizar as diferenças em um DataFrame para facilitar a plotagem
df_differences = pd.DataFrame({
    'Month': months,
    'SSP2-4.5': diff_ssp245_2081_2100,
    'SSP3-7.0': diff_ssp370_2081_2100,
    'SSP5-8.5': diff_ssp585_2081_2100
})


sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
plt.plot(months, diff_ssp245_2081_2100, marker='o', linestyle='-', linewidth=2, markersize=8, label='SSP2-4.5', color='#03A9F4')
plt.plot(months, diff_ssp370_2081_2100, marker='s', linestyle='-', linewidth=2, markersize=8, label='SSP3-7.0', color='#FF9800')
plt.plot(months, diff_ssp585_2081_2100, marker='^', linestyle='-', linewidth=2, markersize=8, label='SSP5-8.5', color='#4CAF50')


plt.xticks(months, ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'], rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Mês', fontsize=18)
plt.ylabel('Diferença da velocidade média da nortada (m/s)', fontsize=18)
plt.legend(loc='upper right', fontsize=15, facecolor='white')
plt.grid(True)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Futuro_vs_Presente/vel_nortada_diferenças_médias_mensais_2081_2100.jpeg", bbox_inches='tight')
print(df_differences)



# MÉDIA POR ESTAÇÃO DIFERENÇAS VELOCIDADES MÉDIAS MENSAIS (1995-2014 e 2081-2100)

# Definir os meses e as estações
months = np.arange(1, 13)
estacoes = {
    'Inverno': [12, 1, 2],
    'Primavera': [3, 4, 5],
    'Verão': [6, 7, 8],
    'Outono': [9, 10, 11]
    
}
# Organizar as diferenças em um DataFrame
df_differences = pd.DataFrame({
    'Month': months,
    'SSP2-4.5': diff_ssp245_2081_2100,
    'SSP3-7.0': diff_ssp370_2081_2100,
    'SSP5-8.5': diff_ssp585_2081_2100
})

# Função para calcular a média por estação
def calcular_media_estacao(df, estacao):
    meses = estacoes[estacao]
    media_ssp245 = df.loc[df['Month'].isin(meses), 'SSP2-4.5'].mean()
    media_ssp370 = df.loc[df['Month'].isin(meses), 'SSP3-7.0'].mean()
    media_ssp585 = df.loc[df['Month'].isin(meses), 'SSP5-8.5'].mean()
    return round(media_ssp245, 2), round(media_ssp370, 2), round(media_ssp585, 2)

# Calcular as médias por estação
medias_estacoes = {estacao: calcular_media_estacao(df_differences, estacao) for estacao in estacoes}

# Exibir os resultados
df_medias_estacoes = pd.DataFrame(medias_estacoes, index=['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']).T
print(df_medias_estacoes)



# %%
