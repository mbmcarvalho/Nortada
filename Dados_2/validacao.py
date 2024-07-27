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
file_name_ERA5 = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\Dados_2\ficheiros\wrfout_ERA5_UV10m_12_18h.nc')
file_name_MPI = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\Dados_2\ficheiros\wrfout_MPI_hist_UV10m_12_18h.nc')

# Extração variáveis wrfout ERA5
u10_ERA5 = file_name_ERA5.variables['U10'][:]
v10_ERA5 = file_name_ERA5.variables['V10'][:]
lon_ERA5 = file_name_ERA5.variables['XLONG'][:]
lat_ERA5 = file_name_ERA5.variables['XLAT'][:]
time_ERA5 = file_name_ERA5.variables['XTIME'][:]

# Extração variáveis wrfout MPI
u10_MPI = file_name_MPI.variables['U10'][:]
v10_MPI = file_name_MPI.variables['V10'][:]
lon_MPI = file_name_MPI.variables['XLONG'][:]
lat_MPI = file_name_MPI.variables['XLAT'][:]
time_MPI = file_name_MPI.variables['XTIME'][:]


# %%

# Conversão de valores de tempo para data e hora 

# wrfout ERA5
time_var_ERA5 = file_name_ERA5.variables['XTIME']  
time_units_ERA5 = time_var_ERA5.units  #hours since 1994-12-8 00:00:00
time_dates_ERA5 = nc.num2date(time_var_ERA5[:], units=time_units_ERA5)
time_dates_py_ERA5 = [np.datetime64(date) for date in time_dates_ERA5]
time_index_ERA5 = pd.DatetimeIndex(time_dates_py_ERA5) #formato pandas DateTimeIndex

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

# wrfout ERA5
lon_flat_ERA5 = lon_ERA5.flatten() #Achatamento das matrizes lon e lat para vetores (os dados não desaparecem, só ficam unidimensionais)
lat_flat_ERA5 = lat_ERA5.flatten()
distances_ERA5 = np.sqrt((lon_flat_ERA5 - lon_barra[0])**2 + (lat_flat_ERA5 - lat_barra[0])**2) # Cálculo distância euclidiana para cada ponto
idx_nearest_point_ERA5 = np.argmin(distances_ERA5) #índice do ponto mais próximo de Barra
lon_nearest_ERA5 = lon_flat_ERA5[idx_nearest_point_ERA5]
lat_nearest_ERA5 = lat_flat_ERA5[idx_nearest_point_ERA5]

# wrfout MPI
lon_flat_MPI = lon_MPI.flatten() #Achatamento das matrizes lon e lat para vetores
lat_flat_MPI = lat_MPI.flatten()
distances_MPI = np.sqrt((lon_flat_MPI - lon_barra[0])**2 + (lat_flat_MPI - lat_barra[0])**2) # Cálculo distância euclidiana para cada ponto
idx_nearest_point_MPI = np.argmin(distances_MPI) #índice do ponto mais próximo de Barra
lon_nearest_MPI = lon_flat_MPI[idx_nearest_point_MPI]
lat_nearest_MPI = lat_flat_MPI[idx_nearest_point_MPI]


# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                    MAPA DO PONTO MAIS PRÓXIMO DA PRAIA DA BARRA

plt.figure(figsize=(10, 8))

m = Basemap(projection='merc', llcrnrlat=np.min(lat_ERA5), urcrnrlat=np.max(lat_ERA5),
            llcrnrlon=np.min(lon_ERA5), urcrnrlon=np.max(lon_ERA5), resolution='i')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# Converter coordenadas para coordenadas do mapa
x, y = m(lon_nearest_ERA5, lat_nearest_ERA5)

# Ponto mais próximo da praia da Barra
m.scatter(x, y, color='red', marker='o', s=100)
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/ponto_mais_próximo_praia_Barra.jpeg", dpi=600, quality=95, bbox_inches='tight', format='png')



# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   CÁLCULO DA DISTÂNCIA DO PONTO + PRÓXIMO DA PRAIA DA BARRA À COSTA
#              

from geopy.distance import distance

coords_nearest = (lat_nearest_ERA5, lon_nearest_ERA5) # Coordenadas do ponto mais próximo
coords_barra = [(lat_barra[0], lon_barra[0]), (lat_barra[1], lon_barra[1])]  # Coordenadas da Praia da Barra

# Cálculo da distância entre o ponto mais próximo e a costa (Praia da Barra)
dist_to_coast_km = min(distance(coords_nearest, coords_barra[0]).km,
                       distance(coords_nearest, coords_barra[1]).km)
print(f"A distância do ponto mais próximo da Praia da Barra à costa é aproximadamente {dist_to_coast_km:.2f} km.")



# %%  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           CÁLCULO DA INTENSIDADE DO VENTO (sqrt((u^2)+(v^2))


# wrfout ERA5

lon_index_ERA5, lat_index_ERA5 = np.unravel_index(idx_nearest_point_ERA5, lon_ERA5.shape) # Desfazer o achatamento para obter os índices originais
u10_var_ERA5 = u10_ERA5[:, lon_index_ERA5, lat_index_ERA5]
v10_var_ERA5 = v10_ERA5[:, lon_index_ERA5, lat_index_ERA5]

u10_daily_mean_ERA5 = []
v10_daily_mean_ERA5 = []
daily_dates_ERA5 = []
unique_days_ERA5 = np.unique(time_index_ERA5.date) # Iterar sobre os dias únicos para calcular as médias diárias
for day_ERA5 in unique_days_ERA5:

    day_indices_ERA5 = np.where(time_index_ERA5.date == day_ERA5)[0]  # Seleção dos índices correspondentes ao dia específico
    u10_day_mean_ERA5 = np.mean(u10_var_ERA5[day_indices_ERA5]) #média diária para u10 
    v10_day_mean_ERA5 = np.mean(v10_var_ERA5[day_indices_ERA5]) #média diária para v10
    u10_daily_mean_ERA5.append(u10_day_mean_ERA5) # Adicionar à lista de médias diárias
    v10_daily_mean_ERA5.append(v10_day_mean_ERA5)
    daily_dates_ERA5.append(pd.Timestamp(day_ERA5)) # Adicionar a data correspondente
    
u10_daily_mean_ERA5 = np.array(u10_daily_mean_ERA5) # Conversão das listas em arrays numpy para facilitar a manipulação dos dados
v10_daily_mean_ERA5 = np.array(v10_daily_mean_ERA5)



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

# wrfout ERA5

velocity_daily_mean_ERA5 = np.sqrt(u10_daily_mean_ERA5**2 + v10_daily_mean_ERA5**2)
media_velocidade_total_ERA5 = np.mean(velocity_daily_mean_ERA5)  # Calcular a média da velocidade média diária

print(f"A média da velocidade média diária ao longo do período (wrfout ERA5) é: {media_velocidade_total_ERA5} m/s")

# wrfout MPI

velocity_daily_mean_MPI = np.sqrt(u10_daily_mean_MPI**2 + v10_daily_mean_MPI**2)
media_velocidade_total_MPI = np.mean(velocity_daily_mean_MPI)  # Calcular a média da velocidade média diária

print(f"A média da velocidade média diária ao longo do período (wrfout MPI) é: {media_velocidade_total_MPI} m/s")

#%% 

# wrfout ERA5
data_ERA5 = {'Date': daily_dates_ERA5,'Velocity': velocity_daily_mean_ERA5} # DataFrame com as datas e as velocidades médias diárias
df_ERA5 = pd.DataFrame(data_ERA5)  #tabela
df_ERA5['Year'] = df_ERA5['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_ERA5 = df_ERA5.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade

# wrfout MPI
data_MPI = {'Date': daily_dates_MPI,'Velocity': velocity_daily_mean_MPI} # DataFrame com as datas e as velocidades médias diárias
df_MPI = pd.DataFrame(data_MPI)  #tabela
df_MPI['Year'] = df_MPI['Date'].dt.year # Extrair o ano de cada data
mean_velocity_by_year_MPI = df_MPI.groupby('Year')['Velocity'].mean()   # Calcular a média anual da velocidade

#%%
# Garantir que os DataFrames tenham os mesmos anos
common_years = mean_velocity_by_year_ERA5.index.intersection(mean_velocity_by_year_MPI.index)
mean_velocity_by_year_ERA5 = mean_velocity_by_year_ERA5.loc[common_years]
mean_velocity_by_year_MPI = mean_velocity_by_year_MPI.loc[common_years]

# Calcular o t-test
t_stat, p_value = ttest_ind(mean_velocity_by_year_ERA5, mean_velocity_by_year_MPI)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Há uma diferença estatisticamente significativa entre as médias das duas séries temporais.")
else:
    print("Não há uma diferença estatisticamente significativa entre as médias das duas séries temporais.")

# Calcular tendências (linhas de tendência)
years = common_years.values

# Tendência wrfout ERA5
coeffs_ERA5 = np.polyfit(years, mean_velocity_by_year_ERA5, 1)
trend_ERA5 = np.polyval(coeffs_ERA5, years)

# Tendência wrfout MPI
coeffs_MPI = np.polyfit(years, mean_velocity_by_year_MPI, 1)
trend_MPI = np.polyval(coeffs_MPI, years)





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
direction_daily_ERA5 = np.degrees(np.arctan2(v10_daily_mean_ERA5, u10_daily_mean_ERA5))
direction_daily_MPI = np.degrees(np.arctan2(v10_daily_mean_MPI, u10_daily_mean_MPI))

# Calcular as direções corrigidas para cada dia individualmente
media_dir_v10_ERA5 = np.zeros_like(direction_daily_ERA5)
ul_ERA5 = np.zeros_like(direction_daily_ERA5)
ll_ERA5 = np.zeros_like(direction_daily_ERA5)

for i in range(direction_daily_ERA5.shape[0]):
    media_dir_v10_ERA5[i], ul_ERA5[i], ll_ERA5[i] = circ_mean_degrees(direction_daily_ERA5[i])


# Calcular as direções corrigidas para cada dia individualmente para MPI
media_dir_v10_MPI = np.zeros_like(direction_daily_MPI)
ul_MPI = np.zeros_like(direction_daily_MPI)
ll_MPI = np.zeros_like(direction_daily_MPI)

for i in range(direction_daily_MPI.shape[0]):
    media_dir_v10_MPI[i], ul_MPI[i], ll_MPI[i] = circ_mean_degrees(direction_daily_MPI[i])

# Imprimir os resultados de um exemplo qualquer de dia
dia_exemplo = 0
print(f'Direção média corrigida para o dia {dia_exemplo + 1}: {media_dir_v10_ERA5[dia_exemplo]:.2f} graus')
print(f'Intervalo de confiança para o dia {dia_exemplo + 1}: [{ll_ERA5[dia_exemplo]:.2f}, {ul_ERA5[dia_exemplo]:.2f}] graus')

# Imprimir os resultados de um exemplo qualquer de dia para MPI
print(f'Direção média corrigida para o dia {dia_exemplo + 1} (MPI): {media_dir_v10_MPI[dia_exemplo]:.2f} graus')
print(f'Intervalo de confiança para o dia {dia_exemplo + 1} (MPI): [{ll_MPI[dia_exemplo]:.2f}, {ul_MPI[dia_exemplo]:.2f}] graus')


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                                  NORTADA

nortada_ERA5 = (media_dir_v10_ERA5 >= 335) | (media_dir_v10_ERA5 <= 25)
nortada_MPI = (media_dir_v10_MPI >= 335) | (media_dir_v10_MPI <= 25)


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           VELOCIDADE MÉDIA DOS DIAS COM NORTADA

velocity_nortada_ERA5 = velocity_daily_mean_ERA5[nortada_ERA5]
mean_velocity_nortada_ERA5 = np.mean(velocity_nortada_ERA5)
velocity_nortada_MPI = velocity_daily_mean_MPI[nortada_MPI]
mean_velocity_nortada_MPI = np.mean(velocity_nortada_MPI)

print(f"A velocidade média nos dias com nortada (ERA5) é: {mean_velocity_nortada_ERA5} m/s")
print(f"A velocidade média nos dias com nortada (MPI) é: {mean_velocity_nortada_MPI} m/s")


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                             DATA FRAMES


# Criando o DataFrame para wrfout ERA5
data_ERA5 = {
    'Date': daily_dates_ERA5,
    'U10_mean': u10_daily_mean_ERA5,
    'V10_mean': v10_daily_mean_ERA5,
    'Velocity_mean': velocity_daily_mean_ERA5,
    'Direction_mean': media_dir_v10_ERA5,
    'CI_lower': ll_ERA5,
    'CI_upper': ul_ERA5,
    'Nortada': nortada_ERA5
}
df_ERA5 = pd.DataFrame(data_ERA5)

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

print("Dados do wrfout ERA5:")
print(df_ERA5.head())
print("\nDados do wrfout MPI:")
print(df_MPI.head())


# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   DATA FRAMES: vel_média|dir_média|nortada_diária|data|ano|mês|dia

df_filtered_ERA5 = pd.DataFrame({
    'velocity_daily_mean_ERA5': velocity_daily_mean_ERA5,
    'media_dir_v10_ERA5': media_dir_v10_ERA5,
    'nortada_ERA5': nortada_ERA5,
    'Date': daily_dates_ERA5
})
df_filtered_ERA5['ano'] = df_filtered_ERA5['Date'].dt.year
df_filtered_ERA5['mês'] = df_filtered_ERA5['Date'].dt.month
df_filtered_ERA5['dia'] = df_filtered_ERA5['Date'].dt.day


df_filtered_MPI = pd.DataFrame({
    'velocity_daily_mean_MPI': velocity_daily_mean_MPI,
    'media_dir_v10_MPI': media_dir_v10_MPI,
    'nortada_MPI': nortada_MPI,
    'Date': daily_dates_MPI
})
df_filtered_MPI['ano'] = df_filtered_MPI['Date'].dt.year
df_filtered_MPI['mês'] = df_filtered_MPI['Date'].dt.month
df_filtered_MPI['dia'] = df_filtered_MPI['Date'].dt.day


print("DataFrame df_filtered_ERA5:")
print(df_filtered_ERA5)
print("\nDataFrame df_filtered_MPI:")
print(df_filtered_MPI)




# %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#        GROUPBY Nº DIAS C/ NORTADA E VELOCIDADE POR ANO

nortada_count_ERA5 = df_filtered_ERA5[df_filtered_ERA5['nortada_ERA5']].groupby('ano').size()
nortada_count_MPI = df_filtered_MPI[df_filtered_MPI['nortada_MPI']].groupby('ano').size()
mean_velocity_nortada_ERA5 = df_filtered_ERA5[df_filtered_ERA5['nortada_ERA5']].groupby('ano')['velocity_daily_mean_ERA5'].mean()
mean_velocity_nortada_MPI = df_filtered_MPI[df_filtered_MPI['nortada_MPI']].groupby('ano')['velocity_daily_mean_MPI'].mean()

# Criar DataFrame para os resultados
df_nortada_count = pd.DataFrame({
    'nortada_count_ERA5': nortada_count_ERA5,
    'nortada_count_MPI': nortada_count_MPI,
    'mean_velocity_nortada_ERA5': mean_velocity_nortada_ERA5,
    'mean_velocity_nortada_MPI': mean_velocity_nortada_MPI
})

# Preencher valores NaN com zero (caso não haja nortadas em algum ano)
df_nortada_count = df_nortada_count.fillna(0)

print("Número de dias com nortada e média da velocidade por ano:")
print(df_nortada_count)


#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#               NÚMERO DE DIAS COM NORTADA POR ANO (BARRA)

import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp
from matplotlib.lines import Line2D


anos = df_nortada_count.index  # Anos são os índices do DataFrame
nortada_count_ERA5 = df_nortada_count['nortada_count_ERA5']
nortada_count_MPI = df_nortada_count['nortada_count_MPI']

# Calcular tendências e coeficientes de determinação (R^2)
coefficients_ERA5 = np.polyfit(anos, nortada_count_ERA5, 1)
polynomial_ERA5 = np.poly1d(coefficients_ERA5)
trend_line_ERA5 = polynomial_ERA5(anos)
a_ERA5 = coefficients_ERA5[0]
b_ERA5 = coefficients_ERA5[1]
r_squared_ERA5 = r2_score(nortada_count_ERA5, trend_line_ERA5)

coefficients_MPI = np.polyfit(anos, nortada_count_MPI, 1)
polynomial_MPI = np.poly1d(coefficients_MPI)
trend_line_MPI = polynomial_MPI(anos)
a_MPI = coefficients_MPI[0]
b_MPI = coefficients_MPI[1]
r_squared_MPI = r2_score(nortada_count_MPI, trend_line_MPI)

# Teste KS para verificar a significância estatística entre os conjuntos de dados
ks_statistic, p_value = ks_2samp(nortada_count_ERA5, nortada_count_MPI)

# Imprimir o resultado do teste KS
print("Estatística KS:", ks_statistic)
print("Valor p:", p_value)
if p_value < 0.05:
    print("A diferença entre os conjuntos de dados é estatisticamente significativa.")
else:
    print("Não há diferença estatisticamente significativa entre os conjuntos de dados.")

# Configurações do gráfico com seaborn e matplotlib
sns.set(style="darkgrid")
plt.figure(figsize=(18, 8))

# Plotar linhas para os dados de ERA5 e MPI
sns.lineplot(data=df_nortada_count, x=anos, y='nortada_count_ERA5', marker="s",  markersize=12, color="#191970", label="WRF-ERA5", linewidth=2.5)
sns.lineplot(data=df_nortada_count, x=anos, y='nortada_count_MPI', marker="s",  markersize=12, color="#03A9F4", label="WRF-MPI", linewidth=2.5)

# Adicionar linhas de tendência
plt.plot(anos, trend_line_ERA5, color='#191970', linestyle='--', linewidth=2.5, label=f'Tendência ERA5: y={a_ERA5:.3f}x{"+" if b_ERA5 > 0 else ""}{b_ERA5:.3f}, $R^2$={r_squared_ERA5:.3f}')
plt.plot(anos, trend_line_MPI, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Tendência MPI: y={a_MPI:.3f}x{"+" if b_MPI > 0 else ""}{b_MPI:.3f}, $R^2$={r_squared_MPI:.3f}')

# Adicionar legenda customizada
legend_text = f'Significância estatística: \n ks-statistic = {ks_statistic:.3f}, p-value = {p_value:.3f}'
handles = [
    Line2D([0], [0], color="#191970", marker='s', linestyle='-', markersize=12, label="WRF-ERA5"),
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="WRF-MPI"),
    Line2D([0], [0], color="#191970", linestyle='--', linewidth=2.5, label=f'Tendência ERA5: y={a_ERA5:.2f}x{"+" if b_ERA5 > 0 else ""}{b_ERA5:.2f}, $R^2$={r_squared_ERA5:.2f}'),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Tendência MPI: y={a_MPI:.2f}x{"+" if b_MPI > 0 else ""}{b_MPI:.2f}, $R^2$={r_squared_MPI:.2f}')
]

plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], labels=["WRF-ERA5","WRF-MPI",f'Regressão linear WRF-ERA5: \n y={a_ERA5:.3f}x{"+" if b_ERA5 > 0 else ""}{b_ERA5:.3f}, $R^2$={r_squared_ERA5:.3f}',f'Regressão linear WRF-MPI: \n y={a_MPI:.3f}x{"+" if b_MPI > 0 else ""}{b_MPI:.3f}, $R^2$={r_squared_MPI:.3f}',legend_text],loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)


plt.xticks(np.arange(1995, 2015, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Validação/num_dias_nortada_por_ano_wrfout.jpeg", bbox_inches='tight')


#%%   MÉDIA NÚMERO DE DIAS COM NORTADA POR ANO - PERÍODO COMPLETO
# Extrair os dados de nortada_count_ERA5 e nortada_count_MPI
nortada_count_ERA5 = df_nortada_count['nortada_count_ERA5']
nortada_count_MPI = df_nortada_count['nortada_count_MPI']

# Calcular a média
media_ERA5 = np.mean(nortada_count_ERA5)
media_MPI = np.mean(nortada_count_MPI)

print(f"Média de dias com nortada por ano (ERA5): {media_ERA5:.0f}")
print(f"Média de dias com nortada por ano (MPI): {media_MPI:.0f}")

#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#            VELOCIDADE MÉDIA DO Nº DIAS COM NORTADA POR ANO (BARRA)


anos = df_nortada_count.index  # Anos são os índices do DataFrame
mean_velocity_nortada_ERA5 = df_nortada_count['mean_velocity_nortada_ERA5']
mean_velocity_nortada_MPI = df_nortada_count['mean_velocity_nortada_MPI']

# Estatística de teste KS e valor p (exemplo com dados aleatórios)
ks_statistic_vel, p_value_vel = ks_2samp(mean_velocity_nortada_ERA5, mean_velocity_nortada_MPI)
print("Estatística de teste t:", ks_statistic_vel)
print("Valor p:", p_value_vel)

# Regressão linear e cálculo de R^2 (exemplo com dados aleatórios)
coefficients_vel_ERA5 = np.polyfit(anos, mean_velocity_nortada_ERA5, 1)
polynomial_vel_ERA5 = np.poly1d(coefficients_vel_ERA5)
trend_line_vel_ERA5 = polynomial_vel_ERA5(anos)

coefficients_vel_MPI = np.polyfit(anos, mean_velocity_nortada_MPI, 1)
polynomial_vel_MPI = np.poly1d(coefficients_vel_MPI)
trend_line_vel_MPI = polynomial_vel_MPI(anos)

a_vel_ERA5, b_vel_ERA5 = coefficients_vel_ERA5
a_vel_MPI, b_vel_MPI = coefficients_vel_MPI

r_squared_vel_ERA5 = r2_score(mean_velocity_nortada_ERA5, trend_line_vel_ERA5)
r_squared_vel_MPI = r2_score(mean_velocity_nortada_MPI, trend_line_vel_MPI)


plt.figure(figsize=(18, 8))
sns.set_theme(font_scale=1.7)

sns.lineplot(x=anos, y=mean_velocity_nortada_ERA5, marker="s", markersize=12, color="#191970", label="WRF-ERA5", linewidth=2.5)
sns.lineplot(x=anos, y=mean_velocity_nortada_MPI, marker="s",  markersize=12, color="#03A9F4", label="WRF-MPI", linewidth=2.5)

plt.plot(anos, trend_line_vel_ERA5, color='#191970', linestyle='--', linewidth=2.5, label=f'Regressão linear WRF-ERA5:\n y={a_vel_ERA5:.3f}x{"+" if b_vel_ERA5 > 0 else ""}{b_vel_ERA5:.3f}, $R^2$={r_squared_vel_ERA5:.3f}')
plt.plot(anos, trend_line_vel_MPI, color='#03A9F4', linestyle='--', linewidth=2.5, label=f'Regressão linear WRF-MPI:\n y={a_vel_MPI:.3f}x{"+" if b_vel_MPI > 0 else ""}{b_vel_MPI:.3f}, $R^2$={r_squared_vel_MPI:.3f}')

plt.xlabel('Ano', fontsize=25)
plt.ylabel('Velocidade média da nortada (m/s)', fontsize=25)
plt.xticks(anos, rotation=45, fontsize=18)
plt.yticks(fontsize=22)

legend_text = f"Significância estatística: \n ks-statistic = {ks_statistic_vel:.3f}, p-value = {p_value_vel:.3f}"

# Criação dos handles para a legenda
handles = [
    Line2D([0], [0], color="#191970", marker='s', linestyle='-', markersize=12,  label="WRF-ERA5"),
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=12,  label="WRF-MPI"),
    Line2D([0], [0], color="#191970", linestyle='--', linewidth=2.5, label=f'Regressão linear WRF-ERA5: \n y={a_vel_ERA5:.3f}x{"+" if b_vel_ERA5 > 0 else ""}{b_vel_ERA5:.3f}, $R^2$={r_squared_vel_ERA5:.3f}'),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2.5, label=f'Regressão linear WRF-MPI: \n y={a_vel_MPI:.3f}x{"+" if b_vel_MPI > 0 else ""}{b_vel_MPI:.3f}, $R^2$={r_squared_vel_MPI:.3f}')
]

# Adição da legenda com o texto de significância estatística
plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=[handle.get_label() for handle in handles] + [legend_text],
           loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)

plt.tight_layout()
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Validação/vel_nortada_por_ano_wrfout.jpeg", bbox_inches='tight')



#%%


# Extrair os dados de mean_velocity_nortada_ERA5 e mean_velocity_nortada_MPI
mean_velocity_nortada_ERA5 = df_nortada_count['mean_velocity_nortada_ERA5']
mean_velocity_nortada_MPI = df_nortada_count['mean_velocity_nortada_MPI']

# Calcular a média
media_vel_ERA5 = np.mean(mean_velocity_nortada_ERA5)
media_vel_MPI = np.mean(mean_velocity_nortada_MPI)

print(f"Média da velocidade média com nortada por ano (ERA5): {media_vel_ERA5:.2f} m/s")
print(f"Média da velocidade média com nortada por ano (MPI): {media_vel_MPI:.2f} m/s")

#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#           GROUBPY - Data|nortadaERA5(0 ou 1)|velERA5|DirERA5
#           GROUBPY - Data|nortadaMPI(0 ou 1)|velERA5|DirMPI

nortada_count_ERA5 = df_filtered_ERA5.groupby('Date')['nortada_ERA5'].sum()
velocity_mean_ERA5 = df_filtered_ERA5.groupby('Date')['velocity_daily_mean_ERA5'].mean()
direction_mean_ERA5 = df_filtered_ERA5.groupby('Date')['media_dir_v10_ERA5'].mean()
nortada_count_MPI = df_filtered_MPI.groupby('Date')['nortada_MPI'].sum()
velocity_mean_MPI = df_filtered_MPI.groupby('Date')['velocity_daily_mean_MPI'].mean()
direction_mean_MPI = df_filtered_MPI.groupby('Date')['media_dir_v10_MPI'].mean()

df_ERA5 = pd.DataFrame({
    'nortada_ERA5': nortada_count_ERA5,
    'velocity_mean_ERA5': velocity_mean_ERA5,
    'direction_mean_ERA5': direction_mean_ERA5
})

df_MPI = pd.DataFrame({
    'nortada_MPI': nortada_count_MPI,
    'velocity_mean_MPI': velocity_mean_MPI,
    'direction_mean_MPI': direction_mean_MPI
})

df_metrics = pd.merge(df_ERA5, df_MPI, left_index=True, right_index=True, how='outer')

# Preencher valores NaN com zero (caso não haja nortada em algum dia)
df_metrics = df_metrics.fillna(0)

df_metrics['Year'] = df_metrics.index.year
df_metrics['Month'] = df_metrics.index.month
df_metrics['Day'] = df_metrics.index.day

print("Tabela combinada com métricas de nortada, velocidade média e direção média por dia:")
print(df_metrics)


#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                         GROUPBY ESTAÇÕES

def get_season(month):
    if month in [3, 4, 5]:
        return 'Primavera'
    elif month in [6, 7, 8]:
        return 'Verão'
    elif month in [9, 10, 11]:
        return 'Outono'
    else:
        return 'Inverno'
    

df_metrics['Season'] = df_metrics['Month'].apply(get_season)

nortada_days_by_season_ERA5 = df_metrics[df_metrics['nortada_ERA5'] > 0].groupby(['Season', 'Year'])['nortada_ERA5'].count()
nortada_days_by_season_MPI = df_metrics[df_metrics['nortada_MPI'] > 0].groupby(['Season', 'Year'])['nortada_MPI'].count()

velocity_mean_ERA5_by_season = df_metrics.groupby(['Season', 'Year'])['velocity_mean_ERA5'].mean()
velocity_mean_MPI_by_season = df_metrics.groupby(['Season', 'Year'])['velocity_mean_MPI'].mean()

nortada_counts_ERA5 = pd.DataFrame(nortada_days_by_season_ERA5)
nortada_counts_MPI = pd.DataFrame(nortada_days_by_season_MPI)
velocity_means_ERA5 = pd.DataFrame(velocity_mean_ERA5_by_season)
velocity_means_MPI = pd.DataFrame(velocity_mean_MPI_by_season)

nortada_counts_ERA5.columns = ['nortada_count_ERA5']
nortada_counts_MPI.columns = ['nortada_count_MPI']
velocity_means_ERA5.columns = ['velocity_mean_ERA5']
velocity_means_MPI.columns = ['velocity_mean_MPI']

nortada_counts_combined = pd.concat([nortada_counts_ERA5, nortada_counts_MPI], axis=1)
velocity_means_combined = pd.concat([velocity_means_ERA5, velocity_means_MPI], axis=1)

metrics_by_season = pd.concat([nortada_counts_combined, velocity_means_combined], axis=1)

print("Métricas por estação do ano:")
print(metrics_by_season)




#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#               NÚMERO DE DIAS COM NORTADA POR ANO (BARRA)
#                               ESTAÇÕES

from matplotlib.ticker import MaxNLocator

df_metrics = pd.DataFrame(metrics_by_season)

# Calcular média de dias com nortada por estação
metrics_by_season = df_metrics.groupby(['Season', 'Year']).mean()

# Configurações do gráfico com seaborn e matplotlib
sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(44, 35), sharex=True)

# Coordenadas para cada legenda
legend_positions = {
    "Inverno": (0.17, 0.875),  # (left/right, top/bottom)
    "Primavera": (0.17, 0.875),
    "Verão": (0.17, 0.88),
    "Outono": (0.825, 0.875)
}

# Ordenar as estações
season_order = ["Inverno", "Primavera", "Verão", "Outono"]

for i, season in enumerate(season_order):
    ax = axs[i // 2, i % 2]  # Determina a posição do subplot na matriz 2x2

    # Dados da estação atual
    season_data = metrics_by_season.loc[season]

    # Plotar linhas para os dados de ERA5 e MPI
    sns.lineplot(data=season_data, x='Year', y='nortada_count_ERA5', color="#191970", label="WRF-ERA5", ax=ax, marker="s", markersize=18, linewidth=4)
    sns.lineplot(data=season_data, x='Year', y='nortada_count_MPI', color="#03A9F4", label="WRF-MPI", ax=ax, marker="s", markersize=18, linewidth=4)

    # Regressão linear e cálculo de R^2 para ERA5
    coefficients_ERA5 = np.polyfit(season_data.index, season_data['nortada_count_ERA5'], 1)
    polynomial_ERA5 = np.poly1d(coefficients_ERA5)
    trend_line_ERA5 = polynomial_ERA5(season_data.index)
    r_squared_ERA5 = r2_score(season_data['nortada_count_ERA5'], trend_line_ERA5)
    ax.plot(season_data.index, trend_line_ERA5, color='#191970', linestyle='--', linewidth=4, label=f'Regressão linear WRF-ERA5: \n y={coefficients_ERA5[0]:.3f}x{coefficients_ERA5[1]:+.3f}, $R^2$={r_squared_ERA5:.3f}')

    # Regressão linear e cálculo de R^2 para MPI
    coefficients_MPI = np.polyfit(season_data.index, season_data['nortada_count_MPI'], 1)
    polynomial_MPI = np.poly1d(coefficients_MPI)
    trend_line_MPI = polynomial_MPI(season_data.index)
    r_squared_MPI = r2_score(season_data['nortada_count_MPI'], trend_line_MPI)
    ax.plot(season_data.index, trend_line_MPI, color='#03A9F4', linestyle='--', linewidth=4, label=f'Regressão linear WRF-MPI: \n y={coefficients_MPI[0]:.3f}x{coefficients_MPI[1]:+.3f}, $R^2$={r_squared_MPI:.3f}')

    # Teste KS para verificar a consistência das distribuições dos dados entre as simulações
    KS_statistic, p_value = ks_2samp(season_data['nortada_count_ERA5'], season_data['nortada_count_MPI'])

    # Adicionar legenda com significância estatística
    legend_text = f'ks-statistic={KS_statistic:.3f}, p-value={p_value:.3f}'
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
              labels=labels + [legend_text],
              loc='center', bbox_to_anchor=legend_positions[season], facecolor='white', framealpha=1,  fontsize=23.5)
    
    ax.set_xticks(np.arange(1995, 2015, 1))
    ax.tick_params(axis='x', rotation=45, labelsize=35)
    ax.tick_params(axis='y', labelsize=35)
    
    # Título do subplot
    ax.set_title(f'{season}', fontsize=40)

    # Ajustar rótulos dos eixos
    ax.set_xlabel("Ano", fontsize=38)
    ax.set_ylabel('Número de dias com nortada', fontsize=38)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Ajustar layout geral e mostrar o gráfico
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.98])
plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Validação/num_dias_nortada_seasons_wrfout.jpeg")

#, dpi=600, quality=95, bbox_inches='tight', format='png'

#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#               VELOCIDADE MÉDIA COM NORTADA POR ANO (BARRA)
#                               ESTAÇÕES


df_metrics = pd.DataFrame(metrics_by_season)

# Calcular média de velocidade média com nortada por estação e por ano
metrics_by_season = df_metrics.groupby(['Season', 'Year']).mean().reset_index()

# Configurações do gráfico com seaborn e matplotlib
sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(44, 35), sharex=True)

# Coordenadas para cada legenda
legend_positions = {
    "Inverno": (0.17, 0.88),  # (left/right, top/bottom)
    "Primavera": (0.38, 0.13),
    "Verão": (0.17, 0.13),
    "Outono": (0.17, 0.13)
}

# Ordenar as estações
season_order = ["Inverno", "Primavera", "Verão", "Outono"]

for i, season in enumerate(season_order):
    ax = axs[i // 2, i % 2]  # Determina a posição do subplot na matriz 2x2

    # Dados da estação atual
    season_data = metrics_by_season[metrics_by_season['Season'] == season]

    # Plotar linhas para os dados de ERA5 e MPI
    sns.lineplot(data=season_data, x='Year', y='velocity_mean_ERA5', color="#191970",  label="WRF-ERA5", ax=ax, marker="s", markersize=18, linewidth=4)
    sns.lineplot(data=season_data, x='Year', y='velocity_mean_MPI', color="#03A9F4", label="WRF-MPI", ax=ax, marker="s", markersize=18, linewidth=4)

    # Regressão linear e cálculo de R^2 para ERA5
    coefficients_ERA5 = np.polyfit(season_data['Year'], season_data['velocity_mean_ERA5'], 1)
    polynomial_ERA5 = np.poly1d(coefficients_ERA5)
    trend_line_ERA5 = polynomial_ERA5(season_data['Year'])
    r_squared_ERA5 = r2_score(season_data['velocity_mean_ERA5'], trend_line_ERA5)
    ax.plot(season_data['Year'], trend_line_ERA5, color='#191970', linestyle='--', linewidth=4, label=f'Regressão linear WRF-ERA5: \n y={coefficients_ERA5[0]:.3f}x{coefficients_ERA5[1]:+.3f}, $R^2$={r_squared_ERA5:.3f}')

    # Regressão linear e cálculo de R^2 para MPI
    coefficients_MPI = np.polyfit(season_data['Year'], season_data['velocity_mean_MPI'], 1)
    polynomial_MPI = np.poly1d(coefficients_MPI)
    trend_line_MPI = polynomial_MPI(season_data['Year'])
    r_squared_MPI = r2_score(season_data['velocity_mean_MPI'], trend_line_MPI)
    ax.plot(season_data['Year'], trend_line_MPI, color='#03A9F4', linestyle='--', linewidth=4, label=f'Regressão linear WRF-MPI: \n y={coefficients_MPI[0]:.3f}x{coefficients_MPI[1]:+.3f}, $R^2$={r_squared_MPI:.3f}')

    # Teste KS para verificar a significância estatística
    ks_statistic, p_value_ks = ks_2samp(season_data['velocity_mean_ERA5'], season_data['velocity_mean_MPI'])

    # Adicionar legenda com significância estatística
    legend_text = f'ks-statistic={ks_statistic:.3f}, p-value={p_value_ks:.3f}'
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')],
              labels=labels + [legend_text],
              loc='center', bbox_to_anchor=legend_positions[season], facecolor='white', framealpha=1,  fontsize=23.5)

    # Título do subplot
    ax.set_title(f'{season}', fontsize=40)

    # Ajustar rótulos dos eixos
    ax.set_xlabel("Ano", fontsize=38)
    ax.set_ylabel('Velocidade média da nortada (m/s)', fontsize=38)

    # Configurações adicionais de formatação dos eixos
    ax.set_xticks(np.arange(1995, 2015, 1))
    ax.tick_params(axis='x', rotation=45, labelsize=35)
    ax.tick_params(axis='y', labelsize=35)

# Ajustar layout geral e mostrar o gráfico
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.98])

plt.savefig("C:/Users/Beatriz/Desktop/Projeto/plots/Validação/vel_nortada_por_ano_seasons.jpeg")



#, dpi=600, quality=95, bbox_inches='tight', format='png'

# %%



