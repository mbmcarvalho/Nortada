#%%
import pandas as pd
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#%%
# Abrir o arquivo NetCDF
file_name = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\Dados_2\ficheiros\wrfout_ERA5_UV10m_12_18h.nc')

# Extrair variáveis
u10 = file_name.variables['U10'][:]
v10 = file_name.variables['V10'][:]
lon = file_name.variables['XLONG'][:]
lat = file_name.variables['XLAT'][:]
time = file_name.variables['XTIME'][:]

#%%
# diz-me as variáveis que existem no ficheiro "file_name"
print(file_name.variables.keys())

# diz-me tipo da variável, tipo de dados, dimensões, atritutos da variável, etc
print(file_name.variables['U10']) 

# %%
time_var = file_name.variables['XTIME']  
time_units = time_var.units  #hours since 1994-12-8 00:00:00

# Converter os valores de tempo para datas e horas
time_dates = nc.num2date(time_var[:], units=time_units)

#%%
for date in time_dates[:5]:  # Imprime as primeiras 5 datas como exemplo
    print(date)

#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#               MAPA DOMNÍNIO DO MODELO WRF

plt.figure(figsize=(10, 8))

# Definir as bordas do mapa
m = Basemap(llcrnrlon=lon.min(), llcrnrlat=lat.min(), urcrnrlon=lon.max(), urcrnrlat=lat.max(), resolution='l')
#llcrnr = lower left corner  e urcrnr = upper right corner
m.drawcoastlines() # contorno do continente
x, y = m(lon, lat)
m.scatter(x, y, marker='o', color='red')
plt.title('Domínio do Modelo WRF')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# %%

tabela_dados_ERA5 = pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/tabela_dados_ERA5.csv')
anual_era5 = pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/anual_ERA5.csv')

# Filtra-se os dados para obter apenas aqueles dias em que houve nortada 
# Cálculo da velocidade média diária dos dias com nortada.
dados_com_nortada_ERA5 = tabela_dados_ERA5[tabela_dados_ERA5['Nortada Médias Diárias'] == 1]
vel_media_nortada_ERA5 = dados_com_nortada_ERA5['Velocidade média diária']
tabela_dados_ERA5['Velocidade Média Nortada'] = vel_media_nortada_ERA5

# Agrupamento mensal
grouped_ERA5 = tabela_dados_ERA5.groupby(['Ano', 'Mês']).agg({
    'Nortada Médias Diárias': 'sum', 'Nortada Média (12h)': 'sum', 'Nortada Média (18h)': 'sum', 
    **{col: 'mean' for col in tabela_dados_ERA5.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)','Nortada Média (18h)']}
}).reset_index()

#Agrupamento anual
grouped_ano_ERA5 = tabela_dados_ERA5.groupby(['Ano']).agg({
    'Nortada Médias Diárias': 'sum', 'Nortada Média (12h)': 'sum', 'Nortada Média (18h)': 'sum', 'Velocidade Média Nortada' : 'mean',
    **{col: 'mean' for col in tabela_dados_ERA5.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)','Nortada Média (18h)', 'Velocidade Média Nortada']}
}).reset_index()

# Agrupamento sazonal
grouped_estacao_ERA5 = tabela_dados_ERA5.groupby(['Ano', 'Estações'], sort=False).agg({
    'Nortada Médias Diárias': 'sum', 
    'Nortada Média (12h)': 'sum', 
    'Nortada Média (18h)': 'sum', 
    'Direções médias diárias': 'mean', 
    **{col: 'mean' for col in tabela_dados_ERA5.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)', 'Direções médias diárias']}
}).reset_index()

# %%
import h5py
vento_Barra_wrfout_ERA5 = 'C:/Users/Beatriz/Desktop/Projeto/Dados_2/armazenamento_dados_1995_2014/vento_Barra_wrfout_ERA5.mat'
#wind_Barra_wrfoutERA5 = h5py.File(vento_Barra_wrfout_ERA5, 'r')

with h5py.File(vento_Barra_wrfout_ERA5, 'r') as wind_Barra_wrfoutERA5:
    u10_12h_Barra = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/u10_12h_Barra'][:]
    v10_12h_Barra = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/v10_12h_Barra'][:]
    u10_18h_Barra = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/u10_18h_Barra'][:]
    v10_18h_Barra = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/v10_18h_Barra'][:]
    vel_12h_Barra = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/vel_12h_Barra'][:]
    vel_18h_Barra = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/vel_18h_Barra'][:]
    media_direcoes_diarias_Barra = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/media_direcoes_diarias_Barra'][:]
    lon_BARRA = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/lon_BARRA'][:]
    lat_BARRA = wind_Barra_wrfoutERA5['vento_Barra_wrfout_ERA5/lat_BARRA'][:]

# %%

# imprime a estrutura do arquivo "vento_Barra_wrfout_ERA5"
def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

# Iterar pela estrutura do arquivo e imprimir
wind_Barra_wrfoutERA5.visititems(print_structure)

# %% CÁLCULO DA MÉDIA DIRECIONAL

import numpy as np

# Implementar a função circ_mean_degrees
def circ_mean_degrees(alpha, w=None, dim=0):
    if w is None:
        w = np.ones_like(alpha)
    if w.shape != alpha.shape:
        raise ValueError("Input dimensions do not match")

    # Compute weighted sum of cos and sin of angles
    r = np.sum(w * (np.cos(np.deg2rad(alpha)) + 1j * np.sin(np.deg2rad(alpha))), axis=dim)

    # Obtain mean by
    mu = np.angle(r, deg=True)
    mu = np.mod(mu, 360)  # Ensure the mean angle is between 0 and 360 degrees

    return mu

# Caminho para o arquivo .mat
vento_Barra_wrfout_ERA5 = 'C:/Users/Beatriz/Desktop/Projeto/Dados_2/armazenamento_dados_1995_2014/vento_Barra_wrfout_ERA5.mat'

# Calcular direções médias
direcao_media_12h = np.arctan2(np.rad2deg(v10_12h_Barra), np.rad2deg(u10_12h_Barra))
direcao_media_18h = np.arctan2(np.rad2deg(v10_18h_Barra), np.rad2deg(u10_18h_Barra))

# Velocidade média diária
vel_media_diaria = (vel_12h_Barra + vel_18h_Barra) / 2

# Direção média diária
direcao_media_diaria = (direcao_media_12h + direcao_media_18h) / 2

# Componentes u e v médios diários
u_media_diaria = np.zeros_like(u10_12h_Barra)
v_media_diaria = np.zeros_like(v10_12h_Barra)

# Iterar sobre os dias
for i in range(u10_12h_Barra.shape[2]):
    # Calcula a média da direção para cada dia utilizando a função circ_mean_degrees
    direcao_media_diaria[:, :, i] = circ_mean_degrees(direcao_media_diaria[:, :, i])

    u_media_diaria[:, :, i] = vel_media_diaria[:, :, i] * np.cos(np.deg2rad(direcao_media_diaria[:, :, i]))
    v_media_diaria[:, :, i] = vel_media_diaria[:, :, i] * np.sin(np.deg2rad(direcao_media_diaria[:, :, i]))

# Converter direção média diária para componentes u e v
u_media_diaria = vel_media_diaria * np.cos(np.deg2rad(direcao_media_diaria))
v_media_diaria = vel_media_diaria * np.sin(np.deg2rad(direcao_media_diaria))



# %%
