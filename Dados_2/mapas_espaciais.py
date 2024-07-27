
#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#              Mapas espaciais

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#%%
file_name_ERA5 = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\Dados_2\ficheiros\wrfout_ERA5_UV10m_12_18h.nc')
file_name_MPI = nc.Dataset(r'C:\Users\Beatriz\Desktop\Projeto\Dados_2\ficheiros\wrfout_MPI_hist_UV10m_12_18h.nc')

#%%
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
time_MPI = file_name_MPI.variables['XTIME'][:]# %%

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



# %%

def plot_wind_map(lon, lat, u10, v10, title):
    fig = plt.figure(figsize=(10, 8))
    m = Basemap(projection='mill', llcrnrlat=np.min(lat), urcrnrlat=np.max(lat),
                llcrnrlon=np.min(lon), urcrnrlon=np.max(lon), resolution='l')
    m.drawcoastlines()
    m.drawcountries()

    x, y = m(lon, lat)
    magnitude = np.sqrt(u10**2 + v10**2)

    # Ajuste da escala das setas de acordo com a magnitude do vento
    # Scale determina o tamanho das setas
    scale = 200  # Ajuste o valor conforme necessário para melhor visualização
    quiver = m.quiver(x, y, u10, v10, magnitude, scale=scale, cmap='viridis', pivot='middle')

    cbar = m.colorbar(quiver, location='right', pad='5%')
    cbar.set_label('Velocidade do Vento (m/s)')

    plt.title(title)
    plt.show()

# Exemplo de uso
plot_wind_map(lon_ERA5, lat_ERA5, u10_ERA5[0], v10_ERA5[0], 'Mapa de Vento com Basemap')