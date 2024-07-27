%% MALHA COMPLETA - MAPAS

% Limpar workspace, fechar todas as figuras e limpar o console
clear all; 
close all; 
clc;

file_name_1 = 'ERA5_UV_gust_1994_2022.nc';  
addpath .\m_map\
load coastlines.mat

% Leitura das Variáveis
lon = ncread(file_name_1, 'longitude');  
lat = ncread(file_name_1, 'latitude');
dados_tempo = ncread(file_name_1, 'time');   % Horas desde 01-01-1900 00:00:00
u10 = ncread(file_name_1, 'u10');
v10 = ncread(file_name_1, 'v10');
rajada_max = ncread(file_name_1, 'i10fg'); 

%Dados horários de 1994-2022
tempo = datevec(hours(dados_tempo) + datetime(1900, 1, 1));  % ano|mês|dia|hora|min|seg
ano = tempo(:, 1); mes = tempo(:, 2); dia = tempo(:, 3); hora = tempo(:, 4);

% Vento de direção entre 25 e 335 graus
direcao_vento = atan2(v10, u10) * 180 / pi + 180;
nortada = (direcao_vento >= 335 | direcao_vento <= 25);

% Calcular média sazonal de velocidade do vento
media_int_vento_DJF = mean(sqrt(u10(:,:,mes>=12 | mes<=2).^2 + v10(:,:,mes>=12 | mes<=2).^2), 3);
media_int_vento_MAM = mean(sqrt(u10(:,:,mes>=3 & mes<=5).^2 + v10(:,:,mes>=3 & mes<=5).^2), 3);
media_int_vento_JJA = mean(sqrt(u10(:,:,mes>=6 & mes<=8).^2 + v10(:,:,mes>=6 & mes<=8).^2), 3);
media_int_vento_SON = mean(sqrt(u10(:,:,mes>=9 & mes<=11).^2 + v10(:,:,mes>=9 & mes<=11).^2), 3);

% Calcular a direção média do vento para cada estação do ano
mean_direction_DJF = mean(direcao_vento(:,:,ano>=1994 & ano<=2022 & (mes>=1 & mes<=2)), 3);
mean_direction_MAM = mean(direcao_vento(:,:,ano>=1994 & ano<=2022 & (mes>=3 & mes<=5)), 3);
mean_direction_JJA = mean(direcao_vento(:,:,ano>=1994 & ano<=2022 & (mes>=6 & mes<=8)), 3);
mean_direction_SON = mean(direcao_vento(:,:,ano>=1994 & ano<=2022 & (mes>=9 & mes<=11)), 3);

% Carregar os dados de linhas de costa
load coastlines.mat;

% Plotar os campos sazonais de velocidade do vento e direção média do vento
figure;

% Limites da região de interesse (área costeira de Portugal)
lon_min = min(lon(:));
lon_max = max(lon(:));
lat_min = min(lat(:));
lat_max = max(lat(:));

% Projeto do mapa
m_proj('mercator', 'lon', [lon_min lon_max], 'lat', [lat_min lat_max]);


% Plotar a grade de velocidade do vento para DJF
subplot(2, 2, 1);
m_pcolor(lon, lat, media_int_vento_DJF');
shading flat;
m_grid('linestyle', 'none', 'tickdir', 'out', 'linewidth', 1);
title('DJF');
colorbar;
% Adicionar setas de direção média do vento
hold on;
m_quiver(lon, lat, cosd(mean_direction_DJF'), sind(mean_direction_DJF'), 'k');
hold off;

% Adicionar linhas de costa ao subplot DJF
hold on;
m_plot(coastlon, coastlat, 'k', 'LineWidth', 1);
hold off;

% Plotar a grade de velocidade do vento para MAM
subplot(2, 2, 2);
m_pcolor(lon, lat, media_int_vento_MAM');
shading flat;
m_grid('linestyle', 'none', 'tickdir', 'out', 'linewidth', 1);
title('MAM');
colorbar;

% Adicionar linhas de costa ao subplot MAM
hold on;
m_plot(coastlon, coastlat, 'k', 'LineWidth', 1);
hold off;

% Plotar a grade de velocidade do vento para JJA
subplot(2, 2, 3);
m_pcolor(lon, lat, media_int_vento_JJA');
shading flat;
m_grid('linestyle', 'none', 'tickdir', 'out', 'linewidth', 1);
title('JJA');
colorbar;

% Adicionar linhas de costa ao subplot JJA
hold on;
m_plot(coastlon, coastlat, 'k', 'LineWidth', 1);
hold off;

% Plotar a grade de velocidade do vento para SON
subplot(2, 2, 4);
m_pcolor(lon, lat, media_int_vento_SON');
shading flat;
m_grid('linestyle', 'none', 'tickdir', 'out', 'linewidth', 1);
title('SON');
colorbar;

% Adicionar linhas de costa ao subplot SON
hold on;
m_plot(coastlon, coastlat, 'k', 'LineWidth', 1);
hold off;





