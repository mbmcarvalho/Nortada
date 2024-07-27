% Limpar workspace, fechar todas as figuras e limpar o console
clear all; 
close all; 
clc;

% Nome do arquivo de dados
file_name_1 = 'ERA5_UV_gust_1994_2022.nc';  

% Leitura das variáveis
lon = ncread(file_name_1, 'longitude');  
lat = ncread(file_name_1, 'latitude');
dados_tempo = ncread(file_name_1, 'time');   % Horas desde 01-01-1900 00:00:00
u10 = ncread(file_name_1, 'u10');
v10 = ncread(file_name_1, 'v10');
rajada_max = ncread(file_name_1, 'i10fg'); 

% Convertendo tempo para formato de data
tempo = datevec(hours(dados_tempo) + datetime(1900, 1, 1));  % ano|mês|dia|hora|min|seg
ano = tempo(:, 1); dia = tempo(:, 3);

% Limites que abrange a Praia da Barra, entre praias à beira e um pouco do interior de Aveiro
lon_min = -8.75; lon_max = -8.50;
lat_min = 40.50; lat_max = 40.75;

lon_indices = find(lon >= lon_min & lon <= lon_max);
lat_indices = find(lat >= lat_min & lat <= lat_max);

% Dados dentro das coordenadas desejadas
u10_barra = u10(lon_indices, lat_indices, :);
v10_barra = v10(lon_indices, lat_indices, :);

% Calcular a direção e a intensidade do vento
direcao_vento = atan2(v10_barra, u10_barra) * 180 / pi + 180;
int_vento = sqrt(u10_barra.^2 + v10_barra.^2);

% Determinar se houve algum evento de nortada/dia, para não repetir dados num só dia
indices_nortada = (direcao_vento >= 335 | direcao_vento <= 25); 
indices_nortada_por_dia = squeeze(any(any(indices_nortada, 1), 2)); % Vetor de booleanos que indica se houve nortada nalgum momento do dia

% Calcular o campo médio da velocidade do vento à superfície para cada estação do ano (DJF, MAM, JJA, SON)
media_int_vento_DJF = mean(int_vento(:,:,ano>=1994 & ano<=2022 & (dia>=1 & dia<=60)), 3);
media_int_vento_MAM = mean(int_vento(:,:,ano>=1994 & ano<=2022 & (dia>=61 & dia<=151)), 3);
media_int_vento_JJA = mean(int_vento(:,:,ano>=1994 & ano<=2022 & (dia>=152 & dia<=243)), 3);
media_int_vento_SON = mean(int_vento(:,:,ano>=1994 & ano<=2022 & (dia>=244 & dia<=334)), 3);

% Calcular a frequência de ocorrência de Nortada para cada ponto da grade
frequencia_nortada = sum(indices_nortada, 3) / size(indices_nortada, 3);

% Calcular a velocidade média do vento quando ocorre ou não ocorre Nortada
velocidade_media_com_nortada = mean(int_vento(indices_nortada), 'all');
velocidade_media_sem_nortada = mean(int_vento(~indices_nortada), 'all');

% Visualização dos campos médios sazonais de velocidade do vento à superfície
figure;
subplot(2, 2, 1);
pcolor(lon(lon_indices), lat(lat_indices), media_int_vento_DJF');
title('DJF');
colorbar;

subplot(2, 2, 2);
pcolor(lon(lon_indices), lat(lat_indices), media_int_vento_MAM');
title('MAM');
colorbar;

subplot(2, 2, 3);
pcolor(lon(lon_indices), lat(lat_indices), media_int_vento_JJA');
title('JJA');
colorbar;

subplot(2, 2, 4);
pcolor(lon(lon_indices), lat(lat_indices), media_int_vento_SON');
title('SON');
colorbar;

% Visualização da frequência de ocorrência de Nortada
figure;
pcolor(lon(lon_indices), lat(lat_indices), frequencia_nortada');
title('Frequência de ocorrência de Nortada');
colorbar;

% Visualização da velocidade média do vento com e sem Nortada
figure;
bar([velocidade_media_com_nortada, velocidade_media_sem_nortada]);
xticklabels({'Com Nortada', 'Sem Nortada'});
ylabel('Velocidade média do vento');
title('Velocidade média do vento com e sem Nortada');
