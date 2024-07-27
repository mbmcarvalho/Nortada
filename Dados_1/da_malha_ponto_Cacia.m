clear all; close all; clc;

file_name_1 = 'ERA5_UV_gust_1994_2022.nc'; % Nome do arquivo NetCDF
lon_desejado = -8.5985; % Longitude desejada
lat_desejada = 40.6946; % Latitude desejada

% Leitura das Variáveis
lon = ncread(file_name_1, 'longitude');
lat = ncread(file_name_1, 'latitude');
dados_tempo = ncread(file_name_1, 'time'); % Horas desde 01-01-1900 00:00:00
u10 = ncread(file_name_1, 'u10');
v10 = ncread(file_name_1, 'v10');

% Dados horários de 1994-2022
tempo = datevec(hours(dados_tempo) + datetime(1900, 1, 1)); % ano|mês|dia|hora|min|seg
ano = tempo(:, 1); mes = tempo(:, 2); dia = tempo(:, 3); hora = tempo(:, 4);

% Encontrar o índice da grade mais próxima ao ponto desejado
[~, idx_lon] = min(abs(lon - lon_desejado));
[~, idx_lat] = min(abs(lat - lat_desejada));

% Determinar se houve nortada para o ponto desejado em pelo menos uma amostra horária para cada dia
direcao_vento = atan2(v10(idx_lon, idx_lat, :), u10(idx_lon, idx_lat, :)) * 180 / pi + 180;
indices_nortada = (direcao_vento >= 335 | direcao_vento <= 25);  % Valores booleanos (0 ou 1), indicando se há ou não nortada

% Contar o número total de dias com nortada para o ponto desejado para cada ano
num_dias_nortada_por_ano = zeros(1, max(ano) - min(ano) + 1);
for i = min(ano):max(ano)
    % Encontrar os índices dos dias do ano atual
    indices = (ano == i);
    
    % Calcular o número total de dias com nortada para o ano atual
    num_dias_nortada_por_ano(i - min(ano) + 1) = sum(indices_nortada(indices)) / 24;  % Converter horas em dias
end

% Plotar o gráfico de barras
anos = min(ano):max(ano); 
bar(anos, num_dias_nortada_por_ano);
xlabel('Ano');
ylabel('Número de dias com Nortada');
title(sprintf('Número de dias com Nortada por Ano no ponto (%.4f, %.4f)', lon_desejado, lat_desejada));

% Calcular a média do número de dias com nortada por ano
media_nortada_por_ano = mean(num_dias_nortada_por_ano);
fprintf('A média do número de dias com nortada por ano é: %.2f\n', media_nortada_por_ano);
