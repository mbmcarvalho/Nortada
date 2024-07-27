clear all; close all; clc;

file_name_1='ERA5_UV_gust_1994_2022.nc'  % abrange toda a costa atlântica portuguesa
%ncdisp(file_name_1)

%Leitura das Variáveis
lon=ncread(file_name_1,'longitude');  
lat=ncread(file_name_1,'latitude');
dados_tempo=ncread(file_name_1,'time');   %horas desde 01-01-1900 00:00:00
vento_este=ncread(file_name_1,'u10');
vento_norte=ncread(file_name_1,'v10');
rajada_max=ncread(file_name_1,'i10fg'); %velocidade


%Limites coordenadas
% geolimits([38 42],[-9.5 -7.5])
% geobasemap streets


%Dados horários de 1994-2022
tempo=datevec(hours(dados_tempo)+datetime(1900,1,1));  % ano|mês|dia|hora|min|seg
ano=tempo(:,1); mes=tempo(:,2); dia=tempo(:,3); hora=tempo(:,4);



direcao_vento = zeros(9, 17, 254208);  % matriz inicial de zeros
for i = 1:9  %lon
    for j = 1:17  %lat
        % direção do vento = atan2(v,u)
        direcao_vento(i, j, :) = atan2(squeeze(vento_norte(i, j, :)), squeeze(vento_este(i, j, :))) * (180/pi);  % em graus      
        direcao_vento(i, j, direcao_vento(i, j, :) < 0) = direcao_vento(i, j, direcao_vento(i, j, :) < 0) + 360;  %todas as direções entre 0 e 360 graus
    end
end


% Vento de direção entre 25 e 335 graus
nortada = (direcao_vento >= 335 | direcao_vento <= 25);  %valores boleanos (0 ou 1), se há ou não vento de norte (nortada)

% índices para ter as coordenadas onde ocorre a nortada
[lon_nortada, lat_nortada, tempo_nortada] = meshgrid(lon, lat, 1:size(direcao_vento, 3));

%seleciona os valores lon, lat e tempo onde ocorre a nortada
lon_nortada = lon_nortada(nortada); lat_nortada = lat_nortada(nortada); tempo_nortada = tempo_nortada(nortada);


num_dias_nortada_por_ano = zeros(1, max(ano) - min(ano) + 1); %vetor inicial de zeros
for i = min(ano):max(ano)
    indices_ano_atual = (ano == i); 
    num_horas_nortada_ano_atual = sum(nortada(indices_ano_atual));  %nº horas c/ nortada para o ano atual
    num_dias_nortada_por_ano(i - min(ano) + 1) = num_horas_nortada_ano_atual / 24;  % horas para dias
end



anos = min(ano):max(ano); 

% Gráfico de barras do nºdias com nortada por ano 
bar(anos, num_dias_nortada_por_ano);
xlabel('Ano');
ylabel('Número de dias com Nortada');
title('Número de dias com Nortada por Ano');

%disp(num_dias_nortada_por_ano(1)) 


