%% MAPAS ESPACIAIS - wrfout ERA5 

clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\armazenamento_dados_1995_2014');
addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\m_map');
addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\Circular_Statistics');

load('nortada_vel_wrfout_ERA5.mat');
load('vento_Barra_wrfout_ERA5.mat');
load('vento_wrfout_ERA5.mat');
load('estacao_ERA5.mat'); 
load('anual_ERA5.mat'); 
load("m_map\coastlines.mat")

lon=vento_wrfout_ERA5.lon;
lat=vento_wrfout_ERA5.lat;

mes=tabela_dados_ERA5.("Mês");
u10_12h=vento_wrfout_ERA5.u10_12h;
v10_12h=vento_wrfout_ERA5.v10_12h;
u10_18h=vento_wrfout_ERA5.u10_18h;
v10_18h=vento_wrfout_ERA5.v10_18h;

direcao_media_12h = atan2d(v10_12h, u10_12h);
direcao_media_18h = atan2d(v10_18h, u10_18h);
vel_media_diaria = (vento_wrfout_ERA5.vel_12h + vento_wrfout_ERA5.vel_18h) / 2;
direcao_media_diaria = (direcao_media_12h + direcao_media_18h) / 2;

u_media_diaria = zeros(size(u10_12h));
v_media_diaria = zeros(size(v10_12h));

for i = 1:size(u10_12h, 3)  % Itera sobre os dias
    % Calcula a média da direção para cada dia utilizando a função circ_mean_degrees
    direcao_media_diaria(:,:,i) = circ_mean_degrees(direcao_media_diaria(:,:,i), [], 3);

    u_media_diaria(:,:,i) = vel_media_diaria(:,:,i) .* cosd(direcao_media_diaria(:,:,i));
    v_media_diaria(:,:,i) = vel_media_diaria(:,:,i) .* sind(direcao_media_diaria(:,:,i));
end


% Converter direção média diária para componentes u e v
u_media_diaria = vel_media_diaria .* cosd(direcao_media_diaria);
v_media_diaria = vel_media_diaria .* sind(direcao_media_diaria);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MAPAS ESPACIAIS SASONAIS wrfout-ERA5 - Média 1995-2014

espacamento = 5;

% Define o intervalo de anos
ano_inicio = 1995;
ano_fim = 2014;

% Inicializa as médias totais para cada estação
media_inverno_total = zeros(size(u_media_diaria(:,:,1)));
media_primavera_total = zeros(size(u_media_diaria(:,:,1)));
media_verao_total = zeros(size(u_media_diaria(:,:,1)));
media_outono_total = zeros(size(u_media_diaria(:,:,1)));

% Itera sobre os anos para calcular as médias totais
for ano = ano_inicio:ano_fim
    % Seleciona os dados do ano atual
    dados_ano = tabela_dados_ERA5(tabela_dados_ERA5.Ano == ano,:);
    
    % Calcula os índices de cada estação para o ano atual
    inverno_indices = find((mes == 12 | mes == 1 | mes == 2));
    primavera_indices = find((mes == 3 | mes == 4 | mes == 5));
    verao_indices = find((mes == 6 | mes == 7 | mes == 8));
    outono_indices = find((mes == 9 | mes == 10 | mes == 11));

    lon_espaco = lon(1:espacamento:end, 1:espacamento:end);
    lat_espaco = lat(1:espacamento:end, 1:espacamento:end);
    u_media_diaria_DJF=u_media_diaria(1:espacamento:end, 1:espacamento:end, inverno_indices(1));
    v_media_diaria_DJF=v_media_diaria(1:espacamento:end, 1:espacamento:end, inverno_indices(1));
    u_media_diaria_MAM=u_media_diaria(1:espacamento:end, 1:espacamento:end, primavera_indices(1));
    v_media_diaria_MAM=v_media_diaria(1:espacamento:end, 1:espacamento:end, primavera_indices(1));
    u_media_diaria_JJA=u_media_diaria(1:espacamento:end, 1:espacamento:end, verao_indices(1));
    v_media_diaria_JJA=v_media_diaria(1:espacamento:end, 1:espacamento:end, verao_indices(1));
    u_media_diaria_SON=u_media_diaria(1:espacamento:end, 1:espacamento:end, outono_indices(1));
    v_media_diaria_SON=v_media_diaria(1:espacamento:end, 1:espacamento:end, outono_indices(1));
   
    % Calcula as médias para o ano atual
    media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
    media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
    media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
    media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);

    % Adiciona as médias do ano atual às médias totais
    media_inverno_total = media_inverno_total + media_inverno;
    media_primavera_total = media_primavera_total + media_primavera;
    media_verao_total = media_verao_total + media_verao;
    media_outono_total = media_outono_total + media_outono;
end

% Calcula as médias totais dividindo pelo número total de anos
media_inverno_total = media_inverno_total / (ano_fim - ano_inicio + 1);
media_primavera_total = media_primavera_total / (ano_fim - ano_inicio + 1);
media_verao_total = media_verao_total / (ano_fim - ano_inicio + 1);
media_outono_total = media_outono_total / (ano_fim - ano_inicio + 1);

% Calcule o intervalo comum para todas as colorbars
cmin = min([media_inverno_total(:); media_primavera_total(:); media_verao_total(:); media_outono_total(:)]);
cmax = max([media_inverno_total(:); media_primavera_total(:); media_verao_total(:); media_outono_total(:)]);

% Cria uma nova figura
figure;

% Plot da estação do Inverno
subplot(2, 2, 1);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_inverno_total, 'linestyle', 'none');
hold on
m_quiver(lon_espaco ,  lat_espaco , u_media_diaria_DJF, v_media_diaria_DJF,'color', 'k');
title('Inverno');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
caxis([cmin cmax]); % Define os limites da colorbar

% Plot da estação da Primavera
subplot(2, 2, 2);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_primavera_total, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, u_media_diaria_MAM, v_media_diaria_MAM,'color', 'k');
title('Primavera');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
caxis([cmin cmax]); % Define os limites da colorbar

% Plot da estação do Verão
subplot(2, 2, 3);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_verao_total, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, u_media_diaria_JJA, v_media_diaria_JJA,'color', 'k');
title('Verão');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
caxis([cmin cmax]); % Define os limites da colorbar

% Plot da estação do Outono
subplot(2, 2, 4);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_outono_total, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, u_media_diaria_SON, v_media_diaria_SON,'color', 'k');
title('Outono');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
caxis([cmin cmax]); % Define os limites da colorbar

sgtitle('wrfout ERA5');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAPA ESPACIAL ANUAL - wrfout ERA5 - Média (1995-2014)


espacamento = 5;

% Define o intervalo de anos
ano_inicio = 1995;
ano_fim = 2014;

% Inicializa as médias totais para cada estação
media_anual_total = zeros(size(u_media_diaria(:,:,1)));

% Itera sobre os anos para calcular as médias anuais totais
for ano = ano_inicio:ano_fim
    % Seleciona os dados do ano atual
    dados_ano = tabela_dados_ERA5(tabela_dados_ERA5.Ano == ano,:);
    
    % Calcula as médias para o ano atual
    media_anual = mean(vel_media_diaria(:,:,mes >= 1 & mes <= 12), 3); % Calcula a média ao longo de todos os meses
    
    % Adiciona a média do ano atual à média anual total
    media_anual_total = media_anual_total + media_anual;
end

% Calcula a média anual total dividindo pelo número total de anos
media_anual_total = media_anual_total / (ano_fim - ano_inicio + 1);

% Calcule o intervalo comum para todas as colorbars
cmin = min([media_inverno_total(:); media_primavera_total(:); media_verao_total(:); media_outono_total(:)]);
cmax = max([media_inverno_total(:); media_primavera_total(:); media_verao_total(:); media_outono_total(:)]);

% Cria uma nova figura
figure;

% Plot da média anual
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_anual_total, 'linestyle', 'none');
hold on

% Reduz a resolução dos dados para plotagem
lon_espaco = lon(1:espacamento:end, 1:espacamento:end);
lat_espaco = lat(1:espacamento:end, 1:espacamento:end);
u_media_diaria_reduzida = mean(u_media_diaria, 3);
v_media_diaria_reduzida = mean(v_media_diaria, 3);
u_media_diaria_reduzida = u_media_diaria_reduzida(1:espacamento:end, 1:espacamento:end);
v_media_diaria_reduzida = v_media_diaria_reduzida(1:espacamento:end, 1:espacamento:end);

% Plot das setas de direção do vento
m_quiver(lon_espaco, lat_espaco, u_media_diaria_reduzida, v_media_diaria_reduzida, 'color', 'k');

title('wrfout ERA5');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% MAPAS ESPACIAIS - wrfout MPI 

clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\armazenamento_dados_1995_2014');
addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\m_map');
addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\Circular_Statistics');


load('nortada_vel_wrfout_MPI_hist.mat');
load('vento_Barra_wrfout_MPI_hist.mat');
load('vento_wrfout_MPI_hist.mat');
load('estacao_MPI.mat'); 
load('anual_MPI.mat'); 
load("m_map\coastlines.mat")

lon=vento_wrfout_MPI_hist.lon;
lat=vento_wrfout_MPI_hist.lat;

mes=tabela_dados_MPI.("Mês");
u10_12h=vento_wrfout_MPI_hist.u10_12h;
v10_12h=vento_wrfout_MPI_hist.v10_12h;
u10_18h=vento_wrfout_MPI_hist.u10_18h;
v10_18h=vento_wrfout_MPI_hist.v10_18h;


direcao_media_12h = atan2d(v10_12h, u10_12h);
direcao_media_18h = atan2d(v10_18h, u10_18h);
vel_media_diaria = (vento_wrfout_MPI_hist.vel_12h + vento_wrfout_MPI_hist.vel_18h) / 2;
direcao_media_diaria = (direcao_media_12h + direcao_media_18h) / 2;


u_media_diaria = zeros(size(u10_12h));
v_media_diaria = zeros(size(v10_12h));

for i = 1:size(u10_12h, 3)  % Itera sobre os dias
    % Calcula a média da direção para cada dia utilizando a função circ_mean_degrees
    direcao_media_diaria(:,:,i) = circ_mean_degrees(direcao_media_diaria(:,:,i), [], 3);

    u_media_diaria(:,:,i) = vel_media_diaria(:,:,i) .* cosd(direcao_media_diaria(:,:,i));
    v_media_diaria(:,:,i) = vel_media_diaria(:,:,i) .* sind(direcao_media_diaria(:,:,i));
end


% Converter direção média diária para componentes u e v
u_media_diaria = vel_media_diaria .* cosd(direcao_media_diaria);
v_media_diaria = vel_media_diaria .* sind(direcao_media_diaria);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MAPAS ESPACIAIS SASONAIS wrfout-ERA5 - Média 1995-2014

espacamento = 5;

% Define o intervalo de anos
ano_inicio = 1995;
ano_fim = 2014;

% Inicializa as médias totais para cada estação
media_inverno_total = zeros(size(u_media_diaria(:,:,1)));
media_primavera_total = zeros(size(u_media_diaria(:,:,1)));
media_verao_total = zeros(size(u_media_diaria(:,:,1)));
media_outono_total = zeros(size(u_media_diaria(:,:,1)));

% Itera sobre os anos para calcular as médias totais
for ano = ano_inicio:ano_fim
    % Seleciona os dados do ano atual
    dados_ano = tabela_dados_MPI(tabela_dados_MPI.Ano == ano,:);
    
    % Calcula os índices de cada estação para o ano atual
    inverno_indices = find((mes == 12 | mes == 1 | mes == 2));
    primavera_indices = find((mes == 3 | mes == 4 | mes == 5));
    verao_indices = find((mes == 6 | mes == 7 | mes == 8));
    outono_indices = find((mes == 9 | mes == 10 | mes == 11));

    lon_espaco = lon(1:espacamento:end, 1:espacamento:end);
    lat_espaco = lat(1:espacamento:end, 1:espacamento:end);
    u_media_diaria_DJF=u_media_diaria(1:espacamento:end, 1:espacamento:end, inverno_indices(1));
    v_media_diaria_DJF=v_media_diaria(1:espacamento:end, 1:espacamento:end, inverno_indices(1));
    u_media_diaria_MAM=u_media_diaria(1:espacamento:end, 1:espacamento:end, primavera_indices(1));
    v_media_diaria_MAM=v_media_diaria(1:espacamento:end, 1:espacamento:end, primavera_indices(1));
    u_media_diaria_JJA=u_media_diaria(1:espacamento:end, 1:espacamento:end, verao_indices(1));
    v_media_diaria_JJA=v_media_diaria(1:espacamento:end, 1:espacamento:end, verao_indices(1));
    u_media_diaria_SON=u_media_diaria(1:espacamento:end, 1:espacamento:end, outono_indices(1));
    v_media_diaria_SON=v_media_diaria(1:espacamento:end, 1:espacamento:end, outono_indices(1));
   
    % Calcula as médias para o ano atual
    media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
    media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
    media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
    media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);

    % Adiciona as médias do ano atual às médias totais
    media_inverno_total = media_inverno_total + media_inverno;
    media_primavera_total = media_primavera_total + media_primavera;
    media_verao_total = media_verao_total + media_verao;
    media_outono_total = media_outono_total + media_outono;
end

% Calcula as médias totais dividindo pelo número total de anos
media_inverno_total = media_inverno_total / (ano_fim - ano_inicio + 1);
media_primavera_total = media_primavera_total / (ano_fim - ano_inicio + 1);
media_verao_total = media_verao_total / (ano_fim - ano_inicio + 1);
media_outono_total = media_outono_total / (ano_fim - ano_inicio + 1);

% Calcule o intervalo comum para todas as colorbars
cmin = min([media_inverno_total(:); media_primavera_total(:); media_verao_total(:); media_outono_total(:)]);
cmax = max([media_inverno_total(:); media_primavera_total(:); media_verao_total(:); media_outono_total(:)]);

% Cria uma nova figura
figure;

% Plot da estação do Inverno
subplot(2, 2, 1);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_inverno_total, 'linestyle', 'none');
hold on
m_quiver(lon_espaco ,  lat_espaco , u_media_diaria_DJF, v_media_diaria_DJF,'color', 'k');
title('Inverno');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
caxis([cmin cmax]); % Define os limites da colorbar

% Plot da estação da Primavera
subplot(2, 2, 2);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_primavera_total, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, u_media_diaria_MAM, v_media_diaria_MAM,'color', 'k');
title('Primavera');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
caxis([cmin cmax]); % Define os limites da colorbar

% Plot da estação do Verão
subplot(2, 2, 3);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_verao_total, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, u_media_diaria_JJA, v_media_diaria_JJA,'color', 'k');
title('Verão');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
caxis([cmin cmax]); % Define os limites da colorbar

% Plot da estação do Outono
subplot(2, 2, 4);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_outono_total, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, u_media_diaria_SON, v_media_diaria_SON,'color', 'k');
title('Outono');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
caxis([cmin cmax]); % Define os limites da colorbar

sgtitle('wrfout MPI');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAPA ESPACIAL ANUAL - wrfout ERA5 - Média (1995-2014)


espacamento = 5;

% Define o intervalo de anos
ano_inicio = 1995;
ano_fim = 2014;

% Inicializa as médias totais para cada estação
media_anual_total = zeros(size(u_media_diaria(:,:,1)));

% Itera sobre os anos para calcular as médias anuais totais
for ano = ano_inicio:ano_fim
    % Seleciona os dados do ano atual
    dados_ano = tabela_dados_MPI(tabela_dados_MPI.Ano == ano,:);
    
    % Calcula as médias para o ano atual
    media_anual = mean(vel_media_diaria(:,:,mes >= 1 & mes <= 12), 3); % Calcula a média ao longo de todos os meses
    
    % Adiciona a média do ano atual à média anual total
    media_anual_total = media_anual_total + media_anual;
end

% Calcula a média anual total dividindo pelo número total de anos
media_anual_total = media_anual_total / (ano_fim - ano_inicio + 1);

% Calcule o intervalo comum para todas as colorbars
cmin = min([media_inverno_total(:); media_primavera_total(:); media_verao_total(:); media_outono_total(:)]);
cmax = max([media_inverno_total(:); media_primavera_total(:); media_verao_total(:); media_outono_total(:)]);

% Cria uma nova figura
figure;

% Plot da média anual
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_anual_total, 'linestyle', 'none');
hold on

% Reduz a resolução dos dados para plotagem
lon_espaco = lon(1:espacamento:end, 1:espacamento:end);
lat_espaco = lat(1:espacamento:end, 1:espacamento:end);
u_media_diaria_reduzida = mean(u_media_diaria, 3);
v_media_diaria_reduzida = mean(v_media_diaria, 3);
u_media_diaria_reduzida = u_media_diaria_reduzida(1:espacamento:end, 1:espacamento:end);
v_media_diaria_reduzida = v_media_diaria_reduzida(1:espacamento:end, 1:espacamento:end);

% Plot das setas de direção do vento
m_quiver(lon_espaco, lat_espaco, u_media_diaria_reduzida, v_media_diaria_reduzida, 'color', 'k');

title('wrfout MPI');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)', 'FontSize', 10); % Título à colorbar
