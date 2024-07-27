%% MAPAS - wrfout ERA5 Aveiro

clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\armazenamento_dados_1995_2014');
addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\m_map');

load('nortada_vel_wrfout_ERA5.mat');
load('vento_Barra_wrfout_ERA5.mat');
load('vento_wrfout_ERA5.mat');
load('estacao_ERA5.mat'); 
load('anual_ERA5.mat'); 
load("m_map\coastlines.mat")


lon=vento_wrfout_ERA5.lon;
lat=vento_wrfout_ERA5.lat;

vel_media_diaria = (vento_wrfout_ERA5.vel_12h + vento_wrfout_ERA5.vel_18h) / 2;
vel_media_tempo = mean(vel_media_diaria, 3);
% u = -vel_media_Barra .* sind(vento_wrfout_ERA5.("media_direcoes_diarias")); 
% v = -vel_media_Barra .* cosd(vento_wrfout_ERA5.("media_direcoes_diarias")); 


mean_u = mean(-vel_media_tempo .* sind(vento_wrfout_ERA5.media_direcoes_diarias), 3);
mean_v = mean(-vel_media_tempo .* cosd(vento_wrfout_ERA5.media_direcoes_diarias), 3); 



% % Mapa - linhas de contorno
% m_proj('lambert','lon',[min(lon(:)) max(lon(:))],'lat',[min(lat(:)) max(lat(:))]);
% m_contourf(lon, lat, vel_media_tempo);
% hold on;
% xlabel('Longitude');
% ylabel('Latitude');
% title('Linhas de contorno da velocidade do vento média');
% colorbar;
% grid on;



%------------------------------------------------------------------------
% Definindo os índices dos meses para cada estação
inverno_indices = [12,1,2] % Dezembro, Janeiro, Fevereiro
primavera_indices = [3, 4, 5];  % Março, Abril, Maio
verao_indices = [6, 7, 8];  % Junho, Julho, Agosto
outono_indices = [9, 10, 11];  % Setembro, Outubro, Novembro

% Calculando as médias de velocidade para cada estação
media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);

% Plotagem
figure;

% Plotagem para cada estação
subplot(2, 2, 1);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_inverno, 'linestyle', 'none');
title('Inverno');
colorbar;

subplot(2, 2, 2);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_primavera, 'linestyle', 'none');
title('Primavera');
colorbar;

subplot(2, 2, 3);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_verao, 'linestyle', 'none');
title('Verão');
colorbar;

subplot(2, 2, 4);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_outono, 'linestyle', 'none');
title('Outono');
colorbar;

% Adicionando limites e decorações do mapa para cada subplot
for i = 1:4
    subplot(2, 2, i);
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
end








%---------------------------------------------------------------------
%VELOCIDADE MÉDIA PARA CADA ESTAÇÃO PARA CADA ANO
ano_inicio = 1995;
ano_fim = 2014;

for ano = ano_inicio:ano_fim
    dados_ano = tabela_dados_ERA5(tabela_dados_ERA5.Ano == ano,:);
    ano_atual = dados_ano.Ano(1);
    
    mes = dados_ano.("Mês");
    dia = dados_ano.Dia;

    inverno_indices = find((mes == 12 | mes == 1 | mes == 2));
    primavera_indices = find((mes == 3 | mes == 4 | mes == 5));
    verao_indices = find((mes == 6 | mes == 7 | mes == 8));
    outono_indices = find((mes == 9 | mes == 10 | mes == 11));

    media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
    media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
    media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
    media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);


    figure;

    subplot(2, 2, 1);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_inverno, 'linestyle', 'none');
    title(sprintf('Inverno - %d', ano_atual));
    colorbar;
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');

    subplot(2, 2, 2);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_primavera, 'linestyle', 'none');
    title(sprintf('Primavera - %d', ano_atual));
    colorbar;
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');

    subplot(2, 2, 3);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_verao, 'linestyle', 'none');
    title(sprintf('Verão - %d', ano_atual));
    colorbar;
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');

    subplot(2, 2, 4);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_outono, 'linestyle', 'none');
    title(sprintf('Outono - %d', ano_atual));
    colorbar;
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
end





%-----------------------------------------------------------------------
%VELOCIDADE MÉDIA PARA CADA ESTAÇÃO (MÉDIA DO PERÍODO 1995-2014)
ano_inicio = 1995;
ano_fim = 2014;

media_inverno_total = zeros(size(lon));
media_primavera_total = zeros(size(lon));
media_verao_total = zeros(size(lon));
media_outono_total = zeros(size(lon));

for ano = ano_inicio:ano_fim
    dados_ano = tabela_dados_ERA5(tabela_dados_ERA5.Ano == ano,:);
    
    mes = dados_ano.("Mês");
    dia = dados_ano.Dia;

    inverno_indices = find((mes == 12 | mes == 1 | mes == 2));
    primavera_indices = find((mes == 3 | mes == 4 | mes == 5));
    verao_indices = find((mes == 6 | mes == 7 | mes == 8));
    outono_indices = find((mes == 9 | mes == 10 | mes == 11));
    
    media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
    media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
    media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
    media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);

    media_inverno_total = media_inverno_total + media_inverno;
    media_primavera_total = media_primavera_total + media_primavera;
    media_verao_total = media_verao_total + media_verao;
    media_outono_total = media_outono_total + media_outono;
end

media_inverno_total = media_inverno_total / (ano_fim - ano_inicio + 1);
media_primavera_total = media_primavera_total / (ano_fim - ano_inicio + 1);
media_verao_total = media_verao_total / (ano_fim - ano_inicio + 1);
media_outono_total = media_outono_total / (ano_fim - ano_inicio + 1);

figure;

subplot(2, 2, 1);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_inverno_total, 'linestyle', 'none');
title('Média Inverno (1995-2014)');
colorbar;
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');

subplot(2, 2, 2);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_primavera_total, 'linestyle', 'none');
title('Média Primavera (1995-2014)');
colorbar;
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');

subplot(2, 2, 3);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_verao_total, 'linestyle', 'none');
title('Média Verão (1995-2014)');
colorbar;
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');

subplot(2, 2, 4);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_outono_total, 'linestyle', 'none');
title('Média Outono (1995-2014)');
colorbar;
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');






% Extrair os dados de ano, mês e dia da tabela
ano = tabela_dados_ERA5.Ano;
mes = tabela_dados_ERA5.("Mês");
dia = tabela_dados_ERA5.Dia;

% Convertendo a direção para radianos
direcoes_radianos = deg2rad(tabela_dados_ERA5.("Direções médias diárias"));

vel_media_diaria=tabela_dados_ERA5.("Velocidade média diária");
% Calculando as componentes u e v
u = vel_media_diaria .* cos(direcoes_radianos);
v = vel_media_diaria .* sin(direcoes_radianos);

% Plotagem dos mapas de média das estações
figure;

subplot(2, 2, 1);
plotar_mapa_media_estacao(media_inverno_total, 'Média Inverno (1995-2014)');
hold on;
m_quiver(lon, lat, u(:,:,inverno_indices), v(:,:,inverno_indices)); % Substitua inverno_indices pelos índices da estação de inverno
hold off;

subplot(2, 2, 2);
plotar_mapa_media_estacao(media_primavera_total, 'Média Primavera (1995-2014)');
hold on;
m_quiver(lon, lat, u(:,:,primavera_indices), v(:,:,primavera_indices)); % Substitua primavera_indices pelos índices da estação de primavera
hold off;

subplot(2, 2, 3);
plotar_mapa_media_estacao(media_verao_total, 'Média Verão (1995-2014)');
hold on;
m_quiver(lon, lat, u(:,:,verao_indices), v(:,:,verao_indices)); % Substitua verao_indices pelos índices da estação de verão
hold off;

subplot(2, 2, 4);
plotar_mapa_media_estacao(media_outono_total, 'Média Outono (1995-2014)');
hold on;
m_quiver(lon, lat, u(:,:,outono_indices), v(:,:,outono_indices)); % Substitua outono_indices pelos índices da estação de outono
hold off;










% Plot dos dados
m_proj('miller','lon',[min(lon(:)) max(lon(:))],'lat',[min(lat(:)) max(lat(:))]);
m_coast('patch',[.8 .8 .8]);
hold on 
m_quiver(lon, lat, mean_u, mean_v, 2, 'color', 'k'); 

% Linhas de costa
m_plot(coastlon, coastlat, 'k', 'linewidth', 1); 

hold off;
m_grid('box','fancy','tickdir','out');


% Barra de cor
caxis([min(mean_u(:)) max(mean_u(:))]); % Ajusta os limites da barra de cor de acordo com a variável de interesse
% ax = m_contfbar([.3 .7], .05, 'h'); % Posição da barra de cor
% set(ax,'fontsize',12)
% xlabel(ax,'Componente U média do vento');

title('vento', 'fontsize', 16);
colormap(flipud(m_colmap('Blues')));

c = colorbar;
c.Label.String = 'vento';












% Configurar a projeção do mapa
m_proj('miller','lon',[min(lon(:)) max(lon(:))],'lat',[min(lat(:)) max(lat(:))]);

% Traçar as linhas da costa
m_coast('patch', [.8 .8 .8]);

hold on;
[CS, CH] = m_contourf(lon, lat, vel_media_tempo, 'edgecolor', 'none');
m_quiver(lon, lat, mean_u, mean_v);
hold off;

m_grid('box', 'fancy', 'tickdir', 'out');

ax = m_contfbar([.3 .7], .05, CS, CH);
set(ax, 'fontsize', 12);
xlabel(ax, 'Mean Daily Precipitation Rate/(kg/m^2/s)');

title([' Wind ' ], 'fontsize', 16);

% Inverter a colormap para que azul represente precipitação
colormap(flipud(m_colmap('Blues')));













% Calcular a magnitude da velocidade
magnitude = sqrt(mean_u.^2 + mean_v.^2);

% Calcular a direção da velocidade
direcao = atan2(mean_v, mean_u);

% Converter de radianos para graus
direcao_graus = rad2deg(direcao);


% Plot dos dados
m_proj('miller','lon',[min(lon(:)) max(lon(:))],'lat',[min(lat(:)) max(lat(:))]);
m_coast('patch',[.8 .8 .8]);
hold on

% Plotagem da magnitude da velocidade com pcolor
p = pcolor(lon, lat, magnitude);
set(p, 'EdgeColor', 'none');

% Plotagem das setas de direção
quiver_scale = 0.1; % Ajuste conforme necessário
quiver_lon = lon(1:5:end, 1:5:end);
quiver_lat = lat(1:5:end, 1:5:end);
quiver_u = quiver_scale * mean_u(1:5:end, 1:5:end);
quiver_v = quiver_scale * mean_v(1:5:end, 1:5:end);
m_quiver(quiver_lon, quiver_lat, quiver_u, quiver_v, 'color', 'k');

% Linhas de costa
m_plot(coastlon, coastlat, 'k', 'linewidth', 1); 

hold off;
m_grid('box','fancy','tickdir','out');

% Barra de cor
caxis([min(magnitude(:)) max(magnitude(:))]); % Ajusta os limites da barra de cor de acordo com a magnitude da velocidade

title('Velocidade e Direção do Vento', 'fontsize', 16);
colormap(flipud(m_colmap('Blues')));

c = colorbar;
c.Label.String = 'Velocidade do Vento (unidades)';




% % Plot dos dados
% m_proj('miller','lon',[min(lon(:)) max(lon(:))],'lat',[min(lat(:)) max(lat(:))]);
% m_coast('patch',[.8 .8 .8],'edgecolor','k');
% hold on 
% m_quiver(lon, lat, mean_u, mean_v, 2, 'color', 'r'); 
% 
% hold off;
% m_grid('box','fancy','tickdir','out');
% 
% % Barra de cor
% caxis([min(mean_u(:)) max(mean_u(:))]); 
% 
% title('Vento', 'fontsize', 16);
% colormap(flipud(m_colmap('Blues')));
% 
% c = colorbar;
% c.Label.String = 'Vento';









%----------------------------------------------------------------------
%BARRA
% 
% % Velocidade média ERA5  (size: lonxlatxtempo)
% vel_media_Barra = (vento_Barra_wrfout_ERA5.vel_12h_Barra + vento_Barra_wrfout_ERA5.vel_18h_Barra) / 2;
% 
% %Velocidade média ERA5 ao longo da terceira dimensão (size: lonxlat)
% vel_media_tempo_Barra = mean(vel_media_Barra, 3);
% 
% % % Componentes do vento 
% % u_Barra = -vel_media_Barra .* sind(vento_Barra_wrfout_ERA5.("media_direcoes_diarias_Barra")); 
% % v_Barra = -vel_media_Barra .* cosd(vento_Barra_wrfout_ERA5.("media_direcoes_diarias_Barra")); 
% 
% % Média ao longo da terceira dimensão (tempo)
% mean_u_Barra = mean(-vel_media_Barra .* sind(vento_Barra_wrfout_ERA5.media_direcoes_diarias_Barra), 3);
% mean_v_Barra = mean(-vel_media_Barra .* cosd(vento_Barra_wrfout_ERA5.media_direcoes_diarias_Barra), 3); 

%------------------------------------------------------------------------







%-------------------------------------------------------------
% Define o tamanho da figura e os subplots
figure;

subplot(1, 2, 1);
quiver(lon, lat, u_media_diaria(:,:,1), v_media_diaria(:,:,1));
title('Velocidade e Direção Média do Vento - Dia 1');

subplot(1, 2, 2);
quiver(lon, lat, u_media_diaria(:,:,2), v_media_diaria(:,:,2));
title('Velocidade e Direção Média do Vento - Dia 2');

% Adicione rótulos e legendas, se necessário
xlabel('Longitude');
ylabel('Latitude');



m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_quiver(lon, lat, u_media_diaria(:,:,inverno_indices(1)), v_media_diaria(:,:,inverno_indices(1)),10);
title(sprintf('Inverno - %d', ano_atual));
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');


% Definindo o espaçamento entre as setas (por exemplo, a cada 6ª posição)
espacamento = 6;

% Selecionar um subconjunto dos dados de vento
lon_subset = lon(1:espacamento:end, 1:espacamento:end);
lat_subset = lat(1:espacamento:end, 1:espacamento:end);
u_subset = u_media_diaria(1:espacamento:end, 1:espacamento:end, inverno_indices(1));
v_subset = v_media_diaria(1:espacamento:end, 1:espacamento:end, inverno_indices(1));

% Plotar o mapa com o subconjunto de dados de vento
figure;
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_quiver(lon_subset, lat_subset, u_subset, v_subset);
title(sprintf('Inverno - %d', ano_atual));
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');



%-------------------------------------------------------------------

% MAPA ESPACIAL (TODOS OS ANOS DO PERÍODO 1995-2014)

espacamento=5;

% Define o intervalo de anos
ano_inicio = 1995;
ano_fim = 2014;

% Itera sobre os anos
for ano = ano_inicio:ano_fim
    % Seleciona os dados do ano atual
    dados_ano = tabela_dados_ERA5(tabela_dados_ERA5.Ano == ano,:);
    ano_atual = dados_ano.Ano(1);
    
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
   
    media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
    media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
    media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
    media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);


    % Cria uma nova figura
    figure;
    
    % Plot da estação do Inverno
    subplot(2, 2, 1);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_inverno, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco ,  lat_espaco , u_media_diaria_DJF, v_media_diaria_DJF,'color', 'k');
    title(sprintf('Inverno - %d', ano_atual));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação da Primavera
    subplot(2, 2, 2);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_primavera, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_MAM, v_media_diaria_MAM,'color', 'k');
    title(sprintf('Primavera - %d', ano_atual));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação do Verão
    subplot(2, 2, 3);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_verao, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_JJA, v_media_diaria_JJA,'color', 'k');
    title(sprintf('Verão - %d', ano_atual));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação do Outono
    subplot(2, 2, 4);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_outono, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_SON, v_media_diaria_SON,'color', 'k');
    title(sprintf('Outono - %d', ano_atual));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    sgtitle('Velocidade e direção do vento por estação - wrfout ERA5');

end




%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MÉDIA DOS ANOS TODOS 

ano_inicio = 1995;
ano_fim = 2014;

% Indices para cada estação
inverno_indices = find((mes == 12 | mes == 1 | mes == 2));
primavera_indices = find((mes == 3 | mes == 4 | mes == 5));
verao_indices = find((mes == 6 | mes == 7 | mes == 8));
outono_indices = find((mes == 9 | mes == 10 | mes == 11));

% Calcula a média de todas as estações ao longo de todos os anos
media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);

% Cria uma nova figura
figure;

% Plot da estação do Inverno
subplot(2, 2, 1);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_inverno, 'linestyle', 'none');
hold on
m_quiver(lon_subset, lat_subset, u_subset, v_subset,'color', 'k');
title('Inverno');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

% Plot da estação da Primavera
subplot(2, 2, 2);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_primavera, 'linestyle', 'none');
hold on
m_quiver(lon_subset, lat_subset, u_subset, v_subset,'color', 'k');
title('Primavera');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

% Plot da estação do Verão
subplot(2, 2, 3);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_verao, 'linestyle', 'none');
hold on
m_quiver(lon_subset, lat_subset, u_subset, v_subset,'color', 'k');
title('Verão');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

% Plot da estação do Outono
subplot(2, 2, 4);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_outono, 'linestyle', 'none');
hold on
m_quiver(lon_subset, lat_subset, u_subset, v_subset,'color', 'k');
title('Outono');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)')

sgtitle('Velocidade e direção do vento por estação (1995-2014) - wrfout ERA5');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Inicialize matrizes para armazenar as médias sazonais
avg_inverno = zeros(size(media_inverno));
avg_primavera = zeros(size(media_primavera));
avg_verao = zeros(size(media_verao));
avg_outono = zeros(size(media_outono));

% Defina o intervalo de anos
ano_inicio = 1995;
ano_fim = 2014;

% Iterando sobre os anos
for ano = ano_inicio:ano_fim
    % Selecione os dados para o ano atual
    dados_ano = tabela_dados_ERA5(tabela_dados_ERA5.Ano == ano,:);
    ano_atual = dados_ano.Ano(1);
    
    % Atualize as médias sazonais adicionando os dados para o ano atual
    avg_inverno = avg_inverno + media_inverno;
    avg_primavera = avg_primavera + media_primavera;
    avg_verao = avg_verao + media_verao;
    avg_outono = avg_outono + media_outono;
end

% Calcule a média dividindo pelo número de anos
avg_inverno = avg_inverno / (ano_fim - ano_inicio + 1);
avg_primavera = avg_primavera / (ano_fim - ano_inicio + 1);
avg_verao = avg_verao / (ano_fim - ano_inicio + 1);
avg_outono = avg_outono / (ano_fim - ano_inicio + 1);

% Crie uma nova figura para a média de todos os anos para cada estação
figure;

% Plote a média para o Inverno
subplot(2, 2, 1);
m_contourf(lon, lat, avg_inverno, 'linestyle', 'none');
hold on
m_quiver(lon_espaco ,  lat_espaco , mean(u_media_diaria_DJF, 3), mean(v_media_diaria_DJF, 3),'color', 'k');
title('Média do Inverno');
xlabel('Longitude');
ylabel('Latitude');
colorbar;

% Plote a média para a Primavera
subplot(2, 2, 2);
m_contourf(lon, lat, avg_primavera, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, mean(u_media_diaria_MAM, 3), mean(v_media_diaria_MAM, 3),'color', 'k');
title('Média da Primavera');
xlabel('Longitude');
ylabel('Latitude');
colorbar;

% Plote a média para o Verão
subplot(2, 2, 3);
m_contourf(lon, lat, avg_verao, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, mean(u_media_diaria_JJA, 3), mean(v_media_diaria_JJA, 3),'color', 'k');
title('Média do Verão');
xlabel('Longitude');
ylabel('Latitude');
colorbar;>

% Plote a média para o Outono
subplot(2, 2, 4);
m_contourf(lon, lat, avg_outono, 'linestyle', 'none');
hold on
m_quiver(lon_espaco, lat_espaco, mean(u_media_diaria_SON, 3), mean(v_media_diaria_SON, 3),'color', 'k');
title('Média do Outono');
xlabel('Longitude');
ylabel('Latitude');
colorbar;



%-----------------------------------------------------------------------
%VELOCIDADE MÉDIA PARA CADA ESTAÇÃO (MÉDIA DO PERÍODO 1995-2014)
ano_inicio = 1995;
ano_fim = 2014;

media_inverno_total = zeros(size(lon));
media_primavera_total = zeros(size(lon));
media_verao_total = zeros(size(lon));
media_outono_total = zeros(size(lon));

for ano = ano_inicio:ano_fim
    dados_ano = tabela_dados_ERA5(tabela_dados_ERA5.Ano == ano,:);
    
    mes = dados_ano.("Mês");
    dia = dados_ano.Dia;

    inverno_indices = find((mes == 12 | mes == 1 | mes == 2));
    primavera_indices = find((mes == 3 | mes == 4 | mes == 5));
    verao_indices = find((mes == 6 | mes == 7 | mes == 8));
    outono_indices = find((mes == 9 | mes == 10 | mes == 11));
    
    media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
    media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
    media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
    media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);

    media_inverno_total = media_inverno_total + media_inverno;
    media_primavera_total = media_primavera_total + media_primavera;
    media_verao_total = media_verao_total + media_verao;
    media_outono_total = media_outono_total + media_outono;
end

media_inverno_total = media_inverno_total / (ano_fim - ano_inicio + 1);
media_primavera_total = media_primavera_total / (ano_fim - ano_inicio + 1);
media_verao_total = media_verao_total / (ano_fim - ano_inicio + 1);
media_outono_total = media_outono_total / (ano_fim - ano_inicio + 1);

figure;

    % Plot da estação do Inverno
    subplot(2, 2, 1);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_inverno_total, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco ,  lat_espaco , u_media_diaria_DJF, v_media_diaria_DJF, 1, 'color', 'k');
    title(sprintf('Inverno '));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação da Primavera
    subplot(2, 2, 2);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_primavera_total, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_MAM, v_media_diaria_MAM, 1, 'color', 'k');
    title(sprintf('Primavera '));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação do Verão
    subplot(2, 2, 3);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_verao_total, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_JJA, v_media_diaria_JJA, 1, 'color', 'k');
    title(sprintf('Verão '));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação do Outono
    subplot(2, 2, 4);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_outono_total, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_SON, v_media_diaria_SON, 1, 'color', 'k');
    title(sprintf('Outono '));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar




%% MAPAS - wrfout MPI Aveiro

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






espacamento=5;

% Define o intervalo de anos
ano_inicio = 1995;
ano_fim = 2014;

% Itera sobre os anos
for ano = ano_inicio:ano_fim
    % Seleciona os dados do ano atual
    dados_ano = tabela_dados_MPI(tabela_dados_MPI.Ano == ano,:);
    ano_atual = dados_ano.Ano(1);
    
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
   
    media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
    media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
    media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
    media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);


    % Cria uma nova figura
    figure;
    
    % Plot da estação do Inverno
    subplot(2, 2, 1);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_inverno, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco ,  lat_espaco , u_media_diaria_DJF, v_media_diaria_DJF,'color', 'k');
    title(sprintf('Inverno - %d', ano_atual));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação da Primavera
    subplot(2, 2, 2);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_primavera, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_MAM, v_media_diaria_MAM,'color', 'k');
    title(sprintf('Primavera - %d', ano_atual));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação do Verão
    subplot(2, 2, 3);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_verao, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_JJA, v_media_diaria_JJA,'color', 'k');
    title(sprintf('Verão - %d', ano_atual));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    % Plot da estação do Outono
    subplot(2, 2, 4);
    m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
    m_contourf(lon, lat, media_outono, 'linestyle', 'none');
    hold on
    m_quiver(lon_espaco, lat_espaco, u_media_diaria_SON, v_media_diaria_SON,'color', 'k');
    title(sprintf('Outono - %d', ano_atual));
    xlabel('Longitude');
    ylabel('Latitude');
    m_coast('linewidth', 1, 'color', 'k');
    m_grid('linestyle', 'none', 'box', 'fancy');
    h = colorbar; 
    ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

    sgtitle('Velocidade e direção do vento por estação - wrfout ERA5');

end


% Definindo o espaçamento entre as setas (por exemplo, a cada 6ª posição)
espacamento = 6;

% Selecionar um subconjunto dos dados de vento
lon_subset = lon(1:espacamento:end, 1:espacamento:end);
lat_subset = lat(1:espacamento:end, 1:espacamento:end);
u_subset = u_media_diaria(1:espacamento:end, 1:espacamento:end, inverno_indices(1));
v_subset = v_media_diaria(1:espacamento:end, 1:espacamento:end, inverno_indices(1));


% MÉDIA DOS ANOS TODOS 

ano_inicio = 1995;
ano_fim = 2014;

% Indices para cada estação
inverno_indices = find((mes == 12 | mes == 1 | mes == 2));
primavera_indices = find((mes == 3 | mes == 4 | mes == 5));
verao_indices = find((mes == 6 | mes == 7 | mes == 8));
outono_indices = find((mes == 9 | mes == 10 | mes == 11));

% Calcula a média de todas as estações ao longo de todos os anos
media_inverno = calculo_media_estacao(vel_media_diaria, inverno_indices);
media_primavera = calculo_media_estacao(vel_media_diaria, primavera_indices);
media_verao = calculo_media_estacao(vel_media_diaria, verao_indices);
media_outono = calculo_media_estacao(vel_media_diaria, outono_indices);

% Cria uma nova figura
figure;

% Plot da estação do Inverno
subplot(2, 2, 1);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_inverno, 'linestyle', 'none');
hold on
m_quiver(lon_subset, lat_subset, u_subset, v_subset,'color', 'k');
title('Inverno');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

% Plot da estação da Primavera
subplot(2, 2, 2);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_primavera, 'linestyle', 'none');
hold on
m_quiver(lon_subset, lat_subset, u_subset, v_subset,'color', 'k');
title('Primavera');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

% Plot da estação do Verão
subplot(2, 2, 3);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_verao, 'linestyle', 'none');
hold on
m_quiver(lon_subset, lat_subset, u_subset, v_subset,'color', 'k');
title('Verão');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)'); % Título à colorbar

% Plot da estação do Outono
subplot(2, 2, 4);
m_proj('Mercator', 'lon', [min(lon(:)), max(lon(:))], 'lat', [min(lat(:)), max(lat(:))]);
m_contourf(lon, lat, media_outono, 'linestyle', 'none');
hold on
m_quiver(lon_subset, lat_subset, u_subset, v_subset,'color', 'k');
title('Outono');
xlabel('Longitude');
ylabel('Latitude');
m_coast('linewidth', 1, 'color', 'k');
m_grid('linestyle', 'none', 'box', 'fancy');
h = colorbar; 
ylabel(h, 'Velocidade do Vento (m/s)')

sgtitle('Velocidade e direção do vento por estação (1995-2014) - wrfout ERA5');

