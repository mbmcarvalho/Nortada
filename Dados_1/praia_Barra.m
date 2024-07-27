clear all; close all; clc;

file_name_1 = 'ERA5_UV_gust_1994_2022.nc';  
%ncdisp(file_name_1);
addpath .\m_map\

% Leitura das Variáveis
lon = ncread(file_name_1, 'longitude');  
lat = ncread(file_name_1, 'latitude');
dados_tempo = ncread(file_name_1, 'time');   % Horas desde 01-01-1900 00:00:00
u10 = ncread(file_name_1, 'u10');
v10 = ncread(file_name_1, 'v10');
rajada_max = ncread(file_name_1, 'i10fg'); 

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

direcao_vento = atan2(v10_barra, u10_barra) * 180 / pi + 180;
int_vento = sqrt(u10_barra.^2 + v10_barra.^2);

% Determinar se houve algum evento de nortada/dia, para não repetir dados num só dia
indices_nortada = (direcao_vento >= 335 | direcao_vento <= 25); 
indices_nortada_por_dia = squeeze(any(any(indices_nortada, 1), 2)); % Vetor de booleanos que indica se houve nortada nalgum momento do dia

num_dias_nortada_por_ano = zeros(1, max(ano) - min(ano) + 1);  %Vetor inicial para contar o número de dias com nortada para cada ano
for i = min(ano):max(ano)
    indices_ano_atual = find(ano == i); % Encontrar índices dos dias do ano atual, vai correndo de ano a ano até ao fim
    
    % Verificar se houve pelo menos um dia com nortada no ano atual
    if any(indices_nortada_por_dia(indices_ano_atual))
        % Se sim, contar o número total de dias únicos com nortada para o ano atual
        num_dias_nortada_por_ano(i - min(ano) + 1) = sum(diff([0; indices_nortada_por_dia(indices_ano_atual); 0]) > 0);
    end
end

anos = min(ano):max(ano);
bar(anos, num_dias_nortada_por_ano);
xlabel('Ano');
ylabel('Número de dias com Nortada');
title('Número de dias com Nortada por Ano na Praia da Barra');


total_dias_com_nortada = sum(num_dias_nortada_por_ano); % Número total de dias com nortada em todos os anos
total_anos = length(num_dias_nortada_por_ano ); % Número total de anos
media_nortada_por_ano = total_dias_com_nortada / total_anos; 
fprintf('A média do número de dias com nortada por ano é: %.2f\n', media_nortada_por_ano);





% Identificar os índices dos meses correspondentes a cada estação
% DJF (dezembro, janeiro, fevereiro)
% MAM (março, abril, maio)
% JJA (junho, julho, agosto)
% SON (setembro, outubro, novembro)

meses = tempo(:, 2); % Extrair o mês dos dados de tempo
% Calcular a média da velocidade do vento para cada estação
% Estações do ano
DJF_indices = find(meses == 12 | meses == 1 | meses == 2);
MAM_indices = find(meses == 3 | meses == 4 | meses == 5);
JJA_indices = find(meses == 6 | meses == 7 | meses == 8);
SON_indices = find(meses == 9 | meses == 10 | meses == 11);

% Calculando as médias sazonais
int_vento_DJF = mean(int_vento(:,:,DJF_indices), 3);
int_vento_MAM = mean(int_vento(:,:,MAM_indices), 3);
int_vento_JJA = mean(int_vento(:,:,JJA_indices), 3);
int_vento_SON = mean(int_vento(:,:,SON_indices), 3);

% Calcular média da velocidade do vento para cada estação
u10_DJF = mean(u10_barra(:, :, DJF_indices), 3);
v10_DJF = mean(v10_barra(:, :, DJF_indices), 3);

u10_MAM = mean(u10_barra(:, :, MAM_indices), 3);
v10_MAM = mean(v10_barra(:, :, MAM_indices), 3);

u10_JJA = mean(u10_barra(:, :, JJA_indices), 3);
v10_JJA = mean(v10_barra(:, :, JJA_indices), 3);

u10_SON = mean(u10_barra(:, :, SON_indices), 3);
v10_SON = mean(u10_barra(:, :, SON_indices), 3);

% Calcular direção média do vento para cada estação
direcao_vento_DJF = atan2(mean(v10_DJF, 'all'), mean(u10_DJF, 'all')) * 180 / pi + 180;
direcao_vento_MAM = atan2(mean(v10_MAM, 'all'), mean(u10_MAM, 'all')) * 180 / pi + 180;
direcao_vento_JJA = atan2(mean(v10_JJA, 'all'), mean(u10_JJA, 'all')) * 180 / pi + 180;
direcao_vento_SON = atan2(mean(v10_SON, 'all'), mean(u10_SON, 'all')) * 180 / pi + 180;

% Exibir resultados
disp(['Velocidade média do vento para DJF: ', num2str(mean(sqrt(u10_DJF.^2 + v10_DJF.^2), 'all')), ' m/s']);
disp(['Direção média do vento para DJF: ', num2str(direcao_vento_DJF), '°']);

disp(['Velocidade média do vento para MAM: ', num2str(mean(sqrt(u10_MAM.^2 + v10_MAM.^2), 'all')), ' m/s']);
disp(['Direção média do vento para MAM: ', num2str(direcao_vento_MAM), '°']);

disp(['Velocidade média do vento para JJA: ', num2str(mean(sqrt(u10_JJA.^2 + v10_JJA.^2), 'all')), ' m/s']);
disp(['Direção média do vento para JJA: ', num2str(direcao_vento_JJA), '°']);

disp(['Velocidade média do vento para SON: ', num2str(mean(sqrt(u10_SON.^2 + v10_SON.^2), 'all')), ' m/s']);
disp(['Direção média do vento para SON: ', num2str(direcao_vento_SON), '°']);

% Criar uma malha de longitude e latitude
[lon_malha, lat_malha] = meshgrid(lon(lon_indices), lat(lat_indices));

% Dividindo os dados em diferentes estações do ano
% Supondo que você já tenha os dados divididos em u10_DJF, v10_DJF, u10_MAM, v10_MAM, u10_JJA, v10_JJA, u10_SON, v10_SON

% DJF
u10_DJF_mean = mean(u10_DJF, 3);
v10_DJF_mean = mean(v10_DJF, 3);
% Calculando a direção média do vento para DJF
direcao_vento_DJF = atan2(mean(v10_DJF, 3), mean(u10_DJF, 3)) * 180 / pi + 180;

% MAM
u10_MAM_mean = mean(u10_MAM, 3);
v10_MAM_mean = mean(v10_MAM, 3);
% Calculando a direção média do vento para MAM
direcao_vento_MAM = atan2(mean(v10_MAM, 3), mean(u10_MAM, 3)) * 180 / pi + 180;

% JJA
u10_JJA_mean = mean(u10_JJA, 3);
v10_JJA_mean = mean(v10_JJA, 3);
% Calculando a direção média do vento para JJA
direcao_vento_JJA = atan2(mean(v10_JJA, 3), mean(u10_JJA, 3)) * 180 / pi + 180;

% SON
u10_SON_mean = mean(u10_SON, 3);
v10_SON_mean = mean(v10_SON, 3);
% Calculando a direção média do vento para SON
direcao_vento_SON = atan2(mean(v10_SON, 3), mean(u10_SON, 3)) * 180 / pi + 180;



% Criar uma malha de longitude e latitude para os dados interpolados
[lon_malha_interp, lat_malha_interp] = meshgrid(lon(lon_indices), lat(lat_indices));

% Interpolar os dados de int_vento_SON para a nova grade
int_vento_SON_interp = interp2(lon_malha, lat_malha, int_vento_SON', lon_malha_interp, lat_malha_interp);


% Projeção do mapa
m_proj('mercator', 'long', [lon_min lon_max], 'lat', [lat_min lat_max]);

% Plotar os campos médios da velocidade do vento para DJF
subplot(2, 2, 1);
m_pcolor(lon, lat, int_vento_DJF');
shading interp;
colorbar;
title('DJF');
m_coast('line');
m_grid('box', 'fancy');

% Plotar os campos médios da velocidade do vento para MAM
subplot(2, 2, 2);
m_pcolor(lon, lat, int_vento_MAM');
shading interp;
colorbar;
title('MAM');
m_coast('line');
m_grid('box', 'fancy');

% Plotar os campos médios da velocidade do vento para JJA
subplot(2, 2, 3);
m_pcolor(lon, lat, int_vento_JJA');
shading interp;
colorbar;
title('JJA');
m_coast('line');
m_grid('box', 'fancy');

% Plotar os campos médios da velocidade do vento para SON
subplot(2, 2, 4);
m_pcolor(lon, lat, int_vento_SON_interp');
shading interp;
colorbar;
title('SON');
m_coast('line');
m_grid('box', 'fancy');



% % Suponha que você já tenha os dados de u10_barra, v10_barra e direcao_vento carregados.
% 
% % Separando os dados em trimestres
% % Definindo índices para cada estação do ano
% % DJF (Dezembro, Janeiro, Fevereiro)
% indices_DJF = (tempo(:,2)==12 | tempo(:,2)==1 | tempo(:,2)==2);
% % MAM (Março, Abril, Maio)
% indices_MAM = (tempo(:,2)==3 | tempo(:,2)==4 | tempo(:,2)==5);
% % JJA (Junho, Julho, Agosto)
% indices_JJA = (tempo(:,2)==6 | tempo(:,2)==7 | tempo(:,2)==8);
% % SON (Setembro, Outubro, Novembro)
% indices_SON = (tempo(:,2)==9 | tempo(:,2)==10 | tempo(:,2)==11);
% 
% % Calculando média para cada trimestre
% media_u_DJF = mean(u10_barra(:,:,indices_DJF), 3);
% media_v_DJF = mean(v10_barra(:,:,indices_DJF), 3);
% media_direcao_DJF = mean(direcao_vento(:,:,indices_DJF), 3);
% 
% media_u_MAM = mean(u10_barra(:,:,indices_MAM), 3);
% media_v_MAM = mean(v10_barra(:,:,indices_MAM), 3);
% media_direcao_MAM = mean(direcao_vento(:,:,indices_MAM), 3);
% 
% media_u_JJA = mean(u10_barra(:,:,indices_JJA), 3);
% media_v_JJA = mean(v10_barra(:,:,indices_JJA), 3);
% media_direcao_JJA = mean(direcao_vento(:,:,indices_JJA), 3);
% 
% media_u_SON = mean(u10_barra(:,:,indices_SON), 3);
% media_v_SON = mean(v10_barra(:,:,indices_SON), 3);
% media_direcao_SON = mean(direcao_vento(:,:,indices_SON), 3);
% 
% % Plotando os campos médios da velocidade do vento e setas da direção média
% figure;
% 
% subplot(2, 2, 1);
% quiver(lon(lon_indices), lat(lat_indices), media_u_DJF, media_v_DJF);
% xlabel('Longitude');
% ylabel('Latitude');
% title('DJF - Velocidade e Direção do Vento Média');
% 
% subplot(2, 2, 2);
% quiver(lon(lon_indices), lat(lat_indices), media_u_MAM, media_v_MAM);
% xlabel('Longitude');
% ylabel('Latitude');
% title('MAM - Velocidade e Direção do Vento Média');
% 
% subplot(2, 2, 3);
% quiver(lon(lon_indices), lat(lat_indices), media_u_JJA, media_v_JJA);
% xlabel('Longitude');
% ylabel('Latitude');
% title('JJA - Velocidade e Direção do Vento Média');
% 
% subplot(2, 2, 4);
% quiver(lon(lon_indices), lat(lat_indices), media_u_SON, media_v_SON);
% xlabel('Longitude');
% ylabel('Latitude');
% title('SON - Velocidade e Direção do Vento Média');


