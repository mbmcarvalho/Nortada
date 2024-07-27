%% DO FICHEIRO "wrfout_ERA5_UV10m_12_18h.nc"

clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\Circular_Statistics');
addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\ficheiros');

file_name = "wrfout_ERA5_UV10m_12_18h.nc";

lon = double(ncread(file_name, 'XLONG'));  % unidades: graus a este
lat = double(ncread(file_name, 'XLAT'));   % unidades: graus a norte
u10 = double(ncread(file_name, 'U10'));
v10 = double(ncread(file_name, 'V10'));
dados_tempo = double(ncread(file_name, "XTIME"));  % unidades: horas desde 1994-12-8 00:00:00


% Converter tempo para data e hora
tempo = datevec(hours(dados_tempo) + datetime(1994, 12, 8));  % ano|mês|dia|hora|min|seg
ano = tempo(:,1); mes = tempo(:,2); dia = tempo(:,3); hora = tempo(:,4);

% Coordenadas da Praia da Barra
lon_Barra = [-8.7578430, -8.7288208]; % este-oeste
lat_Barra = [40.6077614, 40.6470909]; % norte-sul

% %Limites coordenadas
% geolimits([40.6077614 40.6470909],[-8.7578430 -8.7288208])
% geobasemap streets


% Posições da lat e da lon para a região da Barra
lon_pos = find(lon >= lon_Barra(1) & lon <= lon_Barra(2));
lat_pos = find(lat >= lat_Barra(1) & lat <= lat_Barra(2));


%ÍNDICES das posições de lon e lat
[lon_row,lon_col] = ind2sub(size(lon_pos),lon_pos);
[lat_row,lat_col] = ind2sub(size(lat_pos),lat_pos);

[lon_BARRA, lat_BARRA] = meshgrid(lon(lon_pos), lat(lat_pos));


% Posições correspondentes às 12h e 18h
posicoes_12h = find(hora == 12); 
posicoes_18h = find(hora == 18); 

% Dados das componentes de vento para a malha inicial
u10_12h = u10(:, :, posicoes_12h);
v10_12h = v10(:, :, posicoes_12h);
u10_18h = u10(:, :, posicoes_18h);
v10_18h = v10(:, :, posicoes_18h);

% Dados das componentes de vento dentro da região de interesse
u10_12h_Barra = u10(lon_row, lat_col, posicoes_12h);
v10_12h_Barra = v10(lon_row, lat_col, posicoes_12h);
u10_18h_Barra = u10(lon_row, lat_col, posicoes_18h);
v10_18h_Barra = v10(lon_row, lat_col, posicoes_18h);


% Velocidade [sqrt(u.^2 + v.^2)]
vel_12h = sqrt(u10_12h.^2 + v10_12h.^2);
vel_18h = sqrt(u10_18h.^2 + v10_18h.^2);

vel_12h_Barra = sqrt(u10_12h_Barra.^2 + v10_12h_Barra.^2);
vel_18h_Barra = sqrt(u10_18h_Barra.^2 + v10_18h_Barra.^2);
%mean_vel = mean(cat(3, vel_12h_Barra, vel_18h_Barra), 3); %cat concatena as matrizes

media_vel_12h = squeeze(mean(vel_12h_Barra, [1 2])); % Média da velocidade do vento para as 12h ao longo da dimensão temporal
media_vel_18h = squeeze(mean(vel_18h_Barra, [1 2])); % Média da velocidade do vento para as 18h ao longo da dimensão temporal
media_vel_12h_18h = (media_vel_12h + media_vel_18h) / 2;


% MÉDIAS VETORES
% Médias de várias amostras das direções do vento para as 12h e para as 18h
medias_direcao_12h_Barra = circ_mean_degrees(reshape(atan2d(v10_12h_Barra, u10_12h_Barra), [], size(u10_12h_Barra, 3)), [], 1);
medias_direcao_18h_Barra = circ_mean_degrees(reshape(atan2d(v10_18h_Barra, u10_18h_Barra), [], size(u10_18h_Barra, 3)), [], 1);
% Médias diárias das 12h com as 18h
media_direcao_diaria_Barra = circ_mean_degrees([medias_direcao_12h_Barra; medias_direcao_18h_Barra], [], 1);


% MÉDIAS MATRIZES 
media_dir_v10_12h_Barra = circ_mean_degrees(atan2d(v10_12h_Barra, u10_12h_Barra), [], 3);
media_dir_v10_18h_Barra = circ_mean_degrees(atan2d(v10_18h_Barra, u10_18h_Barra), [], 3);
media_direcoes_diarias_Barra = (media_dir_v10_12h_Barra + media_dir_v10_18h_Barra) / 2;

media_dir_v10_12h = circ_mean_degrees(atan2d(v10_12h, u10_12h), [], 3);
media_dir_v10_18h = circ_mean_degrees(atan2d(v10_18h, u10_18h), [], 3);
media_direcoes_diarias= (media_dir_v10_12h + media_dir_v10_18h) / 2;

% NORTADA
nortada_media_diaria = (media_direcao_diaria_Barra >= 335 | media_direcao_diaria_Barra <= 25);
nortada_media_12h = (medias_direcao_12h_Barra >= 335 | medias_direcao_12h_Barra <= 25);
nortada_media_18h = (medias_direcao_18h_Barra >= 335 | medias_direcao_18h_Barra <= 25);


% TABELA COM AS VARIÁVEIS IMPORTANTES PARA CÁLCULOS DA REGIÃO DA PRAIA DA BARRA
tabela_dados_ERA5 = table(ano(posicoes_12h), mes(posicoes_12h), dia(posicoes_12h), 'VariableNames', {'Ano', 'Mês', 'Dia'});

% Adiciona novas variáveis à tabela
tabela_dados_ERA5 = addvars(tabela_dados_ERA5, medias_direcao_12h_Barra', medias_direcao_18h_Barra', media_direcao_diaria_Barra', 'NewVariableNames', {'Direções médias (12h)', 'Direções médias (18h)', 'Direções médias diárias'});
tabela_dados_ERA5 = addvars(tabela_dados_ERA5, nortada_media_diaria', nortada_media_12h', nortada_media_18h', 'NewVariableNames', {'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)'});
tabela_dados_ERA5 = addvars(tabela_dados_ERA5, media_vel_12h, media_vel_18h, media_vel_12h_18h, 'NewVariableNames', {'Velocidade média(12h)', 'Velocidade média(18h)', 'Velocidade média diária'});


% Adiciona uma nova coluna com as estações do ano 
estacoes = arrayfun(@(x) month2season(x), tabela_dados_ERA5.("Mês"), 'UniformOutput', false);
tabela_dados_ERA5 = addvars(tabela_dados_ERA5, estacoes, 'After', 'Dia', 'NewVariableNames', 'Estações');



% Guarda a tabela_dados_ERA5 na pasta 'armazenamento_dados_1995_2014'
nome_da_pasta = 'armazenamento_dados_1995_2014';
% Cria nova pasta se não existir
if ~isfolder(nome_da_pasta)
    mkdir(nome_da_pasta);
end
save(fullfile(nome_da_pasta, 'nortada_vel_wrfout_ERA5.mat'),'tabela_dados_ERA5')


% Guarda a tabela_dados_ERA5 na pasta 'tabelas_1995_2014_em_csv'
nome_da_pasta_2 = 'tabelas_1995_2014_em_csv';
% Cria nova pasta se não existir
if ~isfolder(nome_da_pasta_2)
    mkdir(nome_da_pasta_2);
end
csvFile = 'tabela_dados_ERA5.csv';
writetable(tabela_dados_ERA5, fullfile(nome_da_pasta_2, csvFile));


% Matrizes agrupadas numa estrutura
vento_Barra_wrfout_ERA5.u10_12h_Barra = u10_12h_Barra;
vento_Barra_wrfout_ERA5.v10_12h_Barra = v10_12h_Barra;
vento_Barra_wrfout_ERA5.u10_18h_Barra = u10_18h_Barra;
vento_Barra_wrfout_ERA5.v10_18h_Barra = v10_18h_Barra;
vento_Barra_wrfout_ERA5.vel_12h_Barra = vel_12h_Barra;
vento_Barra_wrfout_ERA5.vel_18h_Barra = vel_18h_Barra;
vento_Barra_wrfout_ERA5.media_direcoes_diarias_Barra = media_direcoes_diarias_Barra;
vento_Barra_wrfout_ERA5.lon_BARRA = lon_BARRA;
vento_Barra_wrfout_ERA5.lat_BARRA = lat_BARRA;
save(fullfile(nome_da_pasta, 'vento_Barra_wrfout_ERA5.mat'), 'vento_Barra_wrfout_ERA5', '-v7.3');


vento_wrfout_ERA5.u10_12h = u10_12h;
vento_wrfout_ERA5.v10_12h = v10_12h;
vento_wrfout_ERA5.u10_18h = u10_18h;
vento_wrfout_ERA5.v10_18h = v10_18h;
vento_wrfout_ERA5.vel_12h = vel_12h;
vento_wrfout_ERA5.vel_18h = vel_18h;
vento_wrfout_ERA5.media_direcoes_diarias = media_direcoes_diarias;
vento_wrfout_ERA5.lon = lon;
vento_wrfout_ERA5.lat = lat;
save(fullfile(nome_da_pasta, 'vento_wrfout_ERA5.mat'), 'vento_wrfout_ERA5', '-v7.3');


writematrix(lon,  fullfile(nome_da_pasta_2, 'lon.csv'));
writematrix(lat,  fullfile(nome_da_pasta_2, 'lat.csv'));
writematrix(vel_12h,  fullfile(nome_da_pasta_2, 'vel_12h.csv'));
writematrix(vel_18h,  fullfile(nome_da_pasta_2, 'vel_18h.csv'));


%-----------------------------------
% NÚMERO DE DIAS COM NORTADA POR ANO 

% Dados agrupados pela coluna de ano
grupos_por_ano = findgroups(tabela_dados_ERA5.Ano);

% Número de dias com nortada por cada grupo (ano)
dias_com_nortada_por_ano = splitapply(@sum, tabela_dados_ERA5.("Nortada Médias Diárias"), grupos_por_ano);
dias_com_nortada_12h_por_ano = splitapply(@sum, tabela_dados_ERA5.("Nortada Média (12h)"), grupos_por_ano);
dias_com_nortada_18h_por_ano = splitapply(@sum, tabela_dados_ERA5.("Nortada Média (18h)"), grupos_por_ano);

% Criar uma nova tabela com os resultados
anual_ERA5 = table(unique(tabela_dados_ERA5.Ano), dias_com_nortada_por_ano, 'VariableNames', {'Ano', 'Número de Dias com Nortada'});
anual_ERA5 = addvars(anual_ERA5, dias_com_nortada_12h_por_ano, dias_com_nortada_18h_por_ano, 'NewVariableNames', {'NDias c/ Nortada(12h)','NDias c/ Nortada(18h)'});


nome_da_pasta = 'armazenamento_dados_1995_2014';
% Cria nova pasta se não existir
if ~isfolder(nome_da_pasta)
    mkdir(nome_da_pasta);
end
save(fullfile(nome_da_pasta, 'anual_ERA5.mat'), 'anual_ERA5');


%-------------------------------------
% VELOCIDADE MÉDIA DOS DIAS C/ NORTADA

indices_dias_nortada = find(tabela_dados_ERA5.("Nortada Médias Diárias"));
velocidade_media_nortada = tabela_dados_ERA5.("Velocidade média diária")(indices_dias_nortada);

% Splitapply para calcular a média da velocidade média para cada ano, considerando apenas os dias com nortada
vel_nortada_por_ano = splitapply(@mean, velocidade_media_nortada, findgroups(tabela_dados_ERA5.Ano(indices_dias_nortada)));

anual_ERA5 = addvars(anual_ERA5, vel_nortada_por_ano, 'NewVariableNames', {'Vel média c/ nortada'});
save(fullfile(nome_da_pasta, 'anual_ERA5.mat'), 'anual_ERA5');


nome_da_pasta_2 = 'tabelas_1995_2014_em_csv';
% Cria nova pasta se não existir
if ~isfolder(nome_da_pasta_2)
    mkdir(nome_da_pasta_2);
end
writetable(anual_ERA5, fullfile(nome_da_pasta_2, 'anual_ERA5.csv'));


%--------------------------------------------------
% Nº DE DIAS C/ NORTADA POR CADA ESTAÇÃO (1995-2014)

ordem_estacoes = {'Inverno', 'Primavera', 'Verão', 'Outono'};
estacao_ERA5= table();
anos_unicos = unique(tabela_dados_ERA5.Ano);

for i = 1:length(anos_unicos)
    dados_ano_atual = tabela_dados_ERA5(tabela_dados_ERA5.Ano == anos_unicos(i), :);
    estacoes_ano_unicas = ordem_estacoes(ismember(ordem_estacoes, unique(dados_ano_atual.("Estações"))));
    num_dias_nortada_por_estacao = zeros(length(estacoes_ano_unicas), 1);

    for j = 1:length(estacoes_ano_unicas)

        dados_estacao_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacoes_ano_unicas{j}), :);
        num_dias_nortada = sum(dados_estacao_atual.("Nortada Médias Diárias"));
        estacao_ERA5 = [estacao_ERA5; table(anos_unicos(i), estacoes_ano_unicas(j), num_dias_nortada)];
    end
end

estacao_ERA5.Properties.VariableNames = {'Ano', 'Estação', 'Nº dias c/ nortada média'};
save(fullfile(nome_da_pasta, 'estacao_ERA5.mat'), 'estacao_ERA5');




%----------------------------------------------------------------
% VELOCIDADE MÉDIA DO NÚMERO DE DIAS COM NORTADA POR CADA ESTAÇÃO

dados_com_nortada = tabela_dados_ERA5(tabela_dados_ERA5.("Nortada Médias Diárias") == 1, :);
estacoes = tabela_dados_ERA5.("Estações")(indices_dias_nortada);
anos = tabela_dados_ERA5.Ano(indices_dias_nortada);
vel_nortada_por_estacao = splitapply(@mean, velocidade_media_nortada, findgroups(anos, estacoes));

dados_sem_nortada = tabela_dados_ERA5(tabela_dados_ERA5.("Nortada Médias Diárias") == 1, :);
estacoes = tabela_dados_ERA5.("Estações")(indices_dias_nortada);

estacao_ERA5 = addvars(estacao_ERA5, vel_nortada_por_estacao, 'NewVariableNames', {'Vel média c/ nortada'});
save(fullfile(nome_da_pasta, 'estacao_ERA5.mat'), 'estacao_ERA5');

writetable(estacao_ERA5, fullfile(nome_da_pasta_2, 'estacao_ERA5.csv'));




%% DO FICHEIRO "wrfout_MPI_hist_UV10m_12_18h.nc"

clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\Circular_Statistics');
addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\ficheiros');

file_name = "wrfout_MPI_hist_UV10m_12_18h.nc";

lon = double(ncread(file_name, 'XLONG'));  % unidades: graus a este
lat = double(ncread(file_name, 'XLAT'));   % unidades: graus a norte
u10 = double(ncread(file_name, 'U10'));
v10 = double(ncread(file_name, 'V10'));
dados_tempo = double( ncread(file_name, "XTIME"));  % unidades: horas desde 1994-12-8 00:00:00

% Converter tempo para data e hora
tempo = datevec(hours(dados_tempo) + datetime(1994, 12, 8));  % ano|mês|dia|hora|min|seg
ano = tempo(:,1); mes = tempo(:,2); dia = tempo(:,3); hora = tempo(:,4);


% Carregar os dados do arquivo NetCDF
lon = double(ncread(file_name, 'XLONG'));  % unidades: graus a este
lat = double(ncread(file_name, 'XLAT'));   % unidades: graus a norte
u10 = double(ncread(file_name, 'U10'));
v10 = double(ncread(file_name, 'V10'));
dados_tempo = double(ncread(file_name, "XTIME"));  % unidades: horas desde 1994-12-8 00:00:00

% Converter tempo para data e hora
tempo = datevec(hours(dados_tempo) + datetime(1994, 12, 8));  % ano|mês|dia|hora|min|seg
ano = tempo(:,1); mes = tempo(:,2); dia = tempo(:,3); hora = tempo(:,4);

% Coordenadas da Praia da Barra
lon_Barra = [-8.7578430, -8.7288208]; % este-oeste
lat_Barra = [40.6077614, 40.6470909]; % norte-sul

% Posições da lat e da lon para a região da Barra
lon_pos = find(lon >= lon_Barra(1) & lon <= lon_Barra(2));
lat_pos = find(lat >= lat_Barra(1) & lat <= lat_Barra(2));

[LON,LAT]=meshgrid(lon_pos,lat_pos);

%ÍNDICES das posições de lon e lat
[lon_row,lon_col] = ind2sub(size(lon_pos),lon_pos);
[lat_row,lat_col] = ind2sub(size(lat_pos),lat_pos);

% Posições correspondentes às 12h e 18h
posicoes_12h = find(hora == 12); 
posicoes_18h = find(hora == 18); 

% Dados das componentes de vento para a malha inicial
u10_12h = u10(:, :, posicoes_12h);
v10_12h = v10(:, :, posicoes_12h);
u10_18h = u10(:, :, posicoes_18h);
v10_18h = v10(:, :, posicoes_18h);

% Dados das componentes de vento dentro da região de interesse
u10_12h_Barra = u10(lon_row, lat_col, posicoes_12h);
v10_12h_Barra = v10(lon_row, lat_col, posicoes_12h);
u10_18h_Barra = u10(lon_row, lat_col, posicoes_18h);
v10_18h_Barra = v10(lon_row, lat_col, posicoes_18h);


% Velocidade [sqrt(u.^2 + v.^2)]
vel_12h = sqrt(u10_12h.^2 + v10_12h.^2);
vel_18h = sqrt(u10_18h.^2 + v10_18h.^2);

% Velocidade [sqrt(u.^2 + v.^2)]
vel_12h_Barra = sqrt(u10_12h_Barra.^2 + v10_12h_Barra.^2);
vel_18h_Barra = sqrt(u10_18h_Barra.^2 + v10_18h_Barra.^2);

media_vel_12h = squeeze(mean(vel_12h_Barra, [1 2])); % Média da velocidade do vento para as 12h ao longo da dimensão temporal
media_vel_18h = squeeze(mean(vel_18h_Barra, [1 2])); % Média da velocidade do vento para as 18h ao longo da dimensão temporal
media_vel_12h_18h = (media_vel_12h + media_vel_18h) / 2;


% MÉDIAS VETORES
% Médias de várias amostras das direções do vento para as 12h e para as 18h
medias_direcao_12h_Barra = circ_mean_degrees(reshape(atan2d(v10_12h_Barra, u10_12h_Barra), [], size(u10_12h_Barra, 3)), [], 1);
medias_direcao_18h_Barra = circ_mean_degrees(reshape(atan2d(v10_18h_Barra, u10_18h_Barra), [], size(u10_18h_Barra, 3)), [], 1);
% Médias diárias das 12h com as 18h
media_direcao_diaria_Barra = circ_mean_degrees([medias_direcao_12h_Barra; medias_direcao_18h_Barra], [], 1);


% MÉDIAS MATRIZES 
media_dir_v10_12h_Barra = circ_mean_degrees(atan2d(v10_12h_Barra, u10_12h_Barra), [], 3);
media_dir_v10_18h_Barra = circ_mean_degrees(atan2d(v10_18h_Barra, u10_18h_Barra), [], 3);
media_direcoes_diarias_Barra = (media_dir_v10_12h_Barra + media_dir_v10_18h_Barra) / 2;

media_dir_v10_12h = circ_mean_degrees(atan2d(v10_12h, u10_12h), [], 3);
media_dir_v10_18h = circ_mean_degrees(atan2d(v10_18h, u10_18h), [], 3);
media_direcoes_diarias= (media_dir_v10_12h + media_dir_v10_18h) / 2;


% NORTADA
nortada_media_diaria = (media_direcao_diaria_Barra >= 335 | media_direcao_diaria_Barra <= 25);
nortada_media_12h = (medias_direcao_12h_Barra >= 335 | medias_direcao_12h_Barra <= 25);
nortada_media_18h = (medias_direcao_18h_Barra >= 335 | medias_direcao_18h_Barra <= 25);



% TABELA COM AS VARIÁVEIS IMPORTANTES PARA CÁLCULOS DA REGIÃO DA PRAIA DA BARRA
tabela_dados_MPI = table(ano(posicoes_12h), mes(posicoes_12h), dia(posicoes_12h), 'VariableNames', {'Ano', 'Mês', 'Dia'});

% Adiciona novas variáveis à tabela
tabela_dados_MPI = addvars(tabela_dados_MPI, medias_direcao_12h_Barra', medias_direcao_18h_Barra', media_direcao_diaria_Barra', 'NewVariableNames', {'Direções médias (12h)', 'Direções médias (18h)', 'Direções médias diárias'});
tabela_dados_MPI = addvars(tabela_dados_MPI, nortada_media_diaria', nortada_media_12h', nortada_media_18h', 'NewVariableNames', {'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)'});
tabela_dados_MPI = addvars(tabela_dados_MPI, media_vel_12h, media_vel_18h, media_vel_12h_18h, 'NewVariableNames', {'Velocidade média(12h)', 'Velocidade média(18h)', 'Velocidade média diária'});


% Adiciona uma nova coluna com as estações do ano 
estacoes = arrayfun(@(x) month2season(x), tabela_dados_MPI.("Mês"), 'UniformOutput', false);
tabela_dados_MPI = addvars(tabela_dados_MPI, estacoes, 'After', 'Dia', 'NewVariableNames', 'Estações');


nome_da_pasta = 'armazenamento_dados_1995_2014';
if ~isfolder(nome_da_pasta)
    mkdir(nome_da_pasta);
end
save(fullfile(nome_da_pasta, 'nortada_vel_wrfout_MPI_hist.mat'),'tabela_dados_MPI')


%Para guardar em csv
nome_da_pasta_2 = 'tabelas_1995_2014_em_csv';
if ~isfolder(nome_da_pasta_2)
    mkdir(nome_da_pasta_2);
end
csvFile = 'tabela_dados_MPI.csv';
writetable(tabela_dados_MPI, fullfile(nome_da_pasta_2, csvFile));


% Matrizes agrupadas numa estrutura
vento_Barra_wrfout_MPI_hist.u10_12h_Barra = u10_12h_Barra;
vento_Barra_wrfout_MPI_hist.v10_12h_Barra = v10_12h_Barra;
vento_Barra_wrfout_MPI_hist.u10_18h_Barra = u10_18h_Barra;
vento_Barra_wrfout_MPI_hist.v10_18h_Barra = v10_18h_Barra;
vento_Barra_wrfout_MPI_hist.vel_12h_Barra = vel_12h_Barra;
vento_Barra_wrfout_MPI_hist.vel_18h_Barra = vel_18h_Barra;
save(fullfile(nome_da_pasta, 'vento_Barra_wrfout_MPI_hist.mat'), 'vento_Barra_wrfout_MPI_hist', '-v7.3');


vento_wrfout_MPI_hist.u10_12h = u10_12h;
vento_wrfout_MPI_hist.v10_12h = v10_12h;
vento_wrfout_MPI_hist.u10_18h = u10_18h;
vento_wrfout_MPI_hist.v10_18h = v10_18h;
vento_wrfout_MPI_hist.vel_12h= vel_12h ;
vento_wrfout_MPI_hist.vel_18h  = vel_18h ;
vento_wrfout_MPI_hist.media_direcoes_diarias = media_direcoes_diarias;
vento_wrfout_MPI_hist.lon = lon;
vento_wrfout_MPI_hist.lat = lat;
save(fullfile(nome_da_pasta, 'vento_wrfout_MPI_hist.mat'), 'vento_wrfout_MPI_hist', '-v7.3');



%-----------------------------------
% NÚMERO DE DIAS COM NORTADA POR ANO 

% Dados agrupados pela coluna de ano
grupos_por_ano = findgroups(tabela_dados_MPI.Ano);

% Número de dias com nortada por cada grupo (ano)
dias_com_nortada_por_ano = splitapply(@sum, tabela_dados_MPI.("Nortada Médias Diárias"), grupos_por_ano);
dias_com_nortada_12h_por_ano = splitapply(@sum, tabela_dados_MPI.("Nortada Média (12h)"), grupos_por_ano);
dias_com_nortada_18h_por_ano = splitapply(@sum, tabela_dados_MPI.("Nortada Média (18h)"), grupos_por_ano);

% Criar uma nova tabela com os resultados
anual_MPI = table(unique(tabela_dados_MPI.Ano), dias_com_nortada_por_ano, 'VariableNames', {'Ano', 'Número de Dias com Nortada'});
anual_MPI = addvars(anual_MPI, dias_com_nortada_12h_por_ano, dias_com_nortada_18h_por_ano, 'NewVariableNames', {'NDias c/ Nortada(12h)','NDias c/ Nortada(18h)'});



nome_da_pasta = 'armazenamento_dados_1995_2014';
% Cria nova pasta se não existir
if ~isfolder(nome_da_pasta)
    mkdir(nome_da_pasta);
end
save(fullfile(nome_da_pasta, 'anual_MPI.mat'), 'anual_MPI');


%-------------------------------------
% VELOCIDADE MÉDIA DOS DIAS C/ NORTADA

indices_dias_nortada = find(tabela_dados_MPI.("Nortada Médias Diárias"));
velocidade_media_nortada = tabela_dados_MPI.("Velocidade média diária")(indices_dias_nortada);

% Splitapply para calcular a média da velocidade média para cada ano, considerando apenas os dias com nortada
vel_nortada_por_ano = splitapply(@mean, velocidade_media_nortada, findgroups(tabela_dados_MPI.Ano(indices_dias_nortada)));

anual_MPI = addvars(anual_MPI, vel_nortada_por_ano, 'NewVariableNames', {'Vel média c/ nortada'});
save(fullfile(nome_da_pasta, 'anual_MPI.mat'), 'anual_MPI');

nome_da_pasta_2 = 'tabelas_1995_2014_em_csv';
% Cria nova pasta se não existir
if ~isfolder(nome_da_pasta_2)
    mkdir(nome_da_pasta_2);
end
writetable(anual_MPI, fullfile(nome_da_pasta_2, 'anual_MPI.csv'));



%--------------------------------------------------
% Nº DE DIAS C/ NORTADA POR CADA ESTAÇÃO (1995-2014)

ordem_estacoes = {'Inverno', 'Primavera', 'Verão', 'Outono'};
estacao_MPI= table();
anos_unicos = unique(tabela_dados_MPI.Ano);

for i = 1:length(anos_unicos)
    dados_ano_atual = tabela_dados_MPI(tabela_dados_MPI.Ano == anos_unicos(i), :);
    estacoes_ano_unicas = ordem_estacoes(ismember(ordem_estacoes, unique(dados_ano_atual.("Estações"))));
    num_dias_nortada_por_estacao = zeros(length(estacoes_ano_unicas), 1);

    for j = 1:length(estacoes_ano_unicas)

        dados_estacao_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacoes_ano_unicas{j}), :);
        num_dias_nortada = sum(dados_estacao_atual.("Nortada Médias Diárias"));
        estacao_MPI = [estacao_MPI; table(anos_unicos(i), estacoes_ano_unicas(j), num_dias_nortada)];
    end
end

estacao_MPI.Properties.VariableNames = {'Ano', 'Estação', 'Nº dias c/ nortada média'};
save(fullfile(nome_da_pasta, 'estacao_MPI.mat'), 'estacao_MPI');




%----------------------------------------------------------------
% VELOCIDADE MÉDIA DO NÚMERO DE DIAS COM NORTADA POR CADA ESTAÇÃO

dados_com_nortada = tabela_dados_MPI(tabela_dados_MPI.("Nortada Médias Diárias") == 1, :);
estacoes = tabela_dados_MPI.("Estações")(indices_dias_nortada);
anos = tabela_dados_MPI.Ano(indices_dias_nortada);
vel_nortada_por_estacao = splitapply(@mean, velocidade_media_nortada, findgroups(anos, estacoes));


estacao_MPI = addvars(estacao_MPI, vel_nortada_por_estacao, 'NewVariableNames', {'Vel média c/ nortada'});
save(fullfile(nome_da_pasta, 'estacao_MPI.mat'), 'estacao_MPI');

writetable(estacao_MPI, fullfile(nome_da_pasta_2, 'estacao_MPI.csv'));


