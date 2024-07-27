clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\dados_wrfout_mpi_ssp\ficheiros_2046_2065');
addpath('C:\Users\Beatriz\Desktop\Projeto\dados_wrfout_mpi_ssp\Circular_Statistics');

file_names = {'wrfout_MPI_ssp245_2046_2065_UV10m_12_18h.nc', 'wrfout_MPI_ssp370_2046_2065_UV10m_12_18h.nc', 'wrfout_MPI_ssp585_2046_2065_UV10m_12_18h.nc'};

for i = 1:numel(file_names)

    file_name = file_names{i};
   
    
    lon = double(ncread(file_name, 'XLONG'));  % unidades: graus a este
    lat = double(ncread(file_name, 'XLAT'));   % unidades: graus a norte
    u10 = double(ncread(file_name, 'U10'));
    v10 = double(ncread(file_name, 'V10'));
    dados_tempo = double(ncread(file_name, "XTIME"));  % unidades: horas desde 1994-12-8 00:00:00
    
    
    % Converter tempo para data e hora
    tempo = datevec(hours(dados_tempo) + datetime(2045, 12, 8));  % ano|mês|dia|hora|min|seg
    ano = tempo(:,1); mes = tempo(:,2); dia = tempo(:,3); hora = tempo(:,4);
    
    % Coordenadas da Praia da Barra
    lon_Barra = [-8.7578430, -8.7288208]; % este-oeste
    lat_Barra = [40.6077614, 40.6470909]; % norte-sul
    
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
    tabela_dados_wrfout_mpi_2046_2065 = table(ano(posicoes_12h), mes(posicoes_12h), dia(posicoes_12h), 'VariableNames', {'Ano', 'Mês', 'Dia'});
    
    % Adiciona novas variáveis à tabela
    tabela_dados_wrfout_mpi_2046_2065 = addvars(tabela_dados_wrfout_mpi_2046_2065, medias_direcao_12h_Barra', medias_direcao_18h_Barra', media_direcao_diaria_Barra', 'NewVariableNames', {'Direções médias (12h)', 'Direções médias (18h)', 'Direções médias diárias'});
    tabela_dados_wrfout_mpi_2046_2065 = addvars(tabela_dados_wrfout_mpi_2046_2065, nortada_media_diaria', nortada_media_12h', nortada_media_18h', 'NewVariableNames', {'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)'});
    tabela_dados_wrfout_mpi_2046_2065 = addvars(tabela_dados_wrfout_mpi_2046_2065, media_vel_12h, media_vel_18h, media_vel_12h_18h, 'NewVariableNames', {'Velocidade média(12h)', 'Velocidade média(18h)', 'Velocidade média diária'});
    
    
    % Adiciona uma nova coluna com as estações do ano 
    estacoes = arrayfun(@(x) month2season(x), tabela_dados_wrfout_mpi_2046_2065.("Mês"), 'UniformOutput', false);
    tabela_dados_wrfout_mpi_2046_2065 = addvars(tabela_dados_wrfout_mpi_2046_2065, estacoes, 'After', 'Dia', 'NewVariableNames', 'Estações');
    
    
    nome_arquivo = sprintf('nortada_vel_%d.mat', i);
    % Guarda a tabela na pasta 'armazenamento_dados_2046_2065'
    nome_da_pasta = 'armazenamento_dados_2046_2065';
    % Cria nova pasta se não existir
    if ~isfolder(nome_da_pasta)
        mkdir(nome_da_pasta);
    end
    save(fullfile(nome_da_pasta, nome_arquivo),'tabela_dados_wrfout_mpi_2046_2065')
    

    nome_arquivo_csv = sprintf('tabela_dados_%d.csv', i);
    nome_da_pasta_2 = 'tabelas_2046_2065_em_csv';
    % Cria nova pasta se não existir
    if ~isfolder(nome_da_pasta_2)
        mkdir(nome_da_pasta_2);
    end
    writetable(tabela_dados_wrfout_mpi_2046_2065, fullfile(nome_da_pasta_2, nome_arquivo_csv));
    
    
    % Matrizes agrupadas numa estrutura
    vento_Barra_wrfout_mpi_2046_2065.u10_12h_Barra = u10_12h_Barra;
    vento_Barra_wrfout_mpi_2046_2065.v10_12h_Barra = v10_12h_Barra;
    vento_Barra_wrfout_mpi_2046_2065.u10_18h_Barra = u10_18h_Barra;
    vento_Barra_wrfout_mpi_2046_2065.v10_18h_Barra = v10_18h_Barra;
    vento_Barra_wrfout_mpi_2046_2065.vel_12h_Barra = vel_12h_Barra;
    vento_Barra_wrfout_mpi_2046_2065.vel_18h_Barra = vel_18h_Barra;
    vento_Barra_wrfout_mpi_2046_2065.media_direcoes_diarias_Barra = media_direcoes_diarias_Barra;
    vento_Barra_wrfout_mpi_2046_2065.lon_BARRA = lon_BARRA;
    vento_Barra_wrfout_mpi_2046_2065.lat_BARRA = lat_BARRA;

    nome_arquivo_vento_Barra_wrfout_mpi_2046_2065 = sprintf('vento_Barra_%d.mat', i);
    save(fullfile(nome_da_pasta, nome_arquivo_vento_Barra_wrfout_mpi_2046_2065), 'vento_Barra_wrfout_mpi_2046_2065', '-v7.3');

    
    vento_wrfout_mpi_2046_2065.u10_12h = u10_12h;
    vento_wrfout_mpi_2046_2065.v10_12h = v10_12h;
    vento_wrfout_mpi_2046_2065.u10_18h = u10_18h;
    vento_wrfout_mpi_2046_2065.v10_18h = v10_18h;
    vento_wrfout_mpi_2046_2065.vel_12h = vel_12h;
    vento_wrfout_mpi_2046_2065.vel_18h = vel_18h;
    vento_wrfout_mpi_2046_2065.media_direcoes_diarias = media_direcoes_diarias;
    vento_wrfout_mpi_2046_2065.lon = lon;
    vento_wrfout_mpi_2046_2065.lat = lat;
    nome_arquivo_vento_wrfout_mpi_2046_2065 = sprintf('vento_%d.mat', i);
    save(fullfile(nome_da_pasta, nome_arquivo_vento_wrfout_mpi_2046_2065), 'vento_wrfout_mpi_2046_2065', '-v7.3');

   
    writematrix(lon,  fullfile(nome_da_pasta_2, sprintf('lon_%d.csv', i)));
    writematrix(lat,  fullfile(nome_da_pasta_2, sprintf('lat_%d.csv', i)));
    writematrix(vel_12h,  fullfile(nome_da_pasta_2, sprintf('vel_12h_%d.csv', i)));
    writematrix(vel_18h,  fullfile(nome_da_pasta_2, sprintf('vel_18h_%d.csv', i))); 
    
    %-----------------------------------
    % NÚMERO DE DIAS COM NORTADA POR ANO 
    
    % Dados agrupados pela coluna de ano
    grupos_por_ano = findgroups(tabela_dados_wrfout_mpi_2046_2065.Ano);
    
    % Número de dias com nortada por cada grupo (ano)
    dias_com_nortada_por_ano = splitapply(@sum, tabela_dados_wrfout_mpi_2046_2065.("Nortada Médias Diárias"), grupos_por_ano);
    dias_com_nortada_12h_por_ano = splitapply(@sum, tabela_dados_wrfout_mpi_2046_2065.("Nortada Média (12h)"), grupos_por_ano);
    dias_com_nortada_18h_por_ano = splitapply(@sum, tabela_dados_wrfout_mpi_2046_2065.("Nortada Média (18h)"), grupos_por_ano);
    
    % Criar uma nova tabela com os resultados
    anual_mpi_2046_2065 = table(unique(tabela_dados_wrfout_mpi_2046_2065.Ano), dias_com_nortada_por_ano, 'VariableNames', {'Ano', 'Número de Dias com Nortada'});
    anual_mpi_2046_2065 = addvars(anual_mpi_2046_2065, dias_com_nortada_12h_por_ano, dias_com_nortada_18h_por_ano, 'NewVariableNames', {'NDias c/ Nortada(12h)','NDias c/ Nortada(18h)'});
    
    nome_arquivo_3 = sprintf('anual_%d.mat', i);
    nome_da_pasta = 'armazenamento_dados_2046_2065';
    % Cria nova pasta se não existir
    if ~isfolder(nome_da_pasta)
        mkdir(nome_da_pasta);
    end
    save(fullfile(nome_da_pasta, nome_arquivo_3), 'anual_mpi_2046_2065');
    
    
    %-------------------------------------
    % VELOCIDADE MÉDIA DOS DIAS C/ NORTADA
    
    indices_dias_nortada = find(tabela_dados_wrfout_mpi_2046_2065.("Nortada Médias Diárias"));
    velocidade_media_nortada = tabela_dados_wrfout_mpi_2046_2065.("Velocidade média diária")(indices_dias_nortada);
    
    % Splitapply para calcular a média da velocidade média para cada ano, considerando apenas os dias com nortada
    vel_nortada_por_ano = splitapply(@mean, velocidade_media_nortada, findgroups(tabela_dados_wrfout_mpi_2046_2065.Ano(indices_dias_nortada)));
    
    anual_mpi_2046_2065 = addvars(anual_mpi_2046_2065, vel_nortada_por_ano, 'NewVariableNames', {'Vel média c/ nortada'});
    nome_arquivo_4 = sprintf('anual_%d.mat', i);
    save(fullfile(nome_da_pasta, nome_arquivo_4), 'anual_mpi_2046_2065');
    
    nome_arquivo_5 = sprintf('anual_%d.csv', i);
    nome_da_pasta_2 = 'tabelas_2046_2065_em_csv';
    % Cria nova pasta se não existir
    if ~isfolder(nome_da_pasta_2)
        mkdir(nome_da_pasta_2);
    end
    writetable(anual_mpi_2046_2065, fullfile(nome_da_pasta_2, nome_arquivo_5));
    
    
    %--------------------------------------------------
    % Nº DE DIAS C/ NORTADA POR CADA ESTAÇÃO (1995-2014)
    
    ordem_estacoes = {'Inverno', 'Primavera', 'Verão', 'Outono'};
    estacao_mpi_2046_2065= table();
    anos_unicos = unique(tabela_dados_wrfout_mpi_2046_2065.Ano);
    
    for i = 1:length(anos_unicos)
        dados_ano_atual = tabela_dados_wrfout_mpi_2046_2065(tabela_dados_wrfout_mpi_2046_2065.Ano == anos_unicos(i), :);
        estacoes_ano_unicas = ordem_estacoes(ismember(ordem_estacoes, unique(dados_ano_atual.("Estações"))));
        num_dias_nortada_por_estacao = zeros(length(estacoes_ano_unicas), 1);
    
        for j = 1:length(estacoes_ano_unicas)
    
            dados_estacao_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacoes_ano_unicas{j}), :);
            num_dias_nortada = sum(dados_estacao_atual.("Nortada Médias Diárias"));
            estacao_mpi_2046_2065 = [estacao_mpi_2046_2065; table(anos_unicos(i), estacoes_ano_unicas(j), num_dias_nortada)];
        end
    end
    
    estacao_mpi_2046_2065.Properties.VariableNames = {'Ano', 'Estação', 'Nº dias c/ nortada média'};
 
    nome_arquivo_estacao_wrfout_mpi_2046_2065 = sprintf('estacao_%d.mat', i);
    save(fullfile(nome_da_pasta, nome_arquivo_estacao_wrfout_mpi_2046_2065), 'estacao_mpi_2046_2065', '-v7.3');
   
    
    
    %----------------------------------------------------------------
    % VELOCIDADE MÉDIA DO NÚMERO DE DIAS COM NORTADA POR CADA ESTAÇÃO
    
    dados_com_nortada = tabela_dados_wrfout_mpi_2046_2065(tabela_dados_wrfout_mpi_2046_2065.("Nortada Médias Diárias") == 1, :);
    estacoes = tabela_dados_wrfout_mpi_2046_2065.("Estações")(indices_dias_nortada);
    anos = tabela_dados_wrfout_mpi_2046_2065.Ano(indices_dias_nortada);
    vel_nortada_por_estacao = splitapply(@mean, velocidade_media_nortada, findgroups(anos, estacoes));
    
    dados_sem_nortada = tabela_dados_wrfout_mpi_2046_2065(tabela_dados_wrfout_mpi_2046_2065.("Nortada Médias Diárias") == 1, :);
    estacoes = tabela_dados_wrfout_mpi_2046_2065.("Estações")(indices_dias_nortada);
    
    estacao_mpi_2046_2065 = addvars(estacao_mpi_2046_2065, vel_nortada_por_estacao, 'NewVariableNames', {'Vel média c/ nortada'});
    nome_arquivo_estacao_wrfout_mpi_2046_2065 = sprintf('estacao_%d.mat', i);
    save(fullfile(nome_da_pasta, nome_arquivo_estacao_wrfout_mpi_2046_2065), 'estacao_mpi_2046_2065', '-v7.3');
    
    nome_arquivo_6 = sprintf('estacao_%d.csv', i);
    writetable(estacao_mpi_2046_2065, fullfile(nome_da_pasta_2, nome_arquivo_6));



end