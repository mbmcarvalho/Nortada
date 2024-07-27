%% NOTAS CÓDIGO MATLAB

%ncdisp(file_name); lê o que está no ficheiro "file_name"


%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

% %Limites coordenadas
% geolimits([40.6077614 40.6470909],[-8.7578430 -8.7288208])
% geobasemap streets

%É preciso o m_map!


%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

% % Plot da malha inicial de lon e lat e da praia da Barra

% figure;
% plot(lon, lat, '+g');
% hold on;
% plot(lon(LON), lat(LAT), 'bo'); % região da Barra 
% title('Malha Inicial de Longitude e Latitude');
% xlabel('Longitude');
% ylabel('Latitude');
% grid on;
% 
% % Limites do gráfico
% xlim([min(lon(:)) max(lon(:))]);
% ylim([min(lat(:)) max(lat(:))]);
% y
% % Retângulo que delimita a região da Barra
% rectangle('Position', [lon_Barra(1), lat_Barra(1), lon_Barra(2) - lon_Barra(1), lat_Barra(2) - lat_Barra(1)], 'EdgeColor', 'r', 'LineWidth', 2);
% 
% legend('Malha Inicial', 'Região da Praia da Barra');


%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

% [LON,LAT]=meshgrid(lon_pos,lat_pos);
%LON é a matriz onde cada linha é uma cópia de lon_pos
%LAT é a matriz onde cada linha é uma cópia de lat_pos

%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>














clear all; close all; clc;

file_name_1="wrfout_ERA5_UV10m_12_18h.nc"
%ncdisp(file_name_1)
addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\Circular_Statistics');


% %Limites coordenadas
% geolimits([40.6075 40.6457],[-8.7579 -8.727583])
% geobasemap streets


lon = ncread(file_name_1, 'XLONG');  %units: degree_east
lat = ncread(file_name_1, 'XLAT');   %units: degree_north
u10 = ncread(file_name_1, 'U10');
v10 = ncread(file_name_1, 'V10');     %disp(v10(1:5, 1:5, 1));
dados_tempo= ncread(file_name_1,"XTIME");  %units: hours since 1994-12-8 00:00:00

tempo = datevec(hours(dados_tempo) + datetime(1994, 12, 8));  % ano|mês|dia|hora|min|seg
ano=tempo(:,1); mes=tempo(:,2); dia=tempo(:,3); hora=tempo(:,4);



% Indices das 12h e 18h
indices_12h = find(hora == 12);
indices_18h = find(hora == 18);

% Dados de u10 e v10 nas horas desejadas
u10_12h = u10(:,:,indices_12h);
v10_12h = v10(:,:,indices_12h);
u10_18h = u10(:,:,indices_18h);
v10_18h = v10(:,:,indices_18h);

% Calcular a média das componentes u e v para cada dia
u_media = mean(cat(3, u10_12h, u10_18h), 2);
v_media = mean(cat(3, v10_12h, v10_18h), 2);


direcao_vento_media = mod(atan2(v_media, u_media) * 180 / pi, 360);
media_direcao_vento_por_dia = circ_mean_degrees(direcao_vento_media, [], 2);

% Encontrar todos os anos presentes nos dados
anos_distintos = unique(ano);

% Inicializar o vetor para armazenar o número de dias com nortada por ano
num_dias_nortada_por_ano = zeros(size(anos_distintos));

% Iterar sobre os anos distintos
for i = 1:length(anos_distintos)
    % Encontrar os índices dos dias pertencentes ao ano atual
    indices_ano_atual = find(ano == anos_distintos(i));
    
    % Calcular nortada apenas para os dias correspondentes ao ano atual
    nortada = (media_direcao_vento_por_dia(indices_ano_atual) >= 335 | media_direcao_vento_por_dia(indices_ano_atual) <= 25);
    
    % Contar o número de dias com nortada no ano atual
    num_dias_nortada_por_ano(i) = sum(nortada);
end

% Plotar o número de dias com nortada por ano
bar(anos_distintos, num_dias_nortada_por_ano);
xlabel('Ano');
ylabel('Número de dias com Nortada');
title('Número de dias com Nortada por Ano');


for i = 1:length(anos_distintos)
    disp(['Ano ', num2str(anos_distintos(i)), ': ', num2str(num_dias_nortada_por_ano(i)), ' dias com nortada']);
end





for i = 1:length(anos_distintos)
    % Encontrar os índices dos dias pertencentes ao ano atual
    indices_ano_atual = find(ano == anos_distintos(i));
    
    % Iterar sobre os meses
    for j = 1:12
        % Encontrar os índices dos dias pertencentes ao mês atual
        indices_mes_atual = find(mes(indices_ano_atual) == j);
        
        % Calcular nortada apenas para os dias correspondentes ao ano e mês atual
        nortada_mes = (media_direcao_vento_por_dia(indices_ano_atual(indices_mes_atual)) >= 335 | media_direcao_vento_por_dia(indices_ano_atual(indices_mes_atual)) <= 25);
        
        % Contar o número de dias com nortada no mês atual
        num_dias_nortada_por_mes(i, j) = sum(nortada_mes);
    end
end

% Imprimir o número de dias com nortada por mês para cada ano
for i = 1:length(anos_distintos)
    for j = 1:12
        disp(['Ano ', num2str(anos_distintos(i)), ', Mês ', num2str(j), ': ', num2str(num_dias_nortada_por_mes(i, j)), ' dias com nortada']);
    end
end




% Tamanho da matriz u10
[tamanho_x, tamanho_y, tamanho_z] = size(u10_12h);

% Acesso aos cantos da primeira profundidade (1)
canto_superior_esquerdo = u10(1, 1, 1);
canto_superior_direito = u10(1, tamanho_y, 1);
canto_inferior_esquerdo = u10(tamanho_x, 1, 1);
canto_inferior_direito = u10(tamanho_x, tamanho_y, 1);

% Exibição dos valores dos cantos
disp(['Canto Superior Esquerdo: ', num2str(canto_superior_esquerdo)]);
disp(['Canto Superior Direito: ', num2str(canto_superior_direito)]);
disp(['Canto Inferior Esquerdo: ', num2str(canto_inferior_esquerdo)]);
disp(['Canto Inferior Direito: ', num2str(canto_inferior_direito)]);



% Tamanho da matriz u10
[tamanho_x, tamanho_y, tamanho_z] = size(u10_12h);

% Acesso aos cantos da última profundidade (14610)
canto_superior_esquerdo = u10(1, 1, tamanho_z);
canto_superior_direito = u10(1, tamanho_y, tamanho_z);
canto_inferior_esquerdo = u10(tamanho_x, 1, tamanho_z);
canto_inferior_direito = u10(tamanho_x, tamanho_y, tamanho_z);

% Exibição dos valores dos cantos
disp(['Canto Superior Esquerdo: ', num2str(canto_superior_esquerdo)]);
disp(['Canto Superior Direito: ', num2str(canto_superior_direito)]);
disp(['Canto Inferior Esquerdo: ', num2str(canto_inferior_esquerdo)]);
disp(['Canto Inferior Direito: ', num2str(canto_inferior_direito)]);

% Suponha que você tenha calculado a média diária das direções do vento (media_direcao_vento_por_dia) e armazenado em uma variável.

% Defina os anos de interesse
anos_interesse = [2000, 2001, 2002]; % Pode ser um único ano ou uma lista de anos

% Inicialize uma matriz para armazenar o número de dias com nortada para cada estação em cada ano
num_dias_nortada_por_estacao = zeros(length(anos_interesse), 4); % 4 estações

% Loop através de cada ano de interesse
for i = 1:length(anos_interesse)
    % Obtenha o índice dos dias correspondentes ao ano atual
    indices_ano_atual = find(ano == anos_interesse(i));

    % Defina os índices para cada estação
    indices_primavera = find(mes(indices_ano_atual) >= 3 & mes(indices_ano_atual) <= 5); % Março, Abril e Maio
    indices_verao = find(mes(indices_ano_atual) >= 6 & mes(indices_ano_atual) <= 8);     % Junho, Julho e Agosto
    indices_outono = find(mes(indices_ano_atual) >= 9 & mes(indices_ano_atual) <= 11);   % Setembro, Outubro e Novembro
    indices_inverno = find(mes(indices_ano_atual) == 12 | mes(indices_ano_atual) <= 2);  % Dezembro, Janeiro e Fevereiro

    % Calcule o número de dias com nortada para cada estação no ano atual
    num_dias_nortada_por_estacao(i, 1) = sum(media_direcao_vento_por_dia(indices_ano_atual(indices_primavera)) >= 335 | media_direcao_vento_por_dia(indices_ano_atual(indices_primavera)) <= 25);
    num_dias_nortada_por_estacao(i, 2) = sum(media_direcao_vento_por_dia(indices_ano_atual(indices_verao)) >= 335 | media_direcao_vento_por_dia(indices_ano_atual(indices_verao)) <= 25);
    num_dias_nortada_por_estacao(i, 3) = sum(media_direcao_vento_por_dia(indices_ano_atual(indices_outono)) >= 335 | media_direcao_vento_por_dia(indices_ano_atual(indices_outono)) <= 25);
    num_dias_nortada_por_estacao(i, 4) = sum(media_direcao_vento_por_dia(indices_ano_atual(indices_inverno)) >= 335 | media_direcao_vento_por_dia(indices_ano_atual(indices_inverno)) <= 25);
end

% Agora você pode plotar os resultados
figure;
bar(anos_interesse, num_dias_nortada_por_estacao, 'stacked');
xlabel('Ano');
ylabel('Número de dias com Nortada');
title('Número de dias com Nortada por Estação (Agrupado por Ano)');
legend('Primavera', 'Verão', 'Outono', 'Inverno', 'Location', 'northwest');







%PROCURA DE ÍNDICE DESEJADO

% Latitude específica que você deseja encontrar
latitude_desejada = 40.28;

% Encontrar o índice da latitude mais próxima
[~, indice_latitude] = min(abs(lat(:) - latitude_desejada));

% Converter o índice linear em índices de linha e coluna
[indice_linha, indice_coluna] = ind2sub(size(lat), indice_latitude);

% Imprimir a posição da latitude específica
disp(['A posição da latitude ', num2str(latitude_desejada), ' é:']);
disp(['Índice da linha: ', num2str(indice_linha)]);
disp(['Índice da coluna: ', num2str(indice_coluna)]);















%% estações do ano notas

% 
% % Obter anos únicos
% anos = unique(tabela_dados.Ano); 
% 
% % Inicializar célula para armazenar os dados agrupados
% dados_agrupados = cell(length(anos), 1);
% 
% % Loop sobre cada ano
% for i = 1:length(anos)
%     ano_atual = anos(i);
% 
%     % Filtrar os dados para o ano atual
%     dados_ano_atual = tabela_dados(tabela_dados.Ano == ano_atual, :);
% 
%     % Obter estações do ano únicas para o ano atual
%     estacoes_ano_unicas = unique(dados_ano_atual.("Estações"));
% 
%     % Inicializar célula para armazenar os dados agrupados para o ano atual
%     dados_agrupados_ano_atual = cell(length(estacoes_ano_unicas), 1);
% 
%     % Loop sobre cada estação do ano única para o ano atual
%     for j = 1:length(estacoes_ano_unicas)
%         estacao_ano_atual = estacoes_ano_unicas{j};
% 
%         % Filtrar os dados para a estação do ano atual
%         dados_estacao_ano_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacao_ano_atual), :);
% 
%         % Armazenar os dados agrupados para a estação do ano atual
%         dados_agrupados_ano_atual{j} = dados_estacao_ano_atual;
%     end
% 
%     % Armazenar os dados agrupados para o ano atual na célula de dados agrupados
%     dados_agrupados{i} = dados_agrupados_ano_atual;
% 
% 
% end








%------------------ NÚMERO DE DIAS COM NORTADA EM 2004 PARA CADA ESTAÇÃO---------------------

indices_ano_especifico = find(tabela_dados.Ano == 2004);
dados_ano_especifico = tabela_dados(indices_ano_especifico, :);
estacoes_ano_unicas = unique(dados_ano_especifico.("Estações"));

% Vetor para armazenar o número de dias com nortada para cada estação do ano
num_dias_nortada_por_estacao_ano = zeros(length(estacoes_ano_unicas), 1);


for i = 1:length(estacoes_ano_unicas)
    estacao_ano_atual = estacoes_ano_unicas{i};
    dados_estacao_ano_atual = dados_ano_especifico(strcmp(dados_ano_especifico.("Estações"), estacao_ano_atual), :);

    num_dias_nortada = sum(dados_estacao_ano_atual.Nortada);

    num_dias_nortada_por_estacao_ano(i) = num_dias_nortada;
end


figure;
bar(num_dias_nortada_por_estacao_ano);
xticks(1:length(estacoes_ano_unicas));
xticklabels(estacoes_ano_unicas);
title(['Número de Dias com Nortada por Estação do Ano em ', num2str(2004)]);
xlabel('Estação do Ano');
ylabel('Número de Dias com Nortada');







%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


% 
% 
% 
% 
% 
% num_dias_12h = size(u10_12h_Barra, 3);
% num_dias_18h = size(u10_18h_Barra, 3);
% 
% medias_direcao_12h = zeros(1, num_dias_12h);
% medias_direcao_18h = zeros(1, num_dias_18h);
% 
% 
% for i = 1:num_dias_12h
%     medias_direcao_12h(i) = circ_mean_degrees(reshape(atan2d(v10_12h_Barra(:,:,i), u10_12h_Barra(:,:,i)), [], 1));
% end
% 
% for i = 1:num_dias_18h
%     medias_direcao_18h(i) = circ_mean_degrees(reshape(atan2d(v10_18h_Barra(:,:,i), u10_18h_Barra(:,:,i)), [], 1));
% end
% 
% media_direcao_diaria = circ_mean_degrees([medias_direcao_12h; medias_direcao_18h], [], 1);
% 
% 
% 
% nortada = (media_direcao_diaria >= 335 | media_direcao_diaria <= 25);
% 


%
% % Identificar os anos únicos presentes nos seus dados
% anos_unicos = unique(ano);
% 
% % índices dos dias com nortada
% indices_nortada = find(nortada);
% 
% % Extrair os anos correspondentes aos dias com nortada
% anos_nortada = ano(indices_nortada);
% 
% % Contagem do número de dias com nortada por ano
% num_dias_nortada_por_ano = accumarray(anos_nortada - min(anos_nortada) + 1, 1);
% 
% % Exibir o número de dias com nortada por ano
% disp('Ano   | Dias com nortada');
% disp('-------------------------');
% for i = 1:numel(anos_unicos)
%     fprintf('%4d  | %d\n', anos_unicos(i), num_dias_nortada_por_ano(i));
% end







% 
% lonmin=-8.7578430;
% lonmax=-8.7288208;
% latmin=40.6077614;
% latmax=40.6470909;
% 
% fa=((latmax-latmin)/(lonmax-lonmin))*0.84;
% quad=[lonmin lonmax latmin latmax];
% 
% u2=u(lon>=quad(1) & lon<=quad(2),lat>=quad(3) & lat<=quad(4));
% v2=v(lon>=quad(1) & lon<=quad(2),lat>=quad(3) & lat<=quad(4));
% 
% mlon=lon_malha(lon>=quad(1) & lon<=quad(2),lat>=quad(3) & lat<=quad(4));
% mlat=lat_malha(lon>=quad(1) & lon<=quad(2),lat>=quad(3) & lat<=quad(4));   









%--------------------------------------------------
% Nº DE DIAS C/ NORTADA POR CADA ESTAÇÃO (1995-2014)

% anos_unicos = unique(tabela_dados_MPI.Ano);
% estacao_MPI = table();
% 
% for i = 1:length(anos_unicos)
%     dados_ano_atual = tabela_dados_MPI(tabela_dados_MPI.Ano == anos_unicos(i), :);
%     estacoes_ano_unicas = unique(dados_ano_atual.("Estações"));
%     num_dias_nortada_por_estacao = zeros(length(estacoes_ano_unicas), 1);
% 
%     for j = 1:length(estacoes_ano_unicas)
%         dados_estacao_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacoes_ano_unicas{j}), :);
%         num_dias_nortada = sum(dados_estacao_atual.("Nortada Médias Diárias"));
%         num_dias_nortada_por_estacao(j) = num_dias_nortada;
%     end
% 
%     estacao_MPI = [estacao_MPI; table(repmat(anos_unicos(i), length(estacoes_ano_unicas), 1), estacoes_ano_unicas, num_dias_nortada_por_estacao)];
% end
% 
% 
% estacao_MPI.Properties.VariableNames = {'Ano', 'Estação', 'Nº dias c/ nortada média'};
% save(fullfile(nome_da_pasta, 'estacao_MPI.mat'), 'estacao_MPI');
