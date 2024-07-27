clear all; close all; clc;

file_name_1='ERA5_UV_gust_1994_2022.nc'  % abrange toda a costa atlântica portuguesa
%ncdisp(file_name_1)

%Leitura das Variáveis
lon=ncread(file_name_1,'longitude');  
lat=ncread(file_name_1,'latitude');
dados_tempo=ncread(file_name_1,'time');   %horas desde 01-01-1900 00:00:00
u10=ncread(file_name_1,'u10');
v10=ncread(file_name_1,'v10');
rajada_max=ncread(file_name_1,'i10fg'); %velocidade


% %Limites coordenadas
% geolimits([38 42],[-9.5 -7.5])
% geobasemap streets


%Dados horários de 1994-2022
tempo=datevec(hours(dados_tempo)+datetime(1900,1,1));  % ano|mês|dia|hora|min|seg
ano=tempo(:,1); mes=tempo(:,2); dia=tempo(:,3); hora=tempo(:,4);



% Vento de direção entre 25 e 335 graus
direcao_vento = atan2(v10, u10) * 180 / pi + 180;
nortada = (direcao_vento >= 335 | direcao_vento <= 25);

% Determinar se houve algum evento de nortada/dia, para não repetir dados num só dia
indices_nortada = (direcao_vento >= 335 | direcao_vento <= 25); 
indices_nortada_por_dia = squeeze(any(any(indices_nortada, 1), 2)); % Vetor de booleanos que indica se houve nortada nalgum momento do dia

num_dias_nortada_por_ano = zeros(1, max(ano) - min(ano) + 1);  % Vetor inicial para contar o número de dias com nortada para cada ano
for i = min(ano):max(ano)
    indices_ano_atual = find(ano == i); % Encontrar índices dos dias do ano atual
    
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
title('Número de dias com Nortada por Ano');

total_dias_com_nortada = sum(num_dias_nortada_por_ano); % Número total de dias com nortada em todos os anos
total_anos = length(num_dias_nortada_por_ano); % Número total de anos
media_nortada_por_ano = total_dias_com_nortada / total_anos; 
fprintf('A média do número de dias com nortada por ano é: %.2f\n', media_nortada_por_ano);





%% 
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

% Definição das estações do ano
seasons = {'DJF', 'MAM', 'JJA', 'SON'};
start_month = [12, 3, 6, 9];
end_month = [2, 5, 8, 11];

% Vento de direção entre 25 e 335 graus
direcao_vento = atan2(v10, u10) * 180 / pi + 180;
nortada = (direcao_vento >= 335 | direcao_vento <= 25);

% Inicializar a estrutura de dados para armazenar médias sazonais
media_int_vento = struct();

% Calcular média sazonal de velocidade do vento
for i = 1:length(seasons)
    season = seasons{i};
    % Selecione os meses correspondentes à estação
    indices_mes = mes >= start_month(i) | mes <= end_month(i);
    media_int_vento.(season) = mean(sqrt(u10(:,:,indices_mes).^2 + v10(:,:,indices_mes).^2), 3);
end

% Calcular a direção média do vento para cada estação do ano
mean_direction = struct();

for i = 1:length(seasons)
    season = seasons{i};
    % Selecione os meses correspondentes à estação
    indices_mes = mes >= start_month(i) | mes <= end_month(i);
    mean_direction.(season) = mean(direcao_vento(:,:,ano>=1994 & ano<=2022 & indices_mes), 3);
end

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

for i = 1:length(seasons)
    season = seasons{i};
    
    % Plotar a grade de velocidade do vento
    subplot(2, 2, i);
    m_pcolor(lon, lat, media_int_vento.(season)');
    shading flat;
    m_grid('linestyle', 'none', 'tickdir', 'out', 'linewidth', 1);
    title(season);
    colorbar;
    
    % Adicionar linhas de costa ao subplot
    hold on;
    m_plot(coastlon, coastlat, 'k', 'LineWidth', 1);
    hold off;
    
    % Criar grade de latitude e longitude correspondente
    [lon_grid, lat_grid] = meshgrid(lon, lat);
    
    % Calcular o comprimento das setas baseado na magnitude da velocidade do vento
    arrow_length = 0.5; % comprimento base das setas
    max_wind_speed = max(media_int_vento.(season)(:)); % magnitude máxima da velocidade do vento
    scale_factor = arrow_length / max_wind_speed;
    scaled_u = cosd(mean_direction.(season)) * scale_factor;
    scaled_v = sind(mean_direction.(season)) * scale_factor;
    
    % Plotar as setas de direção média do vento
    hold on;
    m_quiver(lon_grid, lat_grid, scaled_u', scaled_v', 'k');
    hold off;

    % arra legendada
    h = colorbar;
    ylabel(h, 'Velocidade do Vento (m/s)');
end


% Título geral à figura
sgtitle('Velocidade do Vento e Direção Média do Vento: Padrões sazonais');



%% Velocidade média do vento 

clear all; close all; clc;

file_name_1 = 'ERA5_UV_gust_1994_2022.nc';  

% Leitura das Variáveis
lon = ncread(file_name_1, 'longitude');  
lat = ncread(file_name_1, 'latitude');
dados_tempo = ncread(file_name_1, 'time');   % Horas desde 01-01-1900 00:00:00
u10 = ncread(file_name_1, 'u10');
v10 = ncread(file_name_1, 'v10');

% Convertendo tempo para formato de data
tempo = datevec(hours(dados_tempo) + datetime(1900, 1, 1));  % ano|mês|dia|hora|min|seg
ano = tempo(:, 1); dia = tempo(:, 3);

% Determinar se houve algum evento de nortada/dia, para não repetir dados num só dia
indices_nortada = (atan2(v10, u10) * 180 / pi + 180 >= 335 | atan2(v10, u10) * 180 / pi + 180 <= 25); 
indices_nortada_por_dia = squeeze(any(any(indices_nortada, 1), 2)); % Vetor de booleanos que indica se houve nortada nalgum momento do dia

num_dias_nortada_por_ano = zeros(1, max(ano) - min(ano) + 1);  % Vetor inicial para contar o número de dias com nortada para cada ano
for i = min(ano):max(ano)
    indices_ano_atual = find(ano == i); % Encontrar índices dos dias do ano atual
    
    % Verificar se houve pelo menos um dia com nortada no ano atual
    if any(indices_nortada_por_dia(indices_ano_atual))
        % Se sim, contar o número total de dias únicos com nortada para o ano atual
        num_dias_nortada_por_ano(i - min(ano) + 1) = sum(diff([0; indices_nortada_por_dia(indices_ano_atual); 0]) > 0);
    end
end

% Plotagem do gráfico de barras do número de dias com nortada por ano
figure;
bar(min(ano):max(ano), num_dias_nortada_por_ano);
xlabel('Ano');
ylabel('Número de dias com Nortada');
title('Número de dias com Nortada por Ano');

total_dias_com_nortada = sum(num_dias_nortada_por_ano); % Número total de dias com nortada em todos os anos
total_anos = length(num_dias_nortada_por_ano); % Número total de anos
media_nortada_por_ano = total_dias_com_nortada / total_anos; 
fprintf('A média do número de dias com nortada por ano é: %.2f\n', media_nortada_por_ano);

% Cálculo da Velocidade Média do Vento para um Intervalo de 5 Anos
anos_por_periodo = 5;
velocidade_media_periodo = [];
anos_periodo = [];

for i = min(ano):anos_por_periodo:max(ano)
    indices_periodo = i : i + anos_por_periodo - 1;
    u10_periodo = u10(:,:,indices_periodo);
    v10_periodo = v10(:,:,indices_periodo);
    velocidade_media_periodo = [velocidade_media_periodo; mean(sqrt(u10_periodo.^2 + v10_periodo.^2), 'all')];
    anos_periodo = [anos_periodo; mean(indices_periodo)];
end

% Plotagem do gráfico da Velocidade Média do Vento
figure;
plot(anos_periodo, velocidade_media_periodo, 'o-', 'LineWidth', 2);
xlabel('Ano');
ylabel('Velocidade Média do Vento (m/s)');
title('Velocidade Média do Vento de 5 em 5 anos (Costa Atlântica Portuguesa)');

% Adiciona linhas verticais para indicar os limites de cada período de 5 anos
hold on;
for i = 1:length(anos_periodo)
    line([anos_periodo(i) anos_periodo(i)], [min(velocidade_media_periodo) max(velocidade_media_periodo)], 'Color', 'k', 'LineStyle', '--');
end
hold off;


% Calcular a média da velocidade do vento para dias com nortada e sem nortada para cada ano
velocidade_media_com_nortada = zeros(1, total_anos);
velocidade_media_sem_nortada = zeros(1, total_anos);

for i = 1:total_anos
    % Encontrar os índices dos dias com nortada e sem nortada para o ano atual
    indices_ano_atual = find(ano == anos(i));
    dias_com_nortada = indices_nortada_por_dia(indices_ano_atual);
    dias_sem_nortada = ~indices_nortada_por_dia(indices_ano_atual);
    
    % Calcular a média da velocidade do vento para os dias com nortada e sem nortada
    velocidade_media_com_nortada(i) = mean(int_vento(dias_com_nortada));
    velocidade_media_sem_nortada(i) = mean(int_vento(dias_sem_nortada));
end

% Plotar o gráfico de barras
figure;
bar(anos, [velocidade_media_com_nortada', velocidade_media_sem_nortada']);
xlabel('Ano');
ylabel('Velocidade Média do Vento (m/s)');
title('Velocidade Média do Vento com e sem Nortada por Ano');
legend('Com Nortada', 'Sem Nortada');



%% Velocidade do vento c/ ou s/ nortada 


clear all;
close all;
clc;

file_name_1 = 'ERA5_UV_gust_1994_2022.nc';

% Leitura das Variáveis
lon = ncread(file_name_1, 'longitude');  
lat = ncread(file_name_1, 'latitude');
dados_tempo = ncread(file_name_1, 'time');   % Horas desde 01-01-1900 00:00:00
u10 = ncread(file_name_1, 'u10');
v10 = ncread(file_name_1, 'v10');

% Convertendo tempo para formato de data
tempo = datevec(hours(dados_tempo) + datetime(1900, 1, 1));  % ano|mês|dia|hora|min|seg
ano = tempo(:, 1);

% Vento de direção entre 25 e 335 graus
direcao_vento = atan2(v10, u10) * 180 / pi + 180;
int_vento = sqrt(u10.^2 + v10.^2);

% Determinar se houve algum evento de nortada/dia, para não repetir dados num só dia
indices_nortada = (direcao_vento >= 335 | direcao_vento <= 25); 
indices_nortada_por_dia = squeeze(any(any(indices_nortada, 1), 2)); % Vetor de booleanos que indica se houve nortada nalgum momento do dia

% Calculando a média da velocidade do vento com e sem nortada para cada ano
num_dias_nortada_por_ano = zeros(1, max(ano) - min(ano) + 1);  % Vetor inicial para contar o número de dias com nortada para cada ano
for i = min(ano):max(ano)
    indices_ano_atual = find(ano == i); % Encontrar índices dos dias do ano atual
    
    % Verificar se houve pelo menos um dia com nortada no ano atual
    if any(indices_nortada_por_dia(indices_ano_atual))
        % Se sim, contar o número total de dias únicos com nortada para o ano atual
        num_dias_nortada_por_ano(i - min(ano) + 1) = sum(diff([0; indices_nortada_por_dia(indices_ano_atual); 0]) > 0);
    end
end

% Plotar um gráfico de barras comparando a média da velocidade do vento com e sem nortada para cada ano
anos = min(ano):max(ano);
bar(anos, num_dias_nortada_por_ano);
xlabel('Ano');
ylabel('Número de dias com Nortada');
title('Número de dias com Nortada por Ano');

total_dias_com_nortada = sum(num_dias_nortada_por_ano); % Número total de dias com nortada em todos os anos
total_anos = length(num_dias_nortada_por_ano); % Número total de anos
media_nortada_por_ano = total_dias_com_nortada / total_anos; 
fprintf('A média do número de dias com nortada por ano é: %.2f\n', media_nortada_por_ano);

% Calcular a média da velocidade do vento para dias com nortada e sem nortada apenas nos anos em que há dados disponíveis
velocidade_media_com_nortada = zeros(1, total_anos);
velocidade_media_sem_nortada = zeros(1, total_anos);

for i = 1:total_anos
    % Encontrar os índices dos dias com nortada e sem nortada para o ano atual
    indices_ano_atual = find(ano == anos(i));
    dias_com_nortada = indices_nortada_por_dia(indices_ano_atual);
    dias_sem_nortada = ~indices_nortada_por_dia(indices_ano_atual);
    
    % Calcular a média da velocidade do vento apenas se houver dados disponíveis para o ano atual
    if ~isempty(indices_ano_atual)
        velocidade_media_com_nortada(i) = mean(int_vento(dias_com_nortada));
        velocidade_media_sem_nortada(i) = mean(int_vento(dias_sem_nortada));
    end
end

% Plotar o gráfico de barras
figure;
bar(anos(1:length(velocidade_media_com_nortada)), [velocidade_media_com_nortada', velocidade_media_sem_nortada']);
xlabel('Ano');
ylabel('Velocidade Média do Vento (m/s)');
title('Velocidade Média do Vento com e sem Nortada por Ano');
legend('Com Nortada', 'Sem Nortada');



%% Para cada dia, para cada ano, determinar a hora inicial e final

clear all; 
close all; 
clc;

file_name_1='ERA5_UV_gust_1994_2022.nc'; % abrange toda a costa atlântica portuguesa

% Leitura das Variáveis
lon = ncread(file_name_1,'longitude');  
lat = ncread(file_name_1,'latitude');
dados_tempo = ncread(file_name_1,'time');   % horas desde 01-01-1900 00:00:00
u10 = ncread(file_name_1,'u10');
v10 = ncread(file_name_1,'v10');

% Dados horários de 1994-2022
tempo = datetime(1900,1,1) + hours(dados_tempo);  % converter para datetime
ano = year(tempo); 
dia_ano = day(tempo, 'dayofyear'); % dia do ano (1 a 365/366)
hora_dia = hour(tempo); % hora do dia (0 a 23)

% Vento de direção entre 25 e 335 graus
direcao_vento = atan2(v10, u10) * 180 / pi + 180;
nortada = (direcao_vento >= 335 | direcao_vento <= 25);

% Verificar se houve nortada em algum momento do intervalo de tempo de cada dia para cada ano
num_anos = max(ano) - min(ano) + 1;
presenca_nortada_por_dia_por_ano = zeros(365, num_anos);
for i = 1:num_anos
    for j = 1:365
        indices_dia = find(ano == min(ano) + i - 1 & dia_ano == j);
        if ~isempty(indices_dia)
            presenca_nortada_por_dia_por_ano(j, i) = any(nortada(indices_dia, :, :), 'all');
        end
    end
end

% Plotagem (opcional)
imagesc(presenca_nortada_por_dia_por_ano);
xlabel('Ano');
ylabel('Dia do Ano');
title('Presença de Nortada para cada Dia por Ano');
colorbar;
