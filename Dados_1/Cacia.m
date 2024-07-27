%% Gráfico de Barras - Nº dias com nortada/ano

clear all; close all; clc;

file_name_2 = readtable('ERA5_UV_gust_1994_2022_praia_barra.txt', 'Delimiter', ' '); % Delimitador é um espaço
% disp(head(file_name_2))  % só mostra os primeiros dados

% i10fg=file_name_2.Var10;   %rajada do vento - velocidade média do vento máxima naquela hora

% num_linhas = size(file_name_2, 1);  num_colunas = size(file_name_2, 2);
% numel(u10)); %tamanho de u10

lon= table2array(file_name_2(1:3:end, 1));     
lat = table2array(file_name_2(1:3:end, 2));
u10 = table2array(file_name_2(1:3:end, 12));   %m/s   
v10 = table2array(file_name_2(2:3:end, 12));   %m/s


% time= table2array(file_name_2(1:3:end, 5)); 

date = file_name_2.Var4;
ano = year(date(1:3:end));
mes = month(date(1:3:end));
dia = day(date(1:3:end));

time = file_name_2.Var5;
hora = hours(time(1:3:end));   % para minutos usava-se 'minutes' e 'seconds'

dir_vento_graus = 180 + (180/pi) * atan2(v10,u10); %convenção meteorológica
%atan2 devolve entre -pi e pi
velocidade = sqrt(u10.^2 + v10.^2);

% Determinar se houve nortada em pelo menos uma amostra horária para cada dia
indices_nortada = (dir_vento_graus >= 335 | dir_vento_graus <= 25);  % Valores booleanos (0 ou 1), indicando se há ou não nortada

% Contar o número total de dias com nortada para cada ano
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
title('Número de dias com Nortada por Ano em Cacia');

% Calcular a média do número de dias com nortada por ano
media_nortada_por_ano = mean(num_dias_nortada_por_ano);
fprintf('A média do número de dias com nortada por ano é: %.2f\n', media_nortada_por_ano);



%% Série temporal


clear all; close all; clc;

file_name_2 = readtable('ERA5_UV_gust_1994_2022_praia_barra.txt', 'Delimiter', ' ');

date = file_name_2.Var4;
velocidade = table2array(file_name_2(:, 12)); % Velocidade média do vento

% Calcular os limites superior e inferior para remover outliers (por exemplo, 95% dos dados)
limite_superior = prctile(velocidade, 97.5);
limite_inferior = prctile(velocidade, 2.5);

% Remover outliers
indices= velocidade > limite_superior | velocidade < limite_inferior;
velocidade_indices = velocidade(~indices);
date_indices = date(~indices);

% Plotar a série temporal da velocidade média
plot(date_indices, velocidade_indices);
xlabel('Data');
ylabel('Velocidade Média do Vento (m/s)');
title('Série Temporal da Velocidade Média do Vento (Cacia)');
datetick('x', 'yyyy'); % Mostra apenas o ano no eixo x


% Aplicar filtro de média móvel para suavizar os dados (por exemplo, janela de 5 pontos)
janela = 5;
velocidade_suavizada = movmean(velocidade, janela, 'omitnan');

% Verificar se há dados válidos para plotar
if isempty(date) || isempty(velocidade_suavizada)
    disp('Não há dados para plotar.');
else
    % Plotar a série temporal da velocidade média com filtro aplicado
    plot(date, velocidade_suavizada);
    xlabel('Data');
    ylabel('Velocidade Média do Vento (m/s)');
    title('Série Temporal da Velocidade Média do Vento com Filtro de Média Móvel');
    datetick('x', 'yyyy'); % Mostra apenas o ano no eixo x
end




%% Velocidade média do vento

clear all; close all; clc;

file_name_2 = readtable('ERA5_UV_gust_1994_2022_praia_barra.txt', 'Delimiter', ' '); % Delimitador é um espaço
% disp(head(file_name_2))  % só mostra os primeiros dados

lon= table2array(file_name_2(1:3:end, 1));     
lat = table2array(file_name_2(1:3:end, 2));
u10 = table2array(file_name_2(1:3:end, 12));   %m/s   
v10 = table2array(file_name_2(2:3:end, 12));   %m/s

date = file_name_2.Var4;
ano = year(date(1:3:end));
mes = month(date(1:3:end));
dia = day(date(1:3:end));

time = file_name_2.Var5;
hora = hours(time(1:3:end));   % para minutos usava-se 'minutes' e 'seconds'

dir_vento_graus = 180 + (180/pi) * atan2(v10,u10); %convenção meteorológica
%atan2 devolve entre -pi e pi
velocidade = sqrt(u10.^2 + v10.^2);

% Agora, para calcular a velocidade média para um intervalo de 5 anos:

anos_por_periodo = 5;
velocidade_media_periodo = [];
anos_periodo = [];

for i = min(ano):anos_por_periodo:max(ano)
    indices_periodo = find(ano >= i & ano < i + anos_por_periodo);
    velocidade_periodo = velocidade(indices_periodo);
    velocidade_media_periodo = [velocidade_media_periodo; mean(velocidade_periodo)];
    anos_periodo = [anos_periodo; mean(ano(indices_periodo))];
end

% Plotando o gráfico:
figure;
plot(anos_periodo, velocidade_media_periodo, 'o-', 'LineWidth', 2);
xlabel('Ano');
ylabel('Velocidade Média do Vento (m/s)');
title('Velocidade Média do Vento (Cacia)');

% Adiciona linhas verticais para indicar os limites de cada período de 5 anos
hold on;
for i = 1:length(anos_periodo)
    line([anos_periodo(i) anos_periodo(i)], [min(velocidade_media_periodo) max(velocidade_media_periodo)], 'Color', 'k', 'LineStyle', '--');
end
hold off;






