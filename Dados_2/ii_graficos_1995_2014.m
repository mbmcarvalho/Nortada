%% GRÁFICOS - wrfout ERA5 Aveiro

clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\armazenamento_dados_1995_2014');

load('nortada_vel_wrfout_ERA5.mat');
load('vento_Barra_wrfout_ERA5.mat');
load('vento_wrfout_ERA5.mat');
load('estacao_ERA5.mat'); 
load('anual_ERA5.mat'); 


%----------------------------------------
% NÚMERO DE DIAS COM NORTADA POR CADA ANO

figure(1);
bar(anual_ERA5.Ano, anual_ERA5.('Número de Dias com Nortada'));
xlabel('Ano');
ylabel('Número de Dias com Nortada');
title('Número de Dias com Nortada por Ano (Barra) - wrfout ERA5');
ylim([0, 82])

media_dias_por_ano_ERA5 = mean(anual_ERA5.('Número de Dias com Nortada'));
disp(['A média de dias com nortada por ano é: ', num2str(media_dias_por_ano_ERA5)]);

figure(2);

subplot(2, 1, 1);
bar(anual_ERA5.Ano, anual_ERA5.("NDias c/ Nortada(12h)"));
title('Número de Dias com Nortada(12h) por Ano (Barra) - wrfout ERA5');
ylabel('Nº dias com Nortada(12h)');
xlabel('Ano')

subplot(2, 1, 2);
bar(anual_ERA5.Ano, anual_ERA5.("NDias c/ Nortada(18h)"));
title('Número de Dias com Nortada(18h) por Ano (Barra) - wrfout ERA5');
ylabel('Nº dias com Nortada(12h)');
xlabel('Ano')


media_dias_12h_por_ano_ERA5 = mean(anual_ERA5.("NDias c/ Nortada(12h)"));
media_dias_18h_por_ano_ERA5 = mean(anual_ERA5.("NDias c/ Nortada(18h)"));
disp(['A média de dias com nortada(12h) por ano é: ', num2str(media_dias_12h_por_ano_ERA5), ...
    ', e a média de dias com nortada(18h) por ano é: ', num2str(media_dias_18h_por_ano_ERA5)]);

%----------------------------------------------
%VELOCIDADE MÉDIA DO NÚMERO DE DIAS COM NORTADA

anos = anual_ERA5.Ano;
velocidade_media_nortada = anual_ERA5.('Vel média c/ nortada');

figure(3);
plot(anos, velocidade_media_nortada, 'o-');
xlabel('Ano');
ylabel('Velocidade Média dos Dias com Nortada');
title('Velocidade Média dos Dias com Nortada - wrfout ERA5');
grid on;
legend('Velocidade Média Nortada');
xlim([1994, 2015])


%------------------------------------------------------
% VELOCIDADE MÉDIA DOS DIAS C/ NORTADA P/ ANO P/ESTAÇÃO


anos = unique(estacao_ERA5.Ano);
[~, idx] = unique(estacao_ERA5.("Estação"));
estacoes = estacao_ERA5.("Estação")(sort(idx));
media_por_estacao_por_ano = zeros(length(estacoes), length(anos));

for i = 1:length(anos)
    for j = 1:length(estacoes)
        media = estacao_ERA5.("Vel média c/ nortada")(estacao_ERA5.Ano == anos(i) & strcmp(estacao_ERA5.("Estação"), estacoes{j}));
        if ~isempty(media)
            media_por_estacao_por_ano(j, i) = media;
        end
    end
end

figure(4);
bar(anos, media_por_estacao_por_ano');
xlabel('Ano');
ylabel('Média da Velocidade Média Diária');
title('Média da Velocidade Média Diária p/ ano - wrfout ERA5');
legend(estacoes, 'Location', 'best');
grid on;
xticks(anos);
xtickangle(45);



%-----------------------------------------------------
% Nº DE DIAS C/ NORTADA PARA CADA ESTAÇÃO (5 EM 5 ANOS)

intervalo_anos = 1995:5:2014;
[~, idx] = unique(estacao_ERA5.("Estação"));
estacoes_ano_unicas = estacao_ERA5.("Estação")(sort(idx));
num_dias_nortada_por_estacao_ano_intervalo = zeros(length(intervalo_anos), length(estacoes_ano_unicas));


for j = 1:length(intervalo_anos)
    ano_atual = intervalo_anos(j);
    indices_ano_atual = find(tabela_dados_ERA5.Ano == ano_atual);
    dados_ano_atual = tabela_dados_ERA5(indices_ano_atual, :);
    num_dias_nortada_por_estacao_ano = zeros(length(estacoes_ano_unicas), 1);
    
    
    for i = 1:length(estacoes_ano_unicas)
        estacao_ano_atual = estacoes_ano_unicas{i};
        dados_estacao_ano_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacao_ano_atual), :);
        nortada_logico = logical(dados_estacao_ano_atual.("Nortada Médias Diárias"));
        num_dias_nortada = sum(nortada_logico);
        num_dias_nortada_por_estacao_ano(i) = num_dias_nortada;
    end
    
    num_dias_nortada_por_estacao_ano_intervalo(j, :) = num_dias_nortada_por_estacao_ano';
end



figure(5);
bar(intervalo_anos, num_dias_nortada_por_estacao_ano_intervalo);
legend(estacoes_ano_unicas, 'Location', 'BestOutside');
title(['Nº de Dias c/ Nortada por Estação a cada 5 anos - wrfout ERA5']);
xlabel('Ano');
ylabel('Número de Dias com Nortada');
ylim([0, 27])



%-------------------------------------------------------------
% Nº DE DIAS C/ NORTADA PARA CADA ESTAÇÃO (PARA TODOS OS ANOS)
anos_unicos = unique(tabela_dados_ERA5.Ano);
[~, idx] = unique(estacao_ERA5.("Estação"));
estacoes_ano_unicas = estacao_ERA5.("Estação")(sort(idx));
num_dias_nortada_por_estacao_ano = zeros(length(anos_unicos), length(estacoes_ano_unicas));

for j = 1:length(anos_unicos)
    ano_atual = anos_unicos(j);
    indices_ano_atual = find(tabela_dados_ERA5.Ano == ano_atual);
    dados_ano_atual = tabela_dados_ERA5(indices_ano_atual, :);
    num_dias_nortada_por_estacao_ano_atual = zeros(length(estacoes_ano_unicas), 1);
    
    for i = 1:length(estacoes_ano_unicas)
        estacao_atual = estacoes_ano_unicas{i};
        dados_estacao_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacao_atual), :);
        num_dias_nortada = sum(logical(dados_estacao_atual.("Nortada Médias Diárias")));
        num_dias_nortada_por_estacao_ano_atual(i) = num_dias_nortada;
    end

    num_dias_nortada_por_estacao_ano(j, :) = num_dias_nortada_por_estacao_ano_atual';
end

figure(6);
bar(anos_unicos, num_dias_nortada_por_estacao_ano);
legend(estacoes_ano_unicas, 'Location', 'BestOutside');
title(['Nº de Dias c/ Nortada por Estação - wrfout ERA5']);
xlabel('Ano');
ylabel('Número de Dias com Nortada');
ylim([0, 31]);
xlim([1994, 2015])


media_num_dias_nortada_por_estacao = mean(num_dias_nortada_por_estacao_ano);
disp('Média do número de dias com nortada por cada estação - wrfout ERA5:');
disp(media_num_dias_nortada_por_estacao);



%----------------------------------------------------------------------
% Nº DE DIAS C/ NORTADA PARA CADA ESTAÇÃO (PARA TODOS OS ANOS) subplots
anos_unicos = unique(tabela_dados_ERA5.Ano);
[~, idx] = unique(estacao_ERA5.("Estação"));
estacoes_ano_unicas = estacao_ERA5.("Estação")(sort(idx));
num_subplots = length(estacoes_ano_unicas);

num_rows = ceil(num_subplots / 2); % 2 colunas
num_columns = min(2, num_subplots); % 2 colunas 

cores = colormap(lines(num_subplots));

figure(7);
for subplot_index = 1:num_subplots
    subplot(num_rows, num_columns, subplot_index);

    estacao_atual = estacoes_ano_unicas{subplot_index};
    num_dias_nortada_estacao_atual = zeros(length(anos_unicos), 1);
    
    for j = 1:length(anos_unicos)
        ano_atual = anos_unicos(j);
        dados_ano_atual = tabela_dados_ERA5(tabela_dados_ERA5.Ano == ano_atual, :);
        dados_estacao_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacao_atual), :);
        num_dias_nortada = sum(logical(dados_estacao_atual.("Nortada Médias Diárias")));
        num_dias_nortada_estacao_atual(j) = num_dias_nortada;
    end
    
    cor_atual = cores(subplot_index, :);
    bar(anos_unicos, num_dias_nortada_estacao_atual, 'FaceColor', cor_atual);
    title(['Nº de Dias c/ Nortada - ', estacao_atual]);
    xlabel('Ano');
    ylabel('Número de Dias com Nortada');
    ylim([0, 31]);
    xlim([min(anos_unicos)-1, max(anos_unicos)+1]);
end
sgtitle('Nº de Dias c/ Nortada por Estação - wrfout ERA5');






%% GRÁFICOS - wrfout MPI hist Aveiro

clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\armazenamento_dados_1995_2014');

load('nortada_vel_wrfout_MPI_hist.mat');
load('vento_Barra_wrfout_MPI_hist.mat');
load('vento_wrfout_MPI_hist.mat');
load('estacao_MPI.mat');  
load('anual_MPI.mat'); 


%----------------------------------------
% NÚMERO DE DIAS COM NORTADA POR CADA ANO

figure(1);
bar(anual_MPI.Ano, anual_MPI.('Número de Dias com Nortada'));
xlabel('Ano');
ylabel('Número de Dias com Nortada');
title('Número de Dias com Nortada por Ano (Barra) - wrfout MPI');
ylim([0, 87])

media_dias_por_ano_MPI = mean(anual_MPI.('Número de Dias com Nortada'));
disp(['A média de dias com nortada por ano é: ', num2str(media_dias_por_ano_MPI)]);

figure(2);

subplot(2, 1, 1);
bar(anual_MPI.Ano, anual_MPI.("NDias c/ Nortada(12h)"));
title('Nº de Dias c/ Nortada(12h) por Ano (Barra) - wrfout MPI');
ylabel('Nº dias com Nortada(12h)');
xlabel('Ano')

subplot(2, 1, 2);
bar(anual_MPI.Ano, anual_MPI.("NDias c/ Nortada(18h)"));
title('Nº de Dias c/ Nortada(18h) por Ano (Barra) - wrfout MPI');
ylabel('Nº dias com Nortada(12h)');
xlabel('Ano')

media_dias_12h_por_ano_MPI = mean(anual_MPI.("NDias c/ Nortada(12h)"));
media_dias_18h_por_ano_MPI = mean(anual_MPI.("NDias c/ Nortada(18h)"));
disp(['A média de dias com nortada(12h) por ano é: ', num2str(media_dias_12h_por_ano_MPI), ...
    ', e a média de dias com nortada(18h) por ano é: ', num2str(media_dias_18h_por_ano_MPI)]);

%----------------------------------------------
%VELOCIDADE MÉDIA DO NÚMERO DE DIAS COM NORTADA

anos = anual_MPI.Ano;
velocidade_media_nortada = anual_MPI.('Vel média c/ nortada');

figure(3);
plot(anos, velocidade_media_nortada, 'o-');
xlabel('Ano');
ylabel('Velocidade Média dos Dias com Nortada');
title('Velocidade Média dos Dias com Nortada - wrfout MPI ');
grid on;
legend('Velocidade Média Nortada');
xlim([1994, 2015])


%------------------------------------------------------
% VELOCIDADE MÉDIA DOS DIAS C/ NORTADA P/ ANO P/ESTAÇÃO


anos = unique(estacao_MPI.Ano);
[~, idx] = unique(estacao_MPI.("Estação"));
estacoes = estacao_MPI.("Estação")(sort(idx));
media_por_estacao_por_ano = zeros(length(estacoes), length(anos));

for i = 1:length(anos)
    for j = 1:length(estacoes)
        media = estacao_MPI.("Vel média c/ nortada")(estacao_MPI.Ano == anos(i) & strcmp(estacao_MPI.("Estação"), estacoes{j}));
        if ~isempty(media)
            media_por_estacao_por_ano(j, i) = media;
        end
    end
end

figure(4);
bar(anos, media_por_estacao_por_ano');
xlabel('Ano');
ylabel('Média da Velocidade Média Diária');
title('Média da Velocidade Média Diária c/ Nortada - wrfout MPI');
legend(estacoes, 'Location', 'best');
grid on;
xticks(anos);
xtickangle(45);



%-----------------------------------------------------
% Nº DE DIAS C/ NORTADA PARA CADA ESTAÇÃO (5 EM 5 ANOS)

intervalo_anos = 1995:5:2014;
[~, idx] = unique(estacao_MPI.("Estação"));
estacoes_ano_unicas = estacao_MPI.("Estação")(sort(idx));
num_dias_nortada_por_estacao_ano_intervalo = zeros(length(intervalo_anos), length(estacoes_ano_unicas));


for j = 1:length(intervalo_anos)
    ano_atual = intervalo_anos(j);
    indices_ano_atual = find(tabela_dados_MPI.Ano == ano_atual);
    dados_ano_atual = tabela_dados_MPI(indices_ano_atual, :);
    num_dias_nortada_por_estacao_ano = zeros(length(estacoes_ano_unicas), 1);
    
    
    for i = 1:length(estacoes_ano_unicas)
        estacao_ano_atual = estacoes_ano_unicas{i};
        dados_estacao_ano_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacao_ano_atual), :);
        nortada_logico = logical(dados_estacao_ano_atual.("Nortada Médias Diárias"));
        num_dias_nortada = sum(nortada_logico);
        num_dias_nortada_por_estacao_ano(i) = num_dias_nortada;
    end
    
    num_dias_nortada_por_estacao_ano_intervalo(j, :) = num_dias_nortada_por_estacao_ano';
end



figure(5);
bar(intervalo_anos, num_dias_nortada_por_estacao_ano_intervalo);
legend(estacoes_ano_unicas, 'Location', 'BestOutside');
title(['Nº de Dias c/ Nortada por Estação a cada 5 anos - wrfout MPI']);
xlabel('Ano');
ylabel('Número de Dias com Nortada');
ylim([0, 27])



%-------------------------------------------------------------
% Nº DE DIAS C/ NORTADA PARA CADA ESTAÇÃO (PARA TODOS OS ANOS)
anos_unicos = unique(tabela_dados_MPI.Ano);
[~, idx] = unique(estacao_MPI.("Estação"));
estacoes_ano_unicas = estacao_MPI.("Estação")(sort(idx));
num_dias_nortada_por_estacao_ano = zeros(length(anos_unicos), length(estacoes_ano_unicas));

for j = 1:length(anos_unicos)
    ano_atual = anos_unicos(j);
    indices_ano_atual = find(tabela_dados_MPI.Ano == ano_atual);
    dados_ano_atual = tabela_dados_MPI(indices_ano_atual, :);
    num_dias_nortada_por_estacao_ano_atual = zeros(length(estacoes_ano_unicas), 1);
    
    for i = 1:length(estacoes_ano_unicas)
        estacao_atual = estacoes_ano_unicas{i};
        dados_estacao_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacao_atual), :);
        num_dias_nortada = sum(logical(dados_estacao_atual.("Nortada Médias Diárias")));
        num_dias_nortada_por_estacao_ano_atual(i) = num_dias_nortada;
    end

    num_dias_nortada_por_estacao_ano(j, :) = num_dias_nortada_por_estacao_ano_atual';
end

figure(6);
bar(anos_unicos, num_dias_nortada_por_estacao_ano);
legend(estacoes_ano_unicas, 'Location', 'BestOutside');
title(['Nº de Dias c/ Nortada por Estação - wrfout MPI']);
xlabel('Ano');
ylabel('Número de Dias com Nortada');
ylim([0, 31]);
xlim([1994, 2015])

media_num_dias_nortada_por_estacao = mean(num_dias_nortada_por_estacao_ano);
disp('Média do número de dias com nortada por cada estação - wrfout MPI:');
disp(media_num_dias_nortada_por_estacao);



%----------------------------------------------------------------------
% Nº DE DIAS C/ NORTADA PARA CADA ESTAÇÃO (PARA TODOS OS ANOS) subplots
anos_unicos = unique(tabela_dados_MPI.Ano);
[~, idx] = unique(estacao_MPI.("Estação"));
estacoes_ano_unicas = estacao_MPI.("Estação")(sort(idx));
num_subplots = length(estacoes_ano_unicas);

num_rows = ceil(num_subplots / 2); % 2 colunas
num_columns = min(2, num_subplots); % 2 colunas 

cores = colormap(lines(num_subplots));

figure(7);
for subplot_index = 1:num_subplots
    subplot(num_rows, num_columns, subplot_index);

    estacao_atual = estacoes_ano_unicas{subplot_index};
    num_dias_nortada_estacao_atual = zeros(length(anos_unicos), 1);
    
    for j = 1:length(anos_unicos)
        ano_atual = anos_unicos(j);
        dados_ano_atual = tabela_dados_MPI(tabela_dados_MPI.Ano == ano_atual, :);
        dados_estacao_atual = dados_ano_atual(strcmp(dados_ano_atual.("Estações"), estacao_atual), :);
        num_dias_nortada = sum(logical(dados_estacao_atual.("Nortada Médias Diárias")));
        num_dias_nortada_estacao_atual(j) = num_dias_nortada;
    end
    
    cor_atual = cores(subplot_index, :);
    bar(anos_unicos, num_dias_nortada_estacao_atual, 'FaceColor', cor_atual);
    title(['Nº de Dias c/ Nortada - ', estacao_atual]);
    xlabel('Ano');
    ylabel('Número de Dias com Nortada');
    ylim([0, 31]);
    xlim([min(anos_unicos)-1, max(anos_unicos)+1]);
end
sgtitle('Nº de Dias c/ Nortada por Estação - wrfout MPI');







%% gráficos com wrfout ERA5 e MPI 

clear all; close all; clc;

addpath('C:\Users\Beatriz\Desktop\Projeto\Dados_2\armazenamento_dados_1995_2014');

load('nortada_vel_wrfout_ERA5.mat');
load('vento_Barra_wrfout_ERA5.mat');
load('vento_wrfout_ERA5.mat');
load('estacao_ERA5.mat'); 
load('anual_ERA5.mat'); 

load('nortada_vel_wrfout_MPI_hist.mat');
load('vento_Barra_wrfout_MPI_hist.mat');
load('vento_wrfout_MPI_hist.mat');
load('estacao_MPI.mat');  
load('anual_MPI.mat'); 


%----------------------------------------------
%VELOCIDADE MÉDIA DO NÚMERO DE DIAS COM NORTADA

anos = anual_MPI.Ano;
vel_media_nortada_ERA5 = anual_ERA5.('Vel média c/ nortada');
vel_media_nortada_MPI = anual_MPI.('Vel média c/ nortada');

tendencia_ERA5 = polyval(polyfit(anual_ERA5.Ano, anual_ERA5.("Vel média c/ nortada"), 1), anual_ERA5.Ano);
tendencia_MPI= polyval(polyfit(anual_MPI.Ano, anual_MPI.("Vel média c/ nortada"), 1), anual_MPI.Ano);

cor_ERA5 = [0, 0.4470, 0.7410]; % Azul
cor_MPI = [0.8500, 0.3250, 0.0980]; % Avermelhado

figure(1);
plot(anos, vel_media_nortada_ERA5, 'o-', 'Color', cor_ERA5, 'LineWidth', 1);
hold on
plot(anos, tendencia_ERA5,'--', 'Color', cor_ERA5, 'LineWidth', 1);
plot(anos, vel_media_nortada_MPI, 's-', 'Color', cor_MPI, 'LineWidth', 1);
plot(anos, tendencia_MPI,'--', 'Color', cor_MPI, 'LineWidth', 1);

legend('Velocidade Média Nortada - ERA5', 'Tendência ERA5', 'Velocidade Média Nortada - MPI', 'Tendência MPI', 'Location', 'best');

xlabel('Ano');
ylabel('Velocidade Média dos Dias com Nortada');
title('Velocidade Média dos Dias com Nortada por Ano');
grid on;
xlim([1994, 2015]);




%----------------------------------------
% NÚMERO DE DIAS COM NORTADA POR CADA ANO - linha de tendência linear

figure;
bar(anual_MPI.Ano, anual_MPI.('Número de Dias com Nortada'));
p = polyfit(anual_MPI.Ano, anual_MPI.('Número de Dias com Nortada'), 1);
tendencia = polyval(p, anual_MPI.Ano);
hold on;
plot(anual_MPI.Ano, tendencia, 'r-', 'LineWidth', 2); 
hold off;