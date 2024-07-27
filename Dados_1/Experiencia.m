clear all; close all; clc;

file_name_1='ERA5_UV_gust_1994_2022.nc'  % abrange toda a costa atlântica portuguesa
%ncdisp(file_name_1)

%Leitura das Variáveis
lon=ncread(file_name_1,'longitude');  
lat=ncread(file_name_1,'latitude');
dados_tempo=ncread(file_name_1,'time');   %horas desde 01-01-1900 00:00:00
vento_este=ncread(file_name_1,'u10');
vento_norte=ncread(file_name_1,'v10');


%Limites coordenadas
% geolimits([38 42],[-9.5 -7.5])
% geobasemap streets


%Dados horários de 1994-2022
tempo=datevec(hours(dados_tempo)+datetime(1900,1,1));  % ano|mês|dia|hora|min|seg
ano=tempo(:,1); mes=tempo(:,2); dia=tempo(:,3); hora=tempo(:,4);


% Calcular direção do vento
direcao_vento = atan2(vento_norte, vento_este) * (180/pi); % Converter para graus

% Converter direção negativa para positiva
direcao_vento(direcao_vento < 0) = direcao_vento(direcao_vento < 0) + 360;

% Calcular velocidade do vento
velocidade_vento = sqrt(vento_norte.^2 + vento_este.^2);



% Definir critérios de Nortada
nortada_direcao_min = 315;
nortada_direcao_max = 45;
nortada_velocidade_min = 7;

% Identificar dias com Nortada
nortada_indices = zeros(size(direcao_vento));
for i = 1:numel(direcao_vento)
  if direcao_vento(i) >= nortada_direcao_min && direcao_vento(i) <= nortada_direcao_max && velocidade_vento(i) > nortada_velocidade_min
    nortada_indices(i) = true;
  end
end

% Contar dias com Nortada por ano
unico_ano = unique(ano);
nortada_por_ano = zeros(length(unico_ano), 1);
for i = 1:length(unico_ano)
  ano_atual = unico_ano(i);
  nortada_por_ano(i) = sum(nortada_indices(ano == ano_atual));
end

% Criar o gráfico
bar(unico_ano, nortada_por_ano);
xlabel('Ano');
ylabel('Número de Dias com Nortada');
title('Variação Anual de Dias com Nortada (1994-2022)');





% direcao_vento=atan2d(u,v);
% velocidade=sqrt((u).^2+(v).^2);
% 
% m_lat=repmat(lat,1,length(lon))'; 
% m_lon=repmat(lon,1,length(lat));
% 
% nortada = (direcao_vento >= 25) & (direcao_vento <= 335);
% 
% % Calculando a frequência da "nortada" por ano
% ano_inicio = 1900;
% ano_fim = ano_inicio + floor(max(dados_tempo) / (365*24)); % Supondo que o tempo está em horas desde 1900
% anos = unique(year(datetime(ano_inicio, 1, 1) + hours(dados_tempo)));
% frequencia_nortada_por_ano = zeros(length(anos), 1);






%% NOTAS

%Vamos considerar que o Norte está entre 25°-335°


%% CÓDIGO QUE PODE SER ÚTIL

% Conversão de valores de tempo para datas
data_inicio = datetime(1900, 1, 1, 0, 0, 0); % Data inicial do conjunto de dados
datas = data_inicio + hours(dados_tempo); % Conversão de horas para datas
disp(datas(1));