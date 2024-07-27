#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind

# %%
tabela_dados_ERA5 = pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/tabela_dados_ERA5.csv')
tabela_dados_MPI = pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/tabela_dados_MPI.csv')
estacao_ERA5 =  pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/estacao_ERA5.csv')
estacao_MPI =  pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/estacao_MPI.csv')

#tabela_dados_ERA5.groupby(['Ano'])['Nortada Médias Diárias'].sum()
#grouped_MPI.loc[13:26, ['Ano', 'Nortada Médias Diárias']]

#%%
dados_com_nortada_ERA5 = tabela_dados_ERA5[tabela_dados_ERA5['Nortada Médias Diárias'] == 1]
vel_media_nortada_ERA5 = dados_com_nortada_ERA5['Velocidade média diária']
tabela_dados_ERA5['Velocidade Média Nortada'] = vel_media_nortada_ERA5

dados_com_nortada_MPI = tabela_dados_MPI[tabela_dados_MPI['Nortada Médias Diárias'] == 1]
vel_media_nortada_MPI = dados_com_nortada_MPI['Velocidade média diária']
tabela_dados_MPI['Velocidade Média Nortada'] = vel_media_nortada_MPI

#%%

grouped_ERA5 = tabela_dados_ERA5.groupby(['Ano', 'Mês']).agg({
    'Nortada Médias Diárias': 'sum', 'Nortada Média (12h)': 'sum', 'Nortada Média (18h)': 'sum', # Aggregate sum of 'Nortada Médias Diárias'
    # Include all other columns without aggregation
    **{col: 'mean' for col in tabela_dados_ERA5.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)','Nortada Média (18h)']}
}).reset_index()

grouped_ano_ERA5 = tabela_dados_ERA5.groupby(['Ano']).agg({
    'Nortada Médias Diárias': 'sum', 'Nortada Média (12h)': 'sum', 'Nortada Média (18h)': 'sum', 'Velocidade Média Nortada' : 'mean',# Aggregate sum of 'Nortada Médias Diárias'
    # Include all other columns without aggregation
    **{col: 'mean' for col in tabela_dados_ERA5.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)','Nortada Média (18h)', 'Velocidade Média Nortada']}
}).reset_index()

#%%
grouped_MPI = tabela_dados_MPI.groupby(['Ano', 'Mês']).agg({
    'Nortada Médias Diárias': 'sum', 'Nortada Média (12h)': 'sum', 'Nortada Média (18h)': 'sum', # Aggregate sum of 'Nortada Médias Diárias'
    # Include all other columns without aggregation
    **{col: 'mean' for col in tabela_dados_ERA5.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)']}
}).reset_index()

grouped_ano_MPI = tabela_dados_MPI.groupby(['Ano']).agg({
    'Nortada Médias Diárias': 'sum', 'Nortada Média (12h)': 'sum', 'Nortada Média (18h)': 'sum', 'Velocidade Média Nortada' : 'mean', # Aggregate sum of 'Nortada Médias Diárias'
    # Include all other columns without aggregation
    **{col: 'mean' for col in tabela_dados_MPI.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias','Nortada Média (12h)','Nortada Média (18h)',  'Velocidade Média Nortada' ]}
}).reset_index()


#%%
grouped_estacao_ERA5 = tabela_dados_ERA5.groupby(['Ano', 'Estações'], sort=False).agg({
    'Nortada Médias Diárias': 'sum', 
    'Nortada Média (12h)': 'sum', 
    'Nortada Média (18h)': 'sum', 
    'Direções médias diárias': 'mean', 
    **{col: 'mean' for col in tabela_dados_ERA5.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)', 'Direções médias diárias']}
}).reset_index()


grouped_estacao_MPI = tabela_dados_MPI.groupby(['Ano', 'Estações'],sort=False).agg({
    'Nortada Médias Diárias': 'sum', 
    'Nortada Média (12h)': 'sum', 
    'Nortada Média (18h)': 'sum', 
    # Incluir todas as outras colunas sem agregação
    **{col: 'mean' for col in tabela_dados_MPI.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)']}
}).reset_index()

#TODO: create agregated table for year 


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                   GRÁFICOS DE LINHA C/ REGRESSÃO LINEAR
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#               NÚMERO DE DIAS COM NORTADA POR ANO (BARRA)

# wrfout ERA5
coefficients_ERA5 = np.polyfit(grouped_ano_ERA5["Ano"], grouped_ano_ERA5["Nortada Médias Diárias"], 1)
polynomial_ERA5 = np.poly1d(coefficients_ERA5)
trend_line_ERA5 = polynomial_ERA5(grouped_ano_ERA5["Ano"])
a_ERA5 = coefficients_ERA5[0]
b_ERA5 = coefficients_ERA5[1]
r_squared_ERA5 = r2_score(grouped_ano_ERA5["Nortada Médias Diárias"], trend_line_ERA5)

# wrfout MPI
coefficients_MPI = np.polyfit(grouped_ano_MPI["Ano"], grouped_ano_MPI["Nortada Médias Diárias"], 1)
polynomial_MPI = np.poly1d(coefficients_MPI)
trend_line_MPI = polynomial_MPI(grouped_ano_MPI["Ano"])
a_MPI, b_MPI = coefficients_MPI
r_squared_MPI = r2_score(grouped_ano_MPI["Nortada Médias Diárias"], trend_line_MPI)



#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                  NÚMERO DE DIAS COM NORTADA POR ANO (BARRA)
                        

t_statistic, p_value = ttest_ind(grouped_ano_ERA5["Nortada Médias Diárias"], grouped_ano_MPI["Nortada Médias Diárias"])

print("Estatística t:", t_statistic)
print("Valor p:", p_value)

if p_value < 0.05:
    print("A diferença entre os conjuntos de dados é estatisticamente significativa.")
else:
    print("Não há diferença estatisticamente significativa entre os conjuntos de dados.")


plt.figure(figsize=(20, 9))


# Plotando o gráfico
plt.figure(figsize=(20, 9))

# Plotando os dados
points_ERA5 = sns.lineplot(data=grouped_ano_ERA5, x="Ano", y="Nortada Médias Diárias", marker="s", markeredgecolor='black', markersize=10, color="#191970", label="wrfout ERA5")
points_MPI = sns.lineplot(data=grouped_ano_MPI, x="Ano", y="Nortada Médias Diárias", marker="s", markeredgecolor='black', markersize=10, color="#03A9F4", label="wrfout MPI")

# Plotando as linhas de tendência
line_ERA5, = plt.plot(grouped_ano_ERA5["Ano"], trend_line_ERA5, color='#191970', linestyle='--', linewidth=2, label=f'Tendência ERA5: y={a_ERA5:.2f}x{("+" if b_ERA5 > 0 else "")}{b_ERA5:.2f} \n$R^2$={r_squared_ERA5:.2f}')
line_MPI, = plt.plot(grouped_ano_MPI["Ano"], trend_line_MPI, color='#03A9F4', linestyle='--', linewidth=2, label=f'Tendência MPI: y={a_MPI:.2f}x{("+" if b_MPI > 0 else "")}{b_MPI:.2f} \n$R^2$={r_squared_MPI:.2f}')

# Adicionando estatística t e valor p na legenda
legend_text = f'Significância estatística: \n t-statistic = {t_statistic:.2f}, p-value = {p_value:.4f}'

# Construindo os handles manualmente
from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color="#191970", marker='s', linestyle='-', markersize=10, markeredgecolor='black', label="wrfout ERA5"),
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=10, markeredgecolor='black', label="wrfout MPI"),
    Line2D([0], [0], color="#191970", linestyle='--', linewidth=2, label=f'Tendência ERA5: y={a_ERA5:.2f}x{("+" if b_ERA5 > 0 else "")}{b_ERA5:.2f} \n$R^2$={r_squared_ERA5:.2f}'),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2, label=f'Tendência MPI: y={a_MPI:.2f}x{("+" if b_MPI > 0 else "")}{b_MPI:.2f} \n$R^2$={r_squared_MPI:.2f}')
]

labels = [
    "wrfout ERA5",
    "wrfout MPI",
    f'Regressão linear wrfout ERA5: \n y={a_ERA5:.2f}x{("+" if b_ERA5 > 0 else "")}{b_ERA5:.2f}, $R^2$={r_squared_ERA5:.2f}',
    f'Regressão linear wrfout MPI: \n y={a_MPI:.2f}x{("+" if b_MPI > 0 else "")}{b_MPI:.2f}, $R^2$={r_squared_MPI:.2f}',
    legend_text
]


plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=labels + [legend_text],
           loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=25)


plt.title('Número de dias com Nortada por ano', fontsize=30)
plt.xticks(np.arange(1995, 2015, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com Nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()

plt.savefig("plots/lineplot/num_dias_nortada_por_ano_lineplot_wrfout.jpeg", dpi=600, quality=95, bbox_inches='tight', format='png')


#%% VELOCIDADE MÉDIA COM NORTADA POR ANO (1995-2014)


#wrfout ERA5 e wrfout MPI
x_vel_ERA5 = grouped_ano_ERA5["Ano"]
y_vel_ERA5 = grouped_ano_ERA5["Velocidade Média Nortada"]

x_vel_MPI = grouped_ano_MPI["Ano"]
y_vel_MPI = grouped_ano_MPI["Velocidade Média Nortada"]

t_statistic_vel, p_value_vel = ttest_ind(y_vel_ERA5, y_vel_MPI)

print("Estatística de teste t:", t_statistic)
print("Valor p:", p_value)

coefficients_vel_ERA5 = np.polyfit(x_vel_ERA5, y_vel_ERA5, 1)
polynomial_vel_ERA5 = np.poly1d(coefficients_vel_ERA5)
trend_line_vel_ERA5 = polynomial_vel_ERA5(x_vel_ERA5)

coefficients_vel_MPI = np.polyfit(x_vel_MPI, y_vel_MPI, 1)
polynomial_vel_MPI = np.poly1d(coefficients_vel_MPI)
trend_line_vel_MPI = polynomial_vel_MPI(x_vel_MPI)

a_vel_ERA5, b_vel_ERA5 = coefficients_vel_ERA5
a_vel_MPI, b_vel_MPI = coefficients_vel_MPI

r_squared_vel_ERA5  = r2_score(y_vel_ERA5 , trend_line_vel_ERA5 )
r_squared_vel_MPI = r2_score(y_vel_MPI, trend_line_vel_MPI)

plt.figure(figsize=(18, 8))
sns.set_theme(font_scale=1.7)

sns.lineplot(data=grouped_ano_ERA5, x="Ano", y="Velocidade Média Nortada", marker="s", markeredgecolor='black', markersize=10, color="#191970", label="wrfout ERA5")
sns.lineplot(data=grouped_ano_MPI, x="Ano", y="Velocidade Média Nortada", marker="s", markeredgecolor='black', markersize=10, color="#03A9F4", label="wrfout MPI")

plt.plot(x_vel_ERA5, trend_line_vel_ERA5, color="#191970", linestyle='--', linewidth=2 , label= f'Regressão linear ERA5 \n y={a_vel_ERA5:.4f}x+{b_vel_ERA5:.4f}, $R^2$={r_squared_vel_ERA5:.4f}')
plt.plot(x_vel_MPI, trend_line_vel_MPI, color="#03A9F4", linestyle='--', linewidth=2 , label=f'Regressão linear MPI \n y={a_vel_MPI:.4f}x{b_vel_MPI:.4f}, $R^2$={r_squared_vel_MPI:.4f}')

plt.title('Velocidade média com Nortada por ano',fontsize=30)
plt.xlabel('Ano', fontsize=25)
plt.ylabel('Velocidade média com Nortada', fontsize=25)
plt.yticks(fontsize=22)
plt.xticks(range(int(min(x_vel_ERA5)), int(max(x_vel_ERA5)) + 1), rotation=45)
legend_text = f"t-statistic = {t_statistic_vel:.2f}, p-value = {p_value_vel:.4f}"
plt.legend(title=legend_text, loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)
plt.tight_layout()
plt.savefig("plots/lineplot/vel_media_ERA5_MPI_com_tendencia.jpeg", dpi=600, quality=95, bbox_inches='tight', format='png')





# %% VELOCIDADE MÉDIA COM NORTADA POR ESTAÇÃO E ANO 

# SUBPLOTS - wrfout ERA5
estacoes = ['Inverno', 'Primavera', 'Verão', 'Outono']
coefficients_ERA5 = {}
polynomials_ERA5 = {}
trend_lines_ERA5 = {}
a_values_ERA5 = {}
b_values_ERA5 = {}
r_squared_values_ERA5 = {}

for estacao in estacoes:
    dados_estacao = grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao]
    coefficients_ERA5[estacao] = np.polyfit(dados_estacao["Ano"], dados_estacao["Velocidade Média Nortada"], 1)
    polynomials_ERA5[estacao] = np.poly1d(coefficients_ERA5[estacao])
    trend_lines_ERA5[estacao] = polynomials_ERA5[estacao](dados_estacao["Ano"])
    a_values_ERA5[estacao], b_values_ERA5[estacao] = coefficients_ERA5[estacao]
    r_squared_values_ERA5[estacao] = r2_score(dados_estacao["Velocidade Média Nortada"], trend_lines_ERA5[estacao])

fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
axs = axs.flatten()

for i, estacao in enumerate(estacoes):
    sns.lineplot(data=grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao], x="Ano", y="Velocidade Média Nortada",
                 marker="o", markeredgecolor='black', markersize=10, color="#03A9F4", ax=axs[i])
    axs[i].plot(grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao]["Ano"], trend_lines_ERA5[estacao], 
                color="#FF5733", linestyle='--', label=f'y={a_values_ERA5[estacao]:.3f}x+{b_values_ERA5[estacao]:.3f}, $R^2$={r_squared_values_ERA5[estacao]:.3f}')
    axs[i].set_title(estacao)
    axs[i].set_ylabel('Velocidade média com Nortada')


    if i == 0:  #1º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper left')
    elif i == 1: #2º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper right')
    elif i == 2:  #3º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper left')
    elif i == 3:  #4º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='lower left')

plt.suptitle('Velocidade média com Nortada por ano - wrfout ERA5')
plt.tight_layout()
plt.xticks(np.arange(min(grouped_estacao_ERA5['Ano']), max(grouped_estacao_ERA5['Ano'])+1, 1))

for ax in axs:
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

plt.savefig("plots/lineplot/vel_media_ERA5_com_tendencia_estacoes.jpeg")


#%%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Dados das estações
estacoes = ['Inverno', 'Primavera', 'Verão', 'Outono']

# Função para calcular os coeficientes e linhas de tendência
def calcular_tendencia(dados, estacao):
    dados_estacao = dados[dados['Estações'] == estacao]
    coeficientes = np.polyfit(dados_estacao["Ano"], dados_estacao["Velocidade Média Nortada"], 1)
    polinomio = np.poly1d(coeficientes)
    linha_tendencia = polinomio(dados_estacao["Ano"])
    a, b = coeficientes
    r2 = r2_score(dados_estacao["Velocidade Média Nortada"], linha_tendencia)
    return coeficientes, polinomio, linha_tendencia, a, b, r2

# Inicializando dicionários para armazenar os resultados
coef_ERA5, poly_ERA5, trend_ERA5, a_ERA5, b_ERA5, r2_ERA5 = {}, {}, {}, {}, {}, {}
coef_MPI, poly_MPI, trend_MPI, a_MPI, b_MPI, r2_MPI = {}, {}, {}, {}, {}, {}

# Calculando para cada estação
for estacao in estacoes:
    coef_ERA5[estacao], poly_ERA5[estacao], trend_ERA5[estacao], a_ERA5[estacao], b_ERA5[estacao], r2_ERA5[estacao] = calcular_tendencia(grouped_estacao_ERA5, estacao)
    coef_MPI[estacao], poly_MPI[estacao], trend_MPI[estacao], a_MPI[estacao], b_MPI[estacao], r2_MPI[estacao] = calcular_tendencia(grouped_estacao_MPI, estacao)

# Criando subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
axs = axs.flatten()

for i, estacao in enumerate(estacoes):
    # Plotando os dados ERA5
    sns.lineplot(data=grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao], x="Ano", y="Velocidade Média Nortada",
                 marker="o", markeredgecolor='black', markersize=10, color="#191970", ax=axs[i], label='ERA5')
    axs[i].plot(grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao]["Ano"], trend_ERA5[estacao], 
                color="#FF5733", linestyle='--', label=f'ERA5: y={a_ERA5[estacao]:.3f}x+{b_ERA5[estacao]:.3f}, $R^2$={r2_ERA5[estacao]:.3f}')
 #191970
#03A9F4   
    # Plotando os dados MPI
    sns.lineplot(data=grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao], x="Ano", y="Velocidade Média Nortada",
                 marker="o", markeredgecolor='black', markersize=10, color="#4CAF50", ax=axs[i], label='MPI')
    axs[i].plot(grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao]["Ano"], trend_MPI[estacao], 
                color="#FFA500", linestyle='--', label=f'MPI: y={a_MPI[estacao]:.3f}x+{b_MPI[estacao]:.3f}, $R^2$={r2_MPI[estacao]:.3f}')
    
    axs[i].set_title(estacao)
    axs[i].set_ylabel('Velocidade média com Nortada')

    # Configurando legendas
    axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='best')

plt.suptitle('Velocidade média com Nortada por estação e ano - ERA5 vs MPI')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.xticks(np.arange(min(grouped_estacao_ERA5['Ano']), max(grouped_estacao_ERA5['Ano'])+1, 1))

for ax in axs:
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))


plt.show()



#%% SUBPLOTS - wrfout MPI

coefficients_MPI = {}
polynomials_MPI = {}
trend_lines_MPI = {}
a_values_MPI = {}
b_values_MPI = {}
r_squared_values_MPI = {}

for estacao in estacoes:
    dados_estacao = grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao]
    coefficients_MPI[estacao] = np.polyfit(dados_estacao["Ano"], dados_estacao["Velocidade Média Nortada"], 1)
    polynomials_MPI[estacao] = np.poly1d(coefficients_MPI[estacao])
    trend_lines_MPI[estacao] = polynomials_MPI[estacao](dados_estacao["Ano"])
    a_values_MPI[estacao], b_values_MPI[estacao] = coefficients_MPI[estacao]
    r_squared_values_MPI[estacao] = r2_score(dados_estacao["Velocidade Média Nortada"], trend_lines_MPI[estacao])

fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
axs = axs.flatten()

for i, estacao in enumerate(estacoes):
    sns.lineplot(data=grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao], x="Ano", y="Velocidade Média Nortada",
                 marker="o", markeredgecolor='black', markersize=10, color="#03A9F4", ax=axs[i])
    axs[i].plot(grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao]["Ano"], trend_lines_MPI[estacao], 
                color="#FF5733", linestyle='--', label=f'y={a_values_MPI[estacao]:.3f}x+{b_values_MPI[estacao]:.3f}, $R^2$={r_squared_values_MPI[estacao]:.3f}')
    axs[i].set_title(estacao)
    axs[i].set_ylabel('Velocidade média com Nortada')
    axs[i].legend(frameon=True, facecolor='white', edgecolor='black')

    if i == 0:  #1º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='best')
    elif i == 1: #2º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper right')
    elif i == 2:  #3º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper right')
    elif i == 3:  #4º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='best')

plt.suptitle('Velocidade média com Nortada por estação e ano - wrfout MPI')
plt.tight_layout()
plt.xticks(np.arange(min(grouped_estacao_MPI['Ano']), max(grouped_estacao_MPI['Ano'])+1, 1))

for ax in axs:
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

plt.savefig("plots/lineplot/vel_media_MPI_com_tendencia_estacoes.jpeg")


#%% Velocidade média com Nortada por ano (4 estações juntas num gráfico)

estacoes = ['Inverno', 'Primavera', 'Verão', 'Outono']
cores = ['blue', 'green', 'red', 'orange']

fig, axs = plt.subplots(1, 2, figsize=(20, 6))


# wrfout ERA5
ax = axs[0]
for estacao, cor in zip(estacoes, cores):
    dados_estacao_ERA5 = grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao]
    sns.lineplot(data=dados_estacao_ERA5, x="Ano", y="Velocidade Média Nortada",
                 markeredgecolor='black', markersize=10, color=cor, ax=ax)

ax.set_title('wrfout ERA5')
ax.set_ylabel('Velocidade média com Nortada')
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.set_xticks(np.arange(min(grouped_estacao_ERA5['Ano']), max(grouped_estacao_ERA5['Ano'])+1, 1))

# wrfout MPI
ax = axs[1]
for estacao, cor in zip(estacoes, cores):
    dados_estacao_MPI = grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao]
    sns.lineplot(data=dados_estacao_MPI, x="Ano", y="Velocidade Média Nortada",
                 markeredgecolor='black', markersize=10, color=cor, ax=ax, label=estacao)

ax.set_title('wrfout MPI')
ax.set_ylabel('Velocidade média com Nortada')
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.set_xticks(np.arange(min(grouped_estacao_MPI['Ano']), max(grouped_estacao_MPI['Ano'])+1, 1))


min_y = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
max_y = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])

for ax in axs:
    ax.set_ylim(min_y, max_y)
plt.legend(bbox_to_anchor=(1.0, 1), frameon=True, facecolor='white', edgecolor='black', loc='upper left')
fig.suptitle('Velocidade média com Nortada por ano')
plt.tight_layout()

plt.savefig("plots/lineplot/vel_media_estacoes.jpeg")



#%% Direções médias com Nortada por ano

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# wrfout ERA5
ax = axs[0]
for estacao, cor in zip(estacoes, cores):
    dados_estacao_ERA5 = grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao]
    sns.lineplot(data=dados_estacao_ERA5, x="Ano", y="Direções médias diárias",
                 markeredgecolor='black', markersize=10, color=cor, ax=ax)

ax.set_title('wrfout ERA5')
ax.set_ylabel('Direções médias diárias')
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.set_xticks(np.arange(min(grouped_estacao_ERA5['Ano']), max(grouped_estacao_ERA5['Ano'])+1, 1))

# wrfout MPI
ax = axs[1]
for estacao, cor in zip(estacoes, cores):
    dados_estacao_MPI = grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao]
    sns.lineplot(data=dados_estacao_MPI, x="Ano", y="Direções médias diárias",
                 markeredgecolor='black', markersize=10, color=cor, ax=ax, label=estacao)

ax.set_title('wrfout MPI')
ax.set_ylabel('Direções médias diárias')
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.set_xticks(np.arange(min(grouped_estacao_MPI['Ano']), max(grouped_estacao_MPI['Ano'])+1, 1))


min_y = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
max_y = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])

for ax in axs:
    ax.set_ylim(min_y, max_y)
plt.legend(bbox_to_anchor=(1.0, 1), frameon=True, facecolor='white', edgecolor='black', loc='upper left')
fig.suptitle('Direções médias com Nortada por ano')
plt.tight_layout()

plt.savefig("plots/lineplot/direcoes_estacoes.jpeg")




# %% NÚMERO DE DIAS COM NORTADA POR ESTAÇÃO POR ANO 

# Subplots de linha - wrfout ERA5
cores_estacoes = {'Inverno': 'blue', 'Primavera': 'green', 'Verão': 'red', 'Outono': 'orange'}

coefficients_numdiasnortada_ERA5 = {}
polynomials_numdiasnortada_ERA5 = {}
trend_lines_numdiasnortada_ERA5 = {}
a_values_numdiasnortada_ERA5 = {}
b_values_numdiasnortada_ERA5 = {}
r_squared_values_numdiasnortada_ERA5 = {}

for estacao in estacoes:
    dados_estacao = grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao]
    coefficients_ERA5[estacao] = np.polyfit(dados_estacao["Ano"], dados_estacao["Nortada Médias Diárias"], 1)
    polynomials_ERA5[estacao] = np.poly1d(coefficients_ERA5[estacao])
    trend_lines_ERA5[estacao] = polynomials_ERA5[estacao](dados_estacao["Ano"])
    a_values_ERA5[estacao], b_values_ERA5[estacao] = coefficients_ERA5[estacao]
    r_squared_values_ERA5[estacao] = r2_score(dados_estacao["Nortada Médias Diárias"], trend_lines_ERA5[estacao])

fig, axs = plt.subplots(2, 2, figsize=(18, 14), sharex=True)
axs = axs.flatten()

for i, estacao in enumerate(estacoes):
    sns.lineplot(data=grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao], x="Ano", y="Nortada Médias Diárias",
                  markeredgecolor='black',  linewidth=2, markersize=10, color=cores_estacoes[estacao], ax=axs[i])
    axs[i].plot(grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == estacao]["Ano"], trend_lines_ERA5[estacao], 
                color="black", linestyle='--', linewidth=3 , label=f'y={a_values_ERA5[estacao]:.3f}x{("+" if b_values_ERA5[estacao] > 0 else "")}{b_values_ERA5[estacao]:.3f} \n $R^2$={r_squared_values_ERA5[estacao]:.3f}')
    axs[i].set_title(estacao, fontsize=30)
    axs[i].set_ylabel('Número de dias com Nortada', fontsize=25)
    axs[i].set_xlabel('Ano', fontsize=25)
    axs[i].legend(frameon=True, facecolor='white', edgecolor='black', fontsize=25)

    if i == 0:  #1º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='best', fontsize=25)
    elif i == 1: #2º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper left', fontsize=25)
    elif i == 2:  #3º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='best', fontsize=25)
    elif i == 3:  #4º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='lower left', fontsize=25)


plt.suptitle('Número de dias com Nortada por ano - wrfout ERA5', fontsize=30)
plt.xticks(np.arange(min(grouped_estacao_ERA5['Ano']), max(grouped_estacao_ERA5['Ano'])+1, 5))


for ax in axs:
    ax.tick_params(axis='x', rotation=45, labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

plt.tight_layout()
plt.savefig("plots/lineplot/num_dias_nortada_anual_estacoes_ERA5.jpeg")



#%% Subplots de linha - wrfout MPI
coefficients_numdiasnortada_MPI = {}
polynomials_numdiasnortada_MPI = {}
trend_lines_numdiasnortada_MPI = {}
a_values_numdiasnortada_MPI = {}
b_values_numdiasnortada_MPI = {}
r_squared_values_numdiasnortada_MPI = {}

for estacao in estacoes:
    dados_estacao = grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao]
    coefficients_MPI[estacao] = np.polyfit(dados_estacao["Ano"], dados_estacao["Nortada Médias Diárias"], 1)
    polynomials_MPI[estacao] = np.poly1d(coefficients_MPI[estacao])
    trend_lines_MPI[estacao] = polynomials_MPI[estacao](dados_estacao["Ano"])
    a_values_MPI[estacao], b_values_MPI[estacao] = coefficients_MPI[estacao]
    r_squared_values_MPI[estacao] = r2_score(dados_estacao["Nortada Médias Diárias"], trend_lines_MPI[estacao])

fig, axs = plt.subplots(2, 2, figsize=(18, 14), sharex=True)
axs = axs.flatten()

for i, estacao in enumerate(estacoes):
    sns.lineplot(data=grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao], x="Ano", y="Nortada Médias Diárias",
                  markeredgecolor='black', linewidth=2, markersize=10, color=cores_estacoes[estacao], ax=axs[i])
    axs[i].plot(grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == estacao]["Ano"], trend_lines_MPI[estacao], 
                color="black", linestyle='--', linewidth=3,  label=f'y={a_values_MPI[estacao]:.3f}x{("+" if b_values_MPI[estacao] > 0 else "")}{b_values_MPI[estacao]:.3f} \n $R^2$={r_squared_values_MPI[estacao]:.3f}')
    axs[i].set_title(estacao, fontsize=30)
    axs[i].set_ylabel('Número de dias com Nortada', fontsize=25)
    axs[i].set_xlabel('Ano', fontsize=25)
    axs[i].legend(frameon=True, facecolor='white', edgecolor='black')


    if i == 0:  #1º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='best', fontsize=25)
    elif i == 1: #2º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper right', fontsize=25)
    elif i == 2:  #3º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper left', bbox_to_anchor=(0.10, 0.95), fontsize=25)
    elif i == 3:  #4º subplot
        axs[i].legend(frameon=True, facecolor='white', edgecolor='black', loc='upper left',   fontsize=25)

plt.suptitle('Número de dias com Nortada por ano - wrfout MPI', fontsize=30)
plt.xticks(np.arange(min(grouped_estacao_MPI['Ano']), max(grouped_estacao_MPI['Ano'])+1, 5))

for ax in axs:
    ax.tick_params(axis='x', rotation=45, labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

plt.tight_layout()
plt.savefig("plots/lineplot/num_dias_nortada_anual_estacoes_MPI.jpeg")

# %%

