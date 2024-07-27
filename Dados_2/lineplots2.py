#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind
from matplotlib.lines import Line2D


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


t_statistic, p_value = ttest_ind(grouped_ano_ERA5["Nortada Médias Diárias"], grouped_ano_MPI["Nortada Médias Diárias"])

print("Estatística t:", t_statistic)
print("Valor p:", p_value)

if p_value < 0.05:
    print("A diferença entre os conjuntos de dados é estatisticamente significativa.")
else:
    print("Não há diferença estatisticamente significativa entre os conjuntos de dados.")


sns.set(style="darkgrid") #para definir a grelha e fundo cinza do gráfico
plt.figure(figsize=(18, 8))


points_ERA5 = sns.lineplot(data=grouped_ano_ERA5, x="Ano", y="Nortada Médias Diárias", marker="s", markeredgecolor='black', markersize=10, color="#191970", label="wrfout ERA5")
points_MPI = sns.lineplot(data=grouped_ano_MPI, x="Ano", y="Nortada Médias Diárias", marker="s", markeredgecolor='black', markersize=10, color="#03A9F4", label="wrfout MPI")

line_ERA5, = plt.plot(grouped_ano_ERA5["Ano"], trend_line_ERA5, color='#191970', linestyle='--', linewidth=2, label=f'Tendência ERA5: y={a_ERA5:.2f}x{("+" if b_ERA5 > 0 else "")}{b_ERA5:.2f} \n$R^2$={r_squared_ERA5:.2f}')
line_MPI, = plt.plot(grouped_ano_MPI["Ano"], trend_line_MPI, color='#03A9F4', linestyle='--', linewidth=2, label=f'Tendência MPI: y={a_MPI:.2f}x{("+" if b_MPI > 0 else "")}{b_MPI:.2f} \n$R^2$={r_squared_MPI:.2f}')

legend_text = f'Significância estatística entre \n os conjuntos de dados: \n t-statistic = {t_statistic:.2f}, p-value = {p_value:.4f}'

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
           loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)


#plt.title('Número de dias com Nortada por ano', fontsize=30)
plt.xticks(np.arange(1995, 2015, 1), rotation=45, fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Número de dias com Nortada', fontsize=25)
plt.xlabel('Ano', fontsize=25)
plt.tight_layout()

plt.savefig("plots/lineplot/num_dias_nortada_por_ano_lineplot_wrfout.jpeg", dpi=600, quality=95, bbox_inches='tight', format='png')



#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#               NÚMERO DE DIAS COM NORTADA POR ANO (BARRA)
#                               ESTAÇÕES

sns.set(style="darkgrid")
season_order = ["Inverno", "Primavera", "Verão", "Outono"]
fig, axs = plt.subplots(2, 2, figsize=(60, 35), sharex=True)

# Coordenadas para cada legenda
legend_positions = {
    "Inverno": (0.23, 0.88),  #(esquerda/direita, cima/baixo)
    "Primavera": (0.4, 0.88),
    "Verão": (0.23, 0.12),
    "Outono": (0.23, 0.12)
}

for i, season in enumerate(season_order):
    ax = axs[i // 2, i % 2]  # Determina a posição do subplot na matriz 2x2

    season_data_ERA5 = grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == season]
    season_data_MPI = grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == season]

    sns.lineplot(data=season_data_ERA5, x="Ano", y="Nortada Médias Diárias", color="#191970", label="wrfout ERA5", ax=ax, marker="s", markersize=18, linewidth=4)
    sns.lineplot(data=season_data_MPI, x="Ano", y="Nortada Médias Diárias", color="#03A9F4", label="wrfout MPI", ax=ax, marker="s", markersize=18, linewidth=4)

    # Regressão linear e cálculo de R^2
    coefficients_ERA5 = np.polyfit(season_data_ERA5["Ano"], season_data_ERA5["Nortada Médias Diárias"], 1)
    polynomial_ERA5 = np.poly1d(coefficients_ERA5)
    trend_line_ERA5 = polynomial_ERA5(season_data_ERA5["Ano"])
    r_squared_ERA5 = np.corrcoef(season_data_ERA5["Ano"], season_data_ERA5["Nortada Médias Diárias"])[0, 1]**2
    ax.plot(season_data_ERA5["Ano"], trend_line_ERA5, color='#191970', linestyle='--', linewidth=4, 
            label=f'Regressão linear ERA5: y={coefficients_ERA5[0]:.2f}x{coefficients_ERA5[1]:+.2f}, $R^2$={r_squared_ERA5:.2f}')

    coefficients_MPI = np.polyfit(season_data_MPI["Ano"], season_data_MPI["Nortada Médias Diárias"], 1)
    polynomial_MPI = np.poly1d(coefficients_MPI)
    trend_line_MPI = polynomial_MPI(season_data_MPI["Ano"])
    r_squared_MPI = np.corrcoef(season_data_MPI["Ano"], season_data_MPI["Nortada Médias Diárias"])[0, 1]**2
    ax.plot(season_data_MPI["Ano"], trend_line_MPI, color='#03A9F4', linestyle='--', linewidth=4, 
            label=f'Regressão linear MPI: y={coefficients_MPI[0]:.2f}x{coefficients_MPI[1]:+.2f}, $R^2$={r_squared_MPI:.2f}')

    
    # Adiciona a legenda com fundo branco e posição personalizada
    legend_pos = legend_positions[season]
    ax.legend(loc='center', fontsize=20, bbox_to_anchor=legend_pos, facecolor='white', framealpha=1)
    
    ax.set_title(f'{season}', fontsize=50)

    # eixos
    ax.set_xticks(np.arange(1995, 2015, 1))
    ax.tick_params(axis='x', rotation=45, labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlabel("Ano", fontsize=35)  
    ax.set_ylabel('Número de dias com Nortada', fontsize=35)

# Significância estatística
for i, season in enumerate(season_order):
    season_data_ERA5 = grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == season]["Nortada Médias Diárias"]
    season_data_MPI = grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == season]["Nortada Médias Diárias"]

    t_statistic, p_value = ttest_ind(season_data_ERA5, season_data_MPI, equal_var=False)

    # Adiciona o texto com os valores de t_statistic e p_value à legenda do subplot atual
    handles, labels = axs[i // 2, i % 2].get_legend_handles_labels()
    legend_text = f'Significância estatística: t-statistic: {t_statistic:.2f}, p-value: {p_value:.4f}'
    axs[i // 2, i % 2].legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
                              labels=labels + [legend_text],
                              loc='center', bbox_to_anchor=legend_positions[season], facecolor='white', framealpha=1, edgecolor='black', fontsize=28)

plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.98])
#fig.suptitle('Número de dias com Nortada por ano por estação', fontsize=40)

plt.savefig("plots/lineplot/num_dias_nortada_por_ano_seasons_wrfout.jpeg", bbox_inches='tight')



#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#                VELOCIDADE MÉDIA COM NORTADA POR ANO (BARRA)


# Dados wrfout ERA5 e wrfout MPI
x_vel_ERA5 = grouped_ano_ERA5["Ano"]
y_vel_ERA5 = grouped_ano_ERA5["Velocidade Média Nortada"]

x_vel_MPI = grouped_ano_MPI["Ano"]
y_vel_MPI = grouped_ano_MPI["Velocidade Média Nortada"]

# Estatística de teste t e valor p
t_statistic_vel, p_value_vel = ttest_ind(y_vel_ERA5, y_vel_MPI)
print("Estatística de teste t:", t_statistic_vel)
print("Valor p:", p_value_vel)

# Regressão linear e cálculo de R^2
coefficients_vel_ERA5 = np.polyfit(x_vel_ERA5, y_vel_ERA5, 1)
polynomial_vel_ERA5 = np.poly1d(coefficients_vel_ERA5)
trend_line_vel_ERA5 = polynomial_vel_ERA5(x_vel_ERA5)

coefficients_vel_MPI = np.polyfit(x_vel_MPI, y_vel_MPI, 1)
polynomial_vel_MPI = np.poly1d(coefficients_vel_MPI)
trend_line_vel_MPI = polynomial_vel_MPI(x_vel_MPI)

a_vel_ERA5, b_vel_ERA5 = coefficients_vel_ERA5
a_vel_MPI, b_vel_MPI = coefficients_vel_MPI

r_squared_vel_ERA5 = r2_score(y_vel_ERA5, trend_line_vel_ERA5)
r_squared_vel_MPI = r2_score(y_vel_MPI, trend_line_vel_MPI)

# Plot
plt.figure(figsize=(18, 8))
sns.set_theme(font_scale=1.7)

sns.lineplot(data=grouped_ano_ERA5, x="Ano", y="Velocidade Média Nortada", marker="s", markeredgecolor='black', markersize=10, color="#191970", label="wrfout ERA5")
sns.lineplot(data=grouped_ano_MPI, x="Ano", y="Velocidade Média Nortada", marker="s", markeredgecolor='black', markersize=10, color="#03A9F4", label="wrfout MPI")

plt.plot(x_vel_ERA5, trend_line_vel_ERA5, color="#191970", linestyle='--', linewidth=2, label=f'Regressão linear ERA5 \n y={a_vel_ERA5:.4f}x+{b_vel_ERA5:.4f}, $R^2$={r_squared_vel_ERA5:.4f}')
plt.plot(x_vel_MPI, trend_line_vel_MPI, color="#03A9F4", linestyle='--', linewidth=2, label=f'Regressão linear MPI \n y={a_vel_MPI:.4f}x{b_vel_MPI:.4f}, $R^2$={r_squared_vel_MPI:.4f}')

plt.xlabel('Ano', fontsize=25)
plt.ylabel('Velocidade média do vento com Nortada (m/s)', fontsize=25)
plt.yticks(fontsize=22)
plt.xticks(range(int(min(x_vel_ERA5)), int(max(x_vel_ERA5)) + 1), rotation=45)

# Texto da significância estatística
legend_text = f"Significância estatística: \n t-statistic = {t_statistic_vel:.2f}, p-value = {p_value_vel:.4f}"

# Criação dos handles para a legenda
handles = [
    Line2D([0], [0], color="#191970", marker='s', linestyle='-', markersize=10, markeredgecolor='black', label="wrfout ERA5"),
    Line2D([0], [0], color="#03A9F4", marker='s', linestyle='-', markersize=10, markeredgecolor='black', label="wrfout MPI"),
    Line2D([0], [0], color="#191970", linestyle='--', linewidth=2, label=f'Regressão linear ERA5 \n y={a_vel_ERA5:.4f}x+{b_vel_ERA5:.4f}, $R^2$={r_squared_vel_ERA5:.4f}'),
    Line2D([0], [0], color="#03A9F4", linestyle='--', linewidth=2, label=f'Regressão linear MPI \n y={a_vel_MPI:.4f}x{b_vel_MPI:.4f}, $R^2$={r_squared_vel_MPI:.4f}')
]

# Adição da legenda com o texto de significância estatística
plt.legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
           labels=[handle.get_label() for handle in handles] + [legend_text],
           loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor='black', fontsize=20)

plt.tight_layout()
plt.savefig("plots/lineplot/vel_media_nortada_por_ano_wrfout.jpeg", dpi=600, quality=95, bbox_inches='tight', format='png')


#%% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#               VELOCIDADE MÉDIA COM NORTADA POR ESTAÇÃO E ANO
#                               ESTAÇÕES

sns.set(style="darkgrid")
season_order = ["Inverno", "Primavera", "Verão", "Outono"]
fig, axs = plt.subplots(2, 2, figsize=(60, 35), sharex=True)

# Coordenadas para cada legenda
legend_positions = {
    "Inverno": (0.19, 0.1),  #(esquerda/direita, cima/baixo)
    "Primavera": (0.7, 0.9),
    "Verão": (0.19, 0.9),
    "Outono": (0.19, 0.1)
}

for i, season in enumerate(season_order):
    ax = axs[i // 2, i % 2]  # Determina a posição do subplot na matriz 2x2

    season_data_ERA5 = grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == season]
    season_data_MPI = grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == season]

    sns.lineplot(data=season_data_ERA5, x="Ano", y="Velocidade Média Nortada", color="#191970", label="wrfout ERA5", ax=ax, marker="s", markersize=18, linewidth=4)
    sns.lineplot(data=season_data_MPI, x="Ano", y="Velocidade Média Nortada", color="#03A9F4", label="wrfout MPI", ax=ax, marker="s", markersize=18, linewidth=4)

    # Regressão linear e cálculo de R^2
    coefficients_ERA5 = np.polyfit(season_data_ERA5["Ano"], season_data_ERA5["Velocidade Média Nortada"], 1)
    polynomial_ERA5 = np.poly1d(coefficients_ERA5)
    trend_line_ERA5 = polynomial_ERA5(season_data_ERA5["Ano"])
    r_squared_ERA5 = np.corrcoef(season_data_ERA5["Ano"], season_data_ERA5["Velocidade Média Nortada"])[0, 1]**2
    ax.plot(season_data_ERA5["Ano"], trend_line_ERA5, color='#191970', linestyle='--', linewidth=4, 
            label=f'Regressão linear ERA5: y={coefficients_ERA5[0]:.2f}x{coefficients_ERA5[1]:+.2f}, $R^2$={r_squared_ERA5:.2f}')

    coefficients_MPI = np.polyfit(season_data_MPI["Ano"], season_data_MPI["Velocidade Média Nortada"], 1)
    polynomial_MPI = np.poly1d(coefficients_MPI)
    trend_line_MPI = polynomial_MPI(season_data_MPI["Ano"])
    r_squared_MPI = np.corrcoef(season_data_MPI["Ano"], season_data_MPI["Velocidade Média Nortada"])[0, 1]**2
    ax.plot(season_data_MPI["Ano"], trend_line_MPI, color='#03A9F4', linestyle='--', linewidth=4, 
            label=f'Regressão linear MPI: y={coefficients_MPI[0]:-.2f}x{coefficients_MPI[1]:+.2f}, $R^2$={r_squared_MPI:.2f}')

    
    # Adiciona a legenda com fundo branco e posição personalizada
    legend_pos = legend_positions[season]
    ax.legend(loc='center', fontsize=20, bbox_to_anchor=legend_pos, facecolor='white', framealpha=1)
    
    ax.set_title(f'{season}', fontsize=50)

    # eixos
    ax.set_xticks(np.arange(1995, 2015, 1))
    ax.tick_params(axis='x', rotation=45, labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlabel("Ano", fontsize=35)  
    ax.set_ylabel('Velocidade média com Nortada (m/s)', fontsize=35)

# Significância estatística
for i, season in enumerate(season_order):
    season_data_ERA5 = grouped_estacao_ERA5[grouped_estacao_ERA5['Estações'] == season]["Velocidade Média Nortada"]
    season_data_MPI = grouped_estacao_MPI[grouped_estacao_MPI['Estações'] == season]["Velocidade Média Nortada"]

    t_statistic, p_value = ttest_ind(season_data_ERA5, season_data_MPI, equal_var=False)

    # Adiciona o texto com os valores de t_statistic e p_value à legenda do subplot atual
    handles, labels = axs[i // 2, i % 2].get_legend_handles_labels()
    legend_text = f'Significância estatística: t-statistic: {t_statistic:.2f}, p-value: {p_value:.4f}'
    axs[i // 2, i % 2].legend(handles=handles + [Line2D([0], [0], color='w', linestyle='None')], 
                              labels=labels + [legend_text],
                              loc='center', bbox_to_anchor=legend_positions[season], facecolor='white', framealpha=1, edgecolor='black', fontsize=24)

plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.98])
#fig.suptitle('Velocidade média com Nortada por estação e ano', fontsize=40)

plt.savefig("plots/lineplot/vel_media_estacoes_subplots.jpeg", bbox_inches='tight')


# %%
