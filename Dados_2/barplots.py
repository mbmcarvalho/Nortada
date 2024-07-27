#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

#%%

tabela_dados_ERA5 = pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/tabela_dados_ERA5.csv')
tabela_dados_MPI = pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/tabela_dados_MPI.csv')
estacao_ERA5 =  pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/estacao_ERA5.csv')
estacao_MPI =  pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/estacao_MPI.csv')

#tabela_dados_ERA5.groupby(['Ano'])['Nortada Médias Diárias'].sum()
#grouped_MPI.loc[13:26, ['Ano', 'Nortada Médias Diárias']]

#####################################################################
#VELOCIDADE DADOS DOS DIAS COM NORTADA
#%%
dados_com_nortada_ERA5 = tabela_dados_ERA5[tabela_dados_ERA5['Nortada Médias Diárias'] == 1]
vel_media_nortada_ERA5 = dados_com_nortada_ERA5['Velocidade média diária']
tabela_dados_ERA5['Velocidade Média Nortada'] = vel_media_nortada_ERA5

#%%
dados_com_nortada_MPI = tabela_dados_MPI[tabela_dados_MPI['Nortada Médias Diárias'] == 1]
vel_media_nortada_MPI = dados_com_nortada_MPI['Velocidade média diária']
tabela_dados_MPI['Velocidade Média Nortada'] = vel_media_nortada_MPI



############################################################
# %%


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


#############################################################3
#Nº DIAS C/ NORTADA POR ANO (BARRA)


# %% wrfout ERA5
plt.figure(figsize=(10,8))
sns.set_theme(font_scale = 1.7)
sns.barplot(data=grouped_ERA5,x="Ano",
            y="Nortada Médias Diárias", errorbar='sd',  color="#03A9F4")
plt.title('Número de dias com Nortada por mês por cada ano (Barra) - wrfout ERA5')
plt.xticks(rotation=45)
plt.ylabel('Número de dias com Nortada')
plt.xlabel('Ano')
plt.tight_layout()
plt.savefig("plots/barplot/num_dias_nortada_diaria_por_mes_ERA5.jpeg")




# %%  wrfout MPI
plt.figure(figsize=(10,8))
sns.set_theme(font_scale = 1.7)
sns.barplot(data=grouped_MPI,x="Ano",
            y="Nortada Médias Diárias", errorbar='sd', color="#03A9F4")
plt.title('Número de dias com Nortada por mês por cada ano  (Barra) - wrfout MPI')
plt.xticks(rotation=45)
plt.ylabel('Número de dias com Nortada')
plt.xlabel('Ano')
plt.tight_layout()
plt.savefig("plots/barplot/num_dias_nortada_diaria_por_mes_MPI.jpeg")

 
#TODO: create agregated table for year 

##############################################################
#NÚMERO DE DIAS COM NORTADA DIÁRIA POR ANO (Barra)

# %% BARPLOT - wrfout ERA5
plt.figure(figsize=(10, 8))
sns.set_theme(font_scale=1.7)
sns.barplot(data=grouped_ano_ERA5, x="Ano", y="Nortada Médias Diárias",  color="#03A9F4")
plt.title('Número de Dias com Nortada por Ano (Barra) - wrfout ERA5')
plt.xticks(rotation=45)
plt.tight_layout() 
plt.savefig("plots/barplot/num_dias_nortada_diaria_por_ano_barplot_ERA5.jpeg")


# %% BARPLOT - wrfout MPI
plt.figure(figsize=(10, 8))
sns.set_theme(font_scale=1.7)
sns.barplot(data=grouped_ano_MPI, x="Ano", y="Nortada Médias Diárias",  color="#03A9F4")
plt.title('Número de Dias com Nortada por Ano (Barra) - wrfout ERA5')
plt.xticks(rotation=45)
plt.tight_layout() 
plt.savefig("plots/barplot/num_dias_nortada_diaria_por_ano_barplot_MPI.jpeg")



##################################################################
# NÚMERO DE DIAS COM NORTADA ÀS 12H E ÀS 18H (Barra)

#%% SUBPLOTS DE BARPLOT - wrfout ERA5
plt.figure(figsize=(10,8))

plt.subplot(2, 1, 1) # Subplot 1 (12h)
sns.barplot(data=grouped_ano_ERA5,x="Ano",
            y="Nortada Média (12h)", color="#03A9F4")
plt.title('Número de Dias com Nortada(12h) por Ano (Barra) - wrfout ERA5')
plt.ylabel('Nº dias com Nortada(12h)')
plt.xlabel('Ano')
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(2, 1, 2) # Subplot 2 (18h)
sns.barplot(data=grouped_ano_ERA5,x="Ano",
            y="Nortada Média (18h)", color="#03A9F4")
plt.title('Número de Dias com Nortada(18h) por Ano (Barra) - wrfout ERA5')
plt.ylabel('Nº dias com Nortada(18h)')
plt.xlabel('Ano')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("plots/barplot/num_dias_nortada_12h_18h_por_ano_ERA5.jpeg")


#%% SUBPLOTS DE BARPLOT - wrfout MPI
plt.figure(figsize=(10,8))

plt.subplot(2, 1, 1) # Subplot 1 (12h)
sns.barplot(data=grouped_ano_MPI,x="Ano",
            y="Nortada Média (12h)", color="#03A9F4")
plt.title('Número de Dias com Nortada(12h) por Ano (Barra) - wrfout MPI')
plt.ylabel('Nº dias com Nortada(12h)')
plt.xlabel('Ano')
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(2, 1, 2) # Subplot 2 (18h)
sns.barplot(data=grouped_ano_MPI,x="Ano",
            y="Nortada Média (18h)", color="#03A9F4")
plt.title('Número de Dias com Nortada(18h) por Ano (Barra) - wrfout MPI')
plt.ylabel('Nº dias com Nortada(18h)')
plt.xlabel('Ano')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("plots/barplot/num_dias_nortada_12h_18h_por_ano_MPI.jpeg")


#####################################################
#%%
grouped_estacao_ERA5 = tabela_dados_ERA5.groupby(['Ano', 'Estações'],sort=False).agg({
    'Nortada Médias Diárias': 'sum', 
    'Nortada Média (12h)': 'sum', 
    'Nortada Média (18h)': 'sum', 
    # Incluir todas as outras colunas sem agregação
    **{col: 'mean' for col in tabela_dados_ERA5.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)']}
}).reset_index()

grouped_estacao_MPI = tabela_dados_MPI.groupby(['Ano', 'Estações'],sort=False).agg({
    'Nortada Médias Diárias': 'sum', 
    'Nortada Média (12h)': 'sum', 
    'Nortada Média (18h)': 'sum', 
    # Incluir todas as outras colunas sem agregação
    **{col: 'mean' for col in tabela_dados_MPI.columns if col not in ['Ano', 'Mês', 'Dia', 'Estações', 'Nortada Médias Diárias', 'Nortada Média (12h)', 'Nortada Média (18h)']}
}).reset_index()



#####################################################
#NÚMERO DE DIAS COM NORTADA POR ESTAÇÃO (1995-2014)
#%% wrfout ERA5

cores_estacoes = {'Inverno': "#1E88E5", 'Primavera': "#9CCC65", 'Verão': "#FF5722", 'Outono': "#FFEB3B"}

sns.set_theme()
plt.figure(figsize=(12, 8))
ax=sns.barplot(data=grouped_estacao_ERA5, x='Ano', y='Nortada Médias Diárias', hue='Estações', ci=None, palette=cores_estacoes)
plt.title('Número de dias com Nortada por estação por cada ano ( Barra) - wrfout ERA5', fontsize=20)
plt.xlabel('Ano', fontsize=15)
plt.ylabel('Número de dias com Nortada média diária', fontsize=15)
plt.xticks( fontsize=15, rotation=45)
plt.yticks( fontsize=15)
sns.move_legend(ax, bbox_to_anchor=(1.2,0.5), loc='center right', borderaxespad=0., frameon=False, title='Estações', title_fontsize='large', fontsize='large')
plt.tight_layout()
plt.savefig("plots/barplot/num_dias_nortada_diaria_estacao_anual_ERA5.jpeg")


#%% wrfout MPI

cores_estacoes = {'Inverno': "#1E88E5", 'Primavera': "#9CCC65", 'Verão': "#FF5722", 'Outono': "#FFEB3B"}

sns.set_theme()
plt.figure(figsize=(12, 8))
ax=sns.barplot(data=grouped_estacao_MPI, x='Ano', y='Nortada Médias Diárias', hue='Estações', ci=None, palette=cores_estacoes)
plt.title('Número de dias com Nortada por estação por cada ano ( Barra) - wrfout MPI', fontsize=20)
plt.xlabel('Ano', fontsize=15)
plt.ylabel('Número de dias com Nortada média diária', fontsize=15)
plt.xticks( fontsize=15, rotation=45)
plt.yticks( fontsize=15)
sns.move_legend(ax, bbox_to_anchor=(1.2,0.5), loc='center right', borderaxespad=0., frameon=False, title='Estações',title_fontsize='large', fontsize='large')
plt.tight_layout()
plt.savefig("plots/barplot/num_dias_nortada_diaria_estacao_anual_MPI.jpeg")


##########################################################################
#PARA MÉDIA DO NÚMERO DE DIAS COM NORTADA POR CADA ESTAÇÃO (MÉDIA DE 1995 A 2014)

#%% wrfout ERA5
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
for ax, season in zip(axes.flatten(), grouped_estacao_ERA5['Estações']):
    data_season = estacao_ERA5[estacao_ERA5['Estação'] == season]
    sns.barplot(data=data_season, x='Ano', y='Nº dias c/ nortada média', ax=ax, color=cores_estacoes[season])
    ax.set_title(f'{season}',fontsize=20)
    ax.set_xlabel('Ano', fontsize=15)
    ax.set_ylabel('Número de dias com nortada', fontsize=15)
    ax.tick_params(axis='x', labelsize=15, rotation=45) 
    ax.tick_params(axis='y', labelsize=15) 
fig.suptitle('Número de dias com Nortada por estação por cada ano - wrfout ERA5', fontsize=20)
plt.savefig("plots/barplot/num_dias_nortada_diaria_estacao_anual_ERA5_subplots.jpeg")

#%% wrfout MPI
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
for ax, season in zip(axes.flatten(), grouped_estacao_MPI['Estações']):
    data_season = estacao_MPI[estacao_ERA5['Estação'] == season]
    sns.barplot(data=data_season, x='Ano', y='Nº dias c/ nortada média', ax=ax, color=cores_estacoes[season])
    ax.set_title(f'{season}', fontsize=20)
    ax.set_xlabel('Ano', fontsize=15)
    ax.set_ylabel('Número de dias com nortada', fontsize=15)
    ax.tick_params(axis='x', labelsize=15, rotation=45) 
    ax.tick_params(axis='y', labelsize=15) 
fig.suptitle('Número de dias com Nortada por estação por cada ano - wrfout MPI', fontsize=20)
plt.savefig("plots/barplot/num_dias_nortada_diaria_estacao_anual_MPI_subplots.jpeg")



##########################################################################
#MÉDIA DO NÚMERO DE DIAS COM NORTADA POR ESTAÇÃO (1995-2014)

#%% wrfout ERA5
estacao_df = pd.DataFrame(grouped_estacao_ERA5)
media_por_estacao = estacao_df.groupby('Estações')['Nortada Médias Diárias'].mean()

sns.set_theme()
plt.figure(figsize=(12, 8))

indices_ordenados = sorted(media_por_estacao.index, key=lambda estacao: list(cores_estacoes.keys()).index(estacao))
cores = [cores_estacoes[estacao] for estacao in indices_ordenados]

plt.bar(indices_ordenados, media_por_estacao[indices_ordenados], color=cores)
plt.title('Média do número de dias com Nortada por estação (1995-2014) - wrfout ERA5', fontsize=20)
plt.xlabel('Estação', fontsize=15)
plt.ylabel('Número de dias com nortada', fontsize=15)
plt.xticks( fontsize=15)
plt.yticks( fontsize=15)
plt.savefig("plots/barplot/media_num_dias_nortada_estacao_ERA5.jpeg")



#%% wrfout MPI
estacao_df = pd.DataFrame(grouped_estacao_MPI)
media_por_estacao = estacao_df.groupby('Estações')['Nortada Médias Diárias'].mean()

sns.set_theme()
plt.figure(figsize=(12, 8))

indices_ordenados = sorted(media_por_estacao.index, key=lambda estacao: list(cores_estacoes.keys()).index(estacao))
cores = [cores_estacoes[estacao] for estacao in indices_ordenados]

plt.bar(indices_ordenados, media_por_estacao[indices_ordenados], color=cores)
plt.title('Média do número de dias com Nortada por estação (1995-2014) - wrfout MPI', fontsize=20)
plt.xlabel('Estação', fontsize=15)
plt.ylabel('Número de dias com nortada', fontsize=15)
plt.xticks( fontsize=15)
plt.yticks( fontsize=15)

plt.savefig("plots/barplot/media_num_dias_nortada_estacao_MPI.jpeg")



##################################################################
# INTERVALO DE 5 EM 5 ANOS

#%%
anual_ERA5 = pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/anual_ERA5.csv')
anual_MPI =  pd.read_csv('C:/Users/Beatriz/Desktop/Projeto/Dados_2/tabelas_1995_2014_em_csv/anual_MPI.csv')


#Para wrfout ERA5
lower, higher = anual_ERA5['Ano'].min(), anual_ERA5['Ano'].max()+2
#n_bins = 6
edges = range(lower, higher+2, 5)  # Ajuste para garantir que o último bin seja incluído
lbs = ['[%d, %d['%(edges[i], edges[i+1]) for i in range(len(edges)-1)]
anual_ERA5['Anos'] = pd.cut(anual_ERA5.Ano, bins=edges, labels=lbs, include_lowest=True)

#Para wrfout MPI
low, high = anual_MPI['Ano'].min(), anual_MPI['Ano'].max()+2
#n_bins = 6
edge = range(low, high+2, 5)  # Ajuste para garantir que o último bin seja incluído
lbs = ['[%d, %d['%(edge[i], edge[i+1]) for i in range(len(edge)-1)]
anual_MPI['Anos'] = pd.cut(anual_MPI.Ano, bins=edge, labels=lbs, include_lowest=True)


# %% Nº DIAS C/ NORTADA POR ANO (BARRA) - wrfout ERA5 (intervalo de 5 em 5 anos)
plt.figure(figsize=(10,6))
sns.set_theme(font_scale = 1.7)
sns.barplot(data=anual_ERA5,x="Anos",
            y="Número de Dias com Nortada",
             ci=None, color="#03A9F4")
plt.title('Número de Dias com Nortada a cada 5 anos (Barra) - wrfout ERA5')
plt.tight_layout()
plt.savefig("plots/barplot/num_dias_nortada_diaria_inter_5anos_ERA5.jpeg")

#errorbar = sd

# %% Nº DIAS C/ NORTADA POR ANO (BARRA) - wrfout MPI (intervalo de 5 em 5 anos)
plt.figure(figsize=(10,6))
sns.set_theme(font_scale = 1.7)
sns.barplot(data=anual_MPI,x="Anos",
            y="Número de Dias com Nortada",
             ci=None, color="#03A9F4")
plt.title('Número de Dias com Nortada a cada 5 anos (Barra) - wrfout MPI')
plt.tight_layout()
plt.savefig("plots/barplot/num_dias_nortada_diaria_inter_5anos_MPI.jpeg")



#%% Nº DIAS C/ NORTADA P/ ANO (BARRA) 12h e 18h  - wrfout ERA5 (intervalo de 5 em 5 anos)

plt.figure(figsize=(10,8))

plt.subplot(2, 1, 1) # Subplot 1 (12h)
sns.barplot(data=anual_ERA5,x="Anos",
            y="NDias c/ Nortada(12h)",ci=None, color="#03A9F4")
plt.title('Número de Dias com Nortada(12h) por Ano (Barra) - wrfout ERA5')
plt.ylabel('Nº dias com Nortada(12h)')
plt.xlabel('Ano')
plt.tight_layout()

plt.subplot(2, 1, 2) # Subplot 2 (18h)
sns.barplot(data=anual_MPI,x="Anos",
            y="NDias c/ Nortada(18h)", ci=None, color="#03A9F4")
plt.title('Número de Dias com Nortada(18h) por Ano (Barra) - wrfout ERA5')
plt.ylabel('Nº dias com Nortada(18h)')
plt.xlabel('Ano')
plt.tight_layout()

plt.savefig("plots/barplot/num_dias_nortada_inter5anos_ERA5_12h_18h.jpeg")



#%% Nº DIAS C/ NORTADA P/ ANO (BARRA) 12h e 18h  - wrfout MPI (intervalo de 5 em 5 anos)

plt.figure(figsize=(10,8))

plt.subplot(2, 1, 1) # Subplot 1 (12h)
sns.barplot(data=anual_MPI,x="Anos",
            y="NDias c/ Nortada(12h)", ci=None,  color="#03A9F4")
plt.title('Número de Dias com Nortada(12h) por Ano (Barra) - wrfout MPI')
plt.ylabel('Nº dias com Nortada(12h)')
plt.xlabel('Ano')
plt.tight_layout()

plt.subplot(2, 1, 2) # Subplot 2 (18h)
sns.barplot(data=anual_MPI,x="Anos",
            y="NDias c/ Nortada(18h)", ci=None, color="#03A9F4")
plt.title('Número de Dias com Nortada(18h) por Ano (Barra) - wrfout MPI')
plt.ylabel('Nº dias com Nortada(18h)')
plt.xlabel('Ano')
plt.tight_layout()

plt.savefig("plots/barplot/num_dias_nortada_inter5anos_MPI_12h_18h.jpeg")

# %%
