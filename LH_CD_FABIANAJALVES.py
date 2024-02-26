#!/usr/bin/env python
# coding: utf-8

# ### Desafio Ciências de Dados - LightHouse
# 
# Neste notebook irei desenvolver um modelo de previsão de preços a partir do dataset oferecido pelo programa LightHouse, e avaliar tal modelo utilizando as métricas de avaliação que mais fazem sentido para o problema.

# In[2]:


# Instalando a biblioteca panda
import pandas as pd

# Importando os dados do dataset para iniciar a Análise Exploratória dos dados
Tab_Prec = pd.read_csv('teste_indicium_precificacao.csv')

# Visualizando o dataset
Tab_Prec


# In[3]:


#declarando as variáveis
coluna_alvo = 'disponibilidade_365'

#Limpando a coluna de disponibilidade de ano
Tab_Prec = Tab_Prec[Tab_Prec[coluna_alvo] !=0 ]

# Visualizando o dataset
Tab_Prec


# In[5]:


#declarando as variáveis
coluna_alvo = 'numero_de_reviews'

#Limpando a coluna de reviews
Tab_Prec = Tab_Prec[Tab_Prec[coluna_alvo] !=0 ]

# Visualizando o dataset
Tab_Prec


# Abaixo, irei avaliar se existe algum padrão no texto do nome dos imóveis de maior valor para locação. 

# #### Item 4 - Estudo de preço
# 
# Para fazer a predição do preço de um imóvel com base em nosso dataset, irei utilizar a base de dados inicial já trabalhada, criando um dataFrame com base no 'bairro' = 'Midtown', 'minimo de noite' = 1 e tipo de imóvei 'room_type' = Entire home/apt. 
# 
# 
# Dados para cálculo do preço do imóvel:
# 
# {'id': 2595,
#  'nome': 'Skylit Midtown Castle',
#  'host_id': 2845,
#  'host_name': 'Jennifer',
#  'bairro_group': 'Manhattan',
#  'bairro': 'Midtown',
#  'latitude': 40.75362,
#  'longitude': -73.98377,
#  'room_type': 'Entire home/apt',
#  'price': 225,
#  'minimo_noites': 1,
#  'numero_de_reviews': 45,
#  'ultima_review': '2019-05-21',
#  'reviews_por_mes': 0.38,
#  'calculado_host_listings_count': 2,
#  'disponibilidade_365': 355}

# In[14]:


#declarando as colunas que serão usadas
colunas_desejadas = ['numero_de_reviews' , 'price', 'bairro', 'room_type', 'minimo_noites' , 'disponibilidade_365']

#chamando as colunas que serão usadas
Tab_Prec_filtro = Tab_Prec[colunas_desejadas]

# Visualizando o dataset
Tab_Prec_filtro


# In[13]:


# declarando as variáveis
minimo_noites = 'minimo_noites'
bairro = 'bairro'
room_type = 'room_type'

# inserindo as condições
condicao_noites = (Tab_Prec_filtro['minimo_noites'] == 1)
condicao_bairro = (Tab_Prec_filtro['bairro'] == 'Midtown')
condicao_review = (Tab_Prec_filtro['numero_de_reviews'] > 0)
condicao_tipo = (Tab_Prec_filtro['room_type'] == 'Entire home/apt')
Tab_Prec_filtro = Tab_Prec_filtro[condicao_bairro & condicao_noites & condicao_tipo & condicao_review]

# Visualizando o dataset
Tab_Prec_filtro


# Para as váriaveis do meu modelo de Regressão Linear utilizei: 
# 
# - y: Váriavel dependente: 'price'
# - x: Váriavel independente: 'reviews'

# In[16]:


# importando a biblioteca
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# declarando nossas variáveis
X = Tab_Prec_filtro[['numero_de_reviews']]
y = Tab_Prec_filtro['price']

# declarando o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# exibindo os coeficientes e o termo independente
print('Coeficiente:', modelo.coef_[0])
print('Termo Independente:', modelo.intercept_)

previsoes = modelo.predict(X)


# calculando o erro médio quadrático
mse = mean_squared_error(y, previsoes)
print('Erro Médio Quadrático:', mse)

# calculando o coeficiente de determinação (R²)
r2 = r2_score(y, previsoes)
print('Coeficiente de Determinação (R²):', r2)


#importando a biblioteca para o gráfico
import seaborn as sns
import matplotlib.pyplot as plt

# Configurações do gráfico
sns.regplot(x='numero_de_reviews', y='price', data=Tab_Prec_filtro, scatter_kws={'s': 10}, line_kws={'color': 'red'})
plt.xlabel('Numero de Reviews')
plt.ylabel('Preço')
plt.title('Regressão Linear Simples')

# Exibindo o gráfico
plt.show()



# In[19]:


termo_independente = 158.72722246861176
coeficiente_reviews = -0.23149800684128988
reviews = 45

preco_previsto = termo_independente + (coeficiente_reviews * reviews)

# Exiba o resultado
print(f"O preço previsto para o imóvel é: {preco_previsto}")


# - Com base em minha análise demostrada acima, o preço do imóvel do exercício com base na projeção de reviews é de $148.
# 
# - O Coeficiente de Determinação (R²): 0.004006272087109575 sugere que o modelo tem um ajuste limitado aos dados, ou seja, esta variável tem pouca influencia no valor do imóvel.
#     
# - Já o coefieciente (coeficiente para a variável independente é -0.2315) nos mostra que a medida que o preço diminiu a quantidade de reviews aumenta, como mostra no gráfico.    

import pickle

