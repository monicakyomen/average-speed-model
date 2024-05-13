import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import matplotlib.pyplot as plt

tf.get_logger().setLevel(logging.ERROR)

tempo = np.array([6,5, 3, 6, 9,4,4, 2, 12, 20, 24, 35, 40, 55], dtype=float)        # features de entrada em horas
distancia = np.array([150, 200, 63, 75, 96, 226, 120, 198, 400, 500, 624, 750, 900, 1000], dtype=float) # km

velocidade_media = np.array([25, 20, 21, 12.5, 10.66, 56.50, 30, 30, 33.33, 25, 26, 21.43, 22.50, 18.18], dtype=float)  # velocidade média a ser prevista


from mpl_toolkits.mplot3d import Axes3D


# Criar figura e eixo 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotar o gráfico de dispersão tridimensional
ax.scatter(tempo, distancia, velocidade_media, c='black', marker='o', label='Dados')

# Definir rótulos dos eixos
ax.set_xlabel('Tempo (horas)')
ax.set_ylabel('Distância (km)')
ax.set_zlabel('Velocidade Média (km/h)')

# Definir título do gráfico
ax.set_title('Gráfico 3D - Tempo, Distância vs. Velocidade Média')

# Exibir a legenda
ax.legend()

# Exibir o gráfico
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(tempo, velocidade_media, c='black', marker='o', label='Dados')
plt.xlabel('Tempo (horas)')
plt.ylabel('Velocidade Média (km/h)')
plt.title('Gráfico de Dispersão 2D - Tempo vs. Velocidade Média')
plt.grid(True)
plt.legend()
plt.show()




# Criar modelo sequencial
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),                            # Camada de entrada especificando a forma dos dados
    tf.keras.layers.Dense(10, activation='relu'),                  # Primeira camada oculta com 10 neurônios
    tf.keras.layers.Dense(10, activation='relu'),                  # Segunda camada oculta com 10 neurônios
    tf.keras.layers.Dense(1)                                       # Camada de saída com 1 neurônio (saída única)
])

modelo.compile(loss='mean_squared_error', optimizer='sgd')

historico = modelo.fit(x=np.column_stack((tempo, distancia)), y=velocidade_media, batch_size=2, epochs=1000, verbose=0)

plt.figure(figsize=(10, 5))
plt.plot(historico.history['loss'])
plt.title('Histórico de Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treinamento'], loc='upper right')
plt.show()

novos_valores_tempo = np.array([2, 8, 10], dtype=float)
novos_valores_distancia = np.array([34, 50, 25], dtype=float)

novos_valores_tempo_em_segundos = novos_valores_tempo * 3600  # horas para segundos
novos_valores_distancia_em_metros = novos_valores_distancia * 1000  # km para metros

novos_valores_entrada = np.column_stack((novos_valores_tempo, novos_valores_distancia))
previsoes = modelo.predict(novos_valores_entrada)

for i, valor_tempo in enumerate(novos_valores_tempo):
    print(f"Para tempo {valor_tempo} horas e distância {novos_valores_distancia[i]} km, a velocidade média prevista é {previsoes[i][0]}")
