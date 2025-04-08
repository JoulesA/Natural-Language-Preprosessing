# Libraries
import numpy as np

# Descomentar si se esta trabajando en colab 
# from google.colab import drive
# # Montar el Google Drive
# drive.mount('/content/drive')

# Functions
def similitud_w2w(palabra1,palabra2, Voca, S_MTX):
  index1 = Voca.index(palabra1)
  index2 = Voca.index(palabra2)
  return S_MTX[index1,index2]

def get_Coeficiente(element):
  return element['Coeficiente']

def simil_w2all(palabra, Voca, S_MTX, All = True, k = 10, coseno = False):
  similitudes = []
  wordindex = Voca.index(palabra)

  for i,w in enumerate(Voca):
    similitudes.append({'Palabra' : w , 'Coeficiente': S_MTX[wordindex,i]})

  # Ordenar por mayor a menor
  if coseno:
    similitudes.sort(key= get_Coeficiente, reverse=True)
  else:
    similitudes.sort(key= get_Coeficiente)

  if All:
    for tupla in similitudes:
      print(tupla['Palabra'],tupla['Coeficiente'])
  else:
    for i in range(k):
      print(similitudes[i]['Palabra'],similitudes[i]['Coeficiente'])

  return similitudes

# Variables
# Ajusta segun donde esten los archivos gurados 
path = '/content/drive/My Drive/NLP/'

# RUN
# Cargar matrices y vocabulario
Vocabulario = np.load(path + 'Vocabulario.npy')
Vocabulario = Vocabulario.tolist()

# En este caso es usando las matrices generadas en el otro programa 
# Si tienes una matriz propia deberas cambiar el nombre del archivo
MTX_dist_count = np.load(path + 'MTX_dist_count.npy')
MTX_cos_count = np.load(path + 'MTX_cos_count.npy')

MTX_dist_binary = np.load(path + 'MTX_dist_binary.npy')
MTX_cos_binary = np.load(path + 'MTX_cos_binary.npy')

MTX_dist_unitvec = np.load(path + 'MTX_dist_unitvec.npy')
MTX_cos_unitvec = np.load(path + 'MTX_cos_unitvec.npy')

MTX_dist_normalized = np.load(path + 'MTX_dist_normalized.npy')
MTX_cos_normalized = np.load(path + 'MTX_cos_normalized.npy')

MTX_dist_log = np.load(path + 'MTX_dist_log.npy')
MTX_cos_log = np.load(path + 'MTX_cos_log.npy')

MTX_dist_BMK = np.load(path + 'MtX_dist_BMK.npy')
MTX_cos_BMK = np.load(path + 'MtX_cos_BMK.npy')

MTX_dist_IDFTF = np.load(path + 'MTX_dist_TFIDF.npy')
MTX_cos_IDFTF = np.load(path + 'MTX_cos_TFIDF.npy')

# Visualización 
palabra = 'organización'

print('Metrica: Coseno')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_cos_count, All=False, k=20, coseno=True)
print('\nMetrica: Distancia entre vectores')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_dist_count, All = False, k=20)

print('Metrica: Coseno')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_cos_binary, All=False, k=20, coseno=True)
print('\nMetrica: Distancia entre vectores')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_dist_binary, All = False, k=20)

print('Metrica: Coseno')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_cos_unitvec, All=False, k=20, coseno=True)
print('\nMetrica: Distancia entre vectores')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_dist_unitvec, All = False, k=20)

print('Metrica: Coseno')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_cos_normalized, All=False, k=20, coseno=True)
print('\nMetrica: Distancia entre vectores')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_dist_normalized, All = False, k=20)

print('Metrica: Coseno')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_cos_log, All=False, k=20, coseno=True)
print('\nMetrica: Distancia entre vectores')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_dist_log, All = False, k=20)

print('Metrica: Coseno')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_cos_BMK, All=False, k=20, coseno=True)
print('\nMetrica: Distancia entre vectores')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_dist_BMK, All = False, k=20)

print('Metrica: Coseno')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_cos_IDFTF, All=False, k=20, coseno=True)
print('\nMetrica: Distancia entre vectores')
listaSimilitudesPalabra2all = simil_w2all(palabra, Vocabulario, MTX_dist_IDFTF, All = False, k=20)