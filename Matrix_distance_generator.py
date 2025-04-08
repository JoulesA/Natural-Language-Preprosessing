# Libraries 
import numpy as np
import nltk
from nltk.corpus import PlaintextCorpusReader, stopwords
nltk.download('stopwords')

# Si se ejecuta desde colab descomentar las siguientes lineas
# from google.colab import drive
# # Montar el Google Drive
# drive.mount('/content/drive')

# Functions 
def Save_MTX(MTX, name, path):
  np.save(path + name, MTX)
  print('Matriz guardada')

def CleanWord(w):
    w = w.lower()
    w = w.replace('.', '')
    w = w.replace(',', '')
    w = w.replace(';', '')
    w = w.replace(':', '')
    w = w.replace('!', '')
    w = w.replace('?', '')
    w = w.replace('(', '')
    w = w.replace(')', '')
    w = w.replace('[', '')
    w = w.replace(']', '')
    w = w.replace('{', '')
    w = w.replace('}', '')
    w = w.replace('"', '')
    w = w.replace("'", '')
    w = w.replace('-', '')
    w = w.replace('_', '')
    w = w.replace('%', '')
    w = w.replace('*', '')
    w = w.replace('¿', '')
    w = w.replace('?', '')
    w = w.replace('¡', '')
    w = w.replace('!', '')
    w = w.replace('$', '')
    w = w.replace('`', '')
    w = w.replace('&', '')
    w = w.replace('+', '')
    w = w.replace('/', '')
    w = w.replace('\\', '')
    w = w.replace('=', '')
    w = w.replace('>', '')
    w = w.replace('<', '')
    w = w.replace('~', '')
    w = w.replace('^', '')
    w = w.replace('@', '')
    w = w.replace('#', '')
    w = w.replace('°', '')
    w = w.replace('º', '')
    w = w.replace('ª', '')
    return w

def CleanTxt(txt, Lowercase = True, SpecialCharacters = True, Numbers = True,
             NumberLabel = 'num', Stopwords = True, language='spanish'):

  # Load Stopwords
  if Stopwords:
    stop_words = set(stopwords.words(language))

  for i,w in enumerate (txt):

    # Lower case
    if Lowercase:
      w = w.lower()

    # Special Characters
    if SpecialCharacters:
      w = CleanWord(w)

    # Numbers
    if Numbers:
      try:
        w = int(w)
        if type(w) == int:
          w = NumberLabel
      except:
        pass

    # StopWords
    if Stopwords:
      if w in stop_words:
        w = ''
    # Sobreescribir la palabra por la correccion
    txt[i] = w

  # Eliminar espacios vacios correspondientes a caracteres especiales
  txt = list(filter(None, txt))
  return txt

def Encoding_txt (Voca, Pseu, TF='count', k = 10, b = 0.75):
  Voc_Codec = np.zeros((len(Vocabulario),len(Vocabulario)))
  if TF == 'count':
    for i,psdoc in enumerate(Pseu):
      for word in psdoc:
        word_index = Voca.index(word)
        if TF == 'count':
          Voc_Codec[i,word_index] += 1

  elif TF == 'binary':
    for i,psdoc in enumerate(Pseu):
      for word in psdoc:
        word_index = Voca.index(word)
        Voc_Codec[i,word_index] = 1

  elif TF == 'unitvec':
    for i,psdoc in enumerate(Pseu):
      for word in psdoc:
        word_index = Voca.index(word)
        Voc_Codec[i,word_index] += 1
      Voc_Codec[i] = Voc_Codec[i] / np.linalg.norm(Voc_Codec[i])

  elif TF == 'normalized':
    for i,psdoc in enumerate(Pseu):
      for word in psdoc:
        word_index = Voca.index(word)
        Voc_Codec[i,word_index] += 1
      Voc_Codec[i] = Voc_Codec[i] / len(psdoc)

  elif TF == 'log':
    for i,psdoc in enumerate(Pseu):
      for word in psdoc:
        word_index = Voca.index(word)
        Voc_Codec[i,word_index] += 1
      for j in range(len(Voc_Codec[i])):
        Voc_Codec[i,j] = np.log(1 + Voc_Codec[i,j])

  elif TF == 'BMK':
      if k == 0:
          Voc_Codec = Encoding_txt(Voca, Pseu, TF='binary')
      else:
        for i,psdoc in enumerate(Pseu):
          for word in psdoc:
            word_index = Voca.index(word)
            Voc_Codec[i,word_index] += 1
          for j in range(len(Voc_Codec[i])):
            Voc_Codec[i,j] = ((k+1)*Voc_Codec[i,j])/(k+Voc_Codec[i,j])
  elif TF == 'TFIDF':
       if k == 0:
          Voc_Codec = Encoding_txt(Voca, Pseu, TF='binary')
       else:
        # Calcular el promedio del tamaño del conntexto de cada palabra
        for i,psdoc in enumerate(Pseu):
          if i == 0:
            PROM = len(psdoc)
          else:
            PROM = (PROM + len(psdoc)/2)

        # Calculo de BM25
        Docu_freq = Voc_Codec
        for i,psdoc in enumerate(Pseu):
          for word in psdoc:
            word_index = Voca.index(word)
            Voc_Codec[i,word_index] += 1
            Docu_freq[1,word_index]  = 1
          for j in range(len(Voc_Codec[i])):
            Voc_Codec[i,j] = ((k+1)*Voc_Codec[i,j])/(k+((1-b)+b*(len(psdoc)/PROM))*Voc_Codec[i,j])

        # IDF
        Docu_freq = np.sum(Docu_freq, axis=0)
        for i in range(len(Docu_freq)):
          Docu_freq[i] = np.log((len(Voca)+1)/Docu_freq[i])

        # Multiplicacion por IDF
        for i,word in enumerate(Voc_Codec):
          Voc_Codec[i,:] = np.multiply(Voc_Codec[i,:],Docu_freq)

  return Voc_Codec

def Similitud_MTX(Voca, Voc_Codec, metrica = 'distancia'):
  Similitud = np.zeros((len(Voca),len(Voca)))

  if metrica == 'distancia':
    for i,vec1 in enumerate(Voc_Codec):
      for j,vec2 in enumerate(Voc_Codec):
        if i == j:
          Similitud[i,j] = 0
        elif i > j:
          Similitud[i,j] = Similitud[j,i]
        else:
          dif_vec = vec1 - vec2
          Similitud[i,j] = np.linalg.norm(dif_vec)

  elif metrica == 'coseno':
    for i,vec1 in enumerate(Voc_Codec):
      for j,vec2 in enumerate(Voc_Codec):
        if i == j:
          Similitud[i,j] = 1
        elif i > j:
          Similitud[i,j] = Similitud[j,i]
        else:
          Similitud[i,j] = np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

  return Similitud

def similitud_w2w(palabra1,palabra2, Voca, S_MTX):
  index1 = Voca.index(palabra1)
  index2 = Voca.index(palabra2)
  return S_MTX[index1,index2]

def get_Coeficiente(element):
  return element['Coeficiente']

def simil_w2all(palabra, Voca, S_MTX, All = True, k = 10):
  similitudes = []
  wordindex = Voca.index(palabra)

  for i,w in enumerate(Voca):
    similitudes.append({'Palabra' : w , 'Coeficiente': S_MTX[wordindex,i]})

  # Ordenar por mayor a menor
  similitudes.sort(key= get_Coeficiente)

  if All:
    for tupla in similitudes:
      print(tupla['Palabra'],tupla['Coeficiente'])
  else:
    for i in range(k):
      print(similitudes[i]['Palabra'],similitudes[i]['Coeficiente'])

  return similitudes


# Variables 
# Ajustar según donde esten tus archivos
path = '/content/drive/My Drive/NLP/'
filename = 'e990505_mod_lemmatized_spacy.txt'

# RUN

# Carga del archivo y separar en palabras
with open(path + filename, 'r') as file:
    text = file.read()
text = text.split()

# Limpiar texto
text = CleanTxt(text)

# Obtener vocabulario y ordenarlo
Vocabulario = []
for w in text:
  if w not in Vocabulario:
    Vocabulario.append(w)

Vocabulario.sort()
print(len(Vocabulario))
# print(Vocabulario)

# Pseudodocumento bag of words 
PseudocumentLR8 = []

# Vector del mismo tamaño que el vocabulario para almacenar contexto de cada palabra
for i in range(len(Vocabulario)):
  PseudocumentLR8.append([])

for i in range(len(text)):
  if i == 0 | i == 1 |i == 2 |i == 3:
    Rcontext = text[i+1:i+5]

  elif i == len(text)-1 | i == len(text)-2 |i == len(text)-3 |i == len(text)-4:
    Lcontext = text[i-5:i-1]

  else:
    Lcontext = text[i-5:i-1]
    Rcontext = text[i+1:i+5]

  # Añadir contexto a cada palabra
  word_index = Vocabulario.index(text[i])
  PseudocumentLR8[word_index].extend(Lcontext)
  PseudocumentLR8[word_index].extend(Rcontext)

# Comprobación: Vocabulario y Pseudodocumento deben tener misma longitud
print(len(PseudocumentLR8))
print(len(Vocabulario))

# Codificacion
Voc_encoding = Encoding_txt(Vocabulario, PseudocumentLR8, TF='count')
Voc_encoding_binary = Encoding_txt(Vocabulario, PseudocumentLR8, TF='binary')
Voc_encoding_unitvec = Encoding_txt(Vocabulario, PseudocumentLR8, TF='unitvec')
Voc_encoding_normalized = Encoding_txt(Vocabulario, PseudocumentLR8, TF='normalized')
Voc_encoding_log = Encoding_txt(Vocabulario, PseudocumentLR8, TF='log')
Voc_encoding_BMK = Encoding_txt(Vocabulario, PseudocumentLR8, TF='BMK')
Voc_encoding_TFIDF = Encoding_txt(Vocabulario, PseudocumentLR8, TF='TFIDF')

# Obtener matrices de similitud
MTX_dist_count = Similitud_MTX(Vocabulario, Voc_encoding, metrica = 'distancia')
MTX_cos_count = Similitud_MTX(Vocabulario, Voc_encoding, metrica = 'coseno')
Save_MTX(MTX_dist_count, 'MTX_dist_count', path)
Save_MTX(MTX_cos_count, 'MTX_cos_count', path)

MTX_dist_binary = Similitud_MTX(Vocabulario, Voc_encoding_binary, metrica = 'distancia')
MTX_cos_binary = Similitud_MTX(Vocabulario, Voc_encoding_binary, metrica = 'coseno')
Save_MTX(MTX_dist_binary, 'MTX_dist_binary', path)
Save_MTX(MTX_cos_binary, 'MTX_cos_binary', path)

MTX_dist_unitvec = Similitud_MTX(Vocabulario, Voc_encoding_unitvec, metrica = 'distancia')
MTX_cos_unitvec = Similitud_MTX(Vocabulario, Voc_encoding_unitvec, metrica = 'coseno')
Save_MTX(MTX_dist_unitvec, 'MTX_dist_unitvec', path)
Save_MTX(MTX_cos_unitvec, 'MTX_cos_unitvec', path)

MTX_dist_normalized = Similitud_MTX(Vocabulario, Voc_encoding_normalized, metrica = 'distancia')
MTX_cos_normalized = Similitud_MTX(Vocabulario, Voc_encoding_normalized, metrica = 'coseno')
Save_MTX(MTX_dist_normalized, 'MTX_dist_normalized', path)
Save_MTX(MTX_cos_normalized, 'MTX_cos_normalized', path)

MTX_dist_log = Similitud_MTX(Vocabulario, Voc_encoding_log, metrica = 'distancia')
MTX_cos_log = Similitud_MTX(Vocabulario, Voc_encoding_log, metrica = 'coseno')
Save_MTX(MTX_dist_log, 'MTX_dist_log', path)
Save_MTX(MTX_cos_log, 'MTX_cos_log', path)

MtX_dist_BMK = Similitud_MTX(Vocabulario, Voc_encoding_BMK, metrica = 'distancia')
MtX_cos_BMK = Similitud_MTX(Vocabulario, Voc_encoding_BMK, metrica = 'coseno')
Save_MTX(MtX_dist_BMK, 'MtX_dist_BMK', path)
Save_MTX(MtX_cos_BMK, 'MtX_cos_BMK', path)

MTX_dist_TFIDF = Similitud_MTX(Vocabulario, Voc_encoding_TFIDF, metrica = 'distancia')
MTX_cos_TFIDF = Similitud_MTX(Vocabulario, Voc_encoding_TFIDF, metrica = 'coseno')
Save_MTX(MTX_dist_TFIDF, 'MTX_dist_TFIDF', path)
Save_MTX(MTX_cos_TFIDF, 'MTX_cos_TFIDF', path)

# Visualizacion, Comentar si no deseas visualizar aqui
listaSimilitudesPalabra2all = simil_w2all('señalar', Vocabulario, MTX_cos_count, All = False)
listaSimilitudesPalabra2all = simil_w2all('señalar', Vocabulario, MTX_dist_count, All = False)