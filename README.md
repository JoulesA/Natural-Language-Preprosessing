
# 🧠 Preprocesamiento de Texto en PLN

Este script implementa un pipeline completo de preprocesamiento de texto en Python para tareas de PLN, incluyendo limpieza, codificación, generación de pseudodocumentos, y cálculo de similitud entre palabras. 

---

## 📚 Librerías necesarias

```python
import numpy as np
import nltk
from nltk.corpus import PlaintextCorpusReader, stopwords
nltk.download('stopwords')
```

> Si estás usando Google Colab, descomenta las siguientes líneas para montar Google Drive:

```python
# from google.colab import drive
# drive.mount('/content/drive')
```

---

## 🛠️ Funciones

### `Save_MTX()`
Guarda una matriz NumPy en un archivo `.npy`.

```python
def Save_MTX(MTX, name, path):
  np.save(path + name, MTX)
  print('Matriz guardada')
```

---

### `CleanWord()`
Limpia una palabra removiendo puntuación y caracteres especiales.

```python
def CleanWord(w):
    w = w.lower()
    w = w.replace('.', '')  # Y así sucesivamente con múltiples símbolos...
    return w
```

---

### `CleanTxt()`
Realiza limpieza completa de una lista de palabras.

```python
def CleanTxt(txt, Lowercase=True, SpecialCharacters=True, Numbers=True,
             NumberLabel='num', Stopwords=True, language='spanish'):
    # Lógica de preprocesamiento
    return txt
```

Opciones:
- Conversión a minúsculas.
- Remoción de puntuación.
- Reemplazo de números por una etiqueta.
- Eliminación de stopwords.

---

### `Encoding_txt()`
Codifica el texto con diferentes esquemas de representación vectorial.

```python
def Encoding_txt(Voca, Pseu, TF='count', k=10, b=0.75):
    # Codificaciones: count, binary, unitvec, normalized, log, BMK, TFIDF
    return Voc_Codec
```

Tipos:
- `count`: frecuencia cruda.
- `binary`: presencia/ausencia.
- `unitvec`: vector unitario.
- `normalized`: normalizado por longitud.
- `log`: escala logarítmica.
- `BMK` y `TFIDF`: variantes más complejas con normalización contextual.

---

### `Similitud_MTX()`
Calcula una matriz de similitud entre vectores.

```python
def Similitud_MTX(Voca, Voc_Codec, metrica='distancia'):
    return Similitud
```

Opciones de métrica:
- `distancia` euclidiana.
- `coseno` de similitud.

---

### `similitud_w2w()`
Obtiene la similitud entre dos palabras.

```python
def similitud_w2w(palabra1, palabra2, Voca, S_MTX):
    return S_MTX[index1,index2]
```

---

### `simil_w2all()`
Calcula la similitud de una palabra con todas las demás.

```python
def simil_w2all(palabra, Voca, S_MTX, All=True, k=10):
    return similitudes
```

---

## 📂 Variables y Configuración

```python
path = '/content/drive/My Drive/NLP/'
filename = 'e990505_mod_lemmatized_spacy.txt'
```

---

## 🚀 Ejecución del pipeline

### 1. Cargar y tokenizar el texto
```python
with open(path + filename, 'r') as file:
    text = file.read()
text = text.split()
```

---

### 2. Limpiar el texto
```python
text = CleanTxt(text)
```

---

### 3. Crear el vocabulario
```python
Vocabulario = list(sorted(set(text)))
```

---

### 4. Generar pseudodocumentos con contexto
```python
PseudocumentLR8 = [[] for _ in range(len(Vocabulario))]

for i in range(len(text)):
    if i < 4:
        Rcontext = text[i+1:i+5]
        Lcontext = []
    elif i > len(text)-5:
        Lcontext = text[i-5:i-1]
        Rcontext = []
    else:
        Lcontext = text[i-5:i-1]
        Rcontext = text[i+1:i+5]

    word_index = Vocabulario.index(text[i])
    PseudocumentLR8[word_index].extend(Lcontext + Rcontext)
```

---

### 5. Codificar el vocabulario
```python
Voc_encoding = Encoding_txt(Vocabulario, PseudocumentLR8, TF='count')
Voc_encoding_binary = Encoding_txt(Vocabulario, PseudocumentLR8, TF='binary')
Voc_encoding_unitvec = Encoding_txt(Vocabulario, PseudocumentLR8, TF='unitvec')
Voc_encoding_normalized = Encoding_txt(Vocabulario, PseudocumentLR8, TF='normalized')
Voc_encoding_log = Encoding_txt(Vocabulario, PseudocumentLR8, TF='log')
Voc_encoding_BMK = Encoding_txt(Vocabulario, PseudocumentLR8, TF='BMK')
Voc_encoding_TFIDF = Encoding_txt(Vocabulario, PseudocumentLR8, TF='TFIDF')
```

---

### 6. Calcular matrices de similitud
```python
MTX_dist_count = Similitud_MTX(Vocabulario, Voc_encoding, metrica='distancia')
MTX_cos_count = Similitud_MTX(Vocabulario, Voc_encoding, metrica='coseno')

Save_MTX(MTX_dist_count, 'MTX_dist_count', path)
Save_MTX(MTX_cos_count, 'MTX_cos_count', path)

# Repetir con otras codificaciones...
```

---

## 📌 Notas adicionales

- Este código está diseñado para análisis semántico basado en contexto (ventana de palabras).
- El resultado final incluye matrices que permiten comparar palabras en base a su distribución contextual (similitud semántica).
- **Importante:** Si se desea usar el visualizador hay que generar las matrices de similitud primero
