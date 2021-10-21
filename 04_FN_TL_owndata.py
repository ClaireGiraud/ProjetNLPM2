#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:25:07 2021

@author: Florence & Claire
"""

import pandas as pd # création de df
import matplotlib.pyplot as plt # plots

import numpy as np
import os as os


import tensorflow as tf # word to seq

# Pour le transfert learning
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

# evaluation des modèles
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

import seaborn as sns # representation graphique de la matrice de confusion



# -----------------------------------------------------------------------------------------

# definition des fonctions


def graph(conf_matrix): # conf_matrix = matrice de confusion
    # params graphiques
    plt.figure(figsize=(16, 10))
    ax= plt.subplot()
    # labels, title and ticks
    ax.set_xlabel('Predicted Labels', size=20)
    ax.set_ylabel('True Labels', size=20)
    ax.set_title('Confusion Matrix', size=20) 
    ax.xaxis.set_ticklabels([0,1], size=15)
    ax.yaxis.set_ticklabels([0,1], size=15)
    # heatmap
    sns.heatmap(conf_matrix, annot=True, ax = ax)
    
# Concatenation des colones text et titre
def clean_data(df):
    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']
    return df


# ------------------------------------------------------------------------------------------   

max_vocab =10000  

# -------------------------------------------------------------------------------------------

# import des données

Fausses = pd.read_csv("Fake.csv")
Fausses.drop(Fausses.index[10000:23481],0,inplace=True)
Vraies = pd.read_csv("True.csv")
Vraies.drop(Vraies.index[10000:21417],0,inplace=True)

# ajout de la colonne catégorie (à prédire)
Fausses['category'] = 0
Vraies['category'] = 1

# dataframe de travail
df = pd.concat([Fausses,Vraies]) 

# Concatenation des colones text et titre
clean_data(df)

# -------------------------------------------------------------------------------------------
## Préparation des données pour le transfert learning


print("Modèle de transfert learning")
print('##################################################################')
print('##################################################################')

news=df['text'].values.tolist()
#print(news)
cat=df['category'].values.tolist()


# Shuffle the data
seed = 2000
rng = np.random.RandomState(seed)
rng.shuffle(news)
rng = np.random.RandomState(seed)
rng.shuffle(cat)
# -------------------------------------------------------------------------------------------

# Extract a training & validation split
validation_split = 0.2
num_validation_news = int(validation_split * len(news))
X_train = news[-num_validation_news:]
X_test = news[:-num_validation_news]
train_cat = cat[-num_validation_news:]
test_cat = cat[:-num_validation_news]

# -------------------------------------------------------------------------------------------

#vectorization du jeu de données

vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

# -------------------------------------------------------------------------------------------
#importation du jeu de données annexe pour la création du modèle 

path_to_glove_file = os.path.join(os.path.expanduser("~"), "/Users/clairegiraud/Desktop/01. SupAgro IA /01. 3A - SDAA/Cours/05. Machine learning/00. Projet/glove/glove.6B.100d.txt")
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs


num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# -------------------------------------------------------------------------------------------
# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1


from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)
# -------------------------------------------------------------------------------------------

# création du modèle 
from tensorflow.keras import layers

int_sequences_input = keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(2, activation="softmax")(x)
mod_TL = keras.Model(int_sequences_input, preds)

# -------------------------------------------------------------------------------------------
# Ré-entrainement du modèle

x_train = vectorizer(np.array([[s] for s in X_train])).numpy()
x_test = vectorizer(np.array([[s] for s in X_test])).numpy()

y_train = np.array(train_cat)
y_test = np.array(test_cat)


mod_TL.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
mod_TL.fit(x_train, y_train, batch_size=30, epochs=3, validation_data=(x_test, y_test))

# -------------------------------------------------------------------------------------------
# jeu test et évaluation du modèle

own_data = pd.read_excel("notre_base.xlsx", header=None)
categorie = [0] * 16 + [1] * 15
own_data['category'] = categorie
own_data = own_data.rename(columns={0: 'text'})
#own_data = own_data.sample(frac = 1)

X_test = own_data['text']
y_test = own_data['category']


print('test du modèle de transfert learning')

mod_TL.evaluate(X_test, y_test)
pred = mod_TL.predict(X_test)

pred = pd.DataFrame(data=pred, columns=["0", "1"])
del pred['1']

pred_array = pred.to_numpy()

binary_predictions = []

for j in pred_array:
    if j >= 0.5:
        binary_predictions.append(0)
    else:
        binary_predictions.append(1) 
        
print('##################################################################')
print("méthode : Transfert Learning- jeu test : échantillon du jeu de données")
        
print('Accuracy on testing set:', accuracy_score(binary_predictions, y_test))
print('Precision on testing set:', precision_score(binary_predictions, y_test))

matrix_TL = confusion_matrix(binary_predictions, y_test, normalize='all')
graph (matrix_TL)







