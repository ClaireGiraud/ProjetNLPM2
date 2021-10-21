#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:12:06 2021

@author: Florence & Claire
"""

# Import des packages

#import os #import des fichiers
import pandas as pd # création de df
import matplotlib.pyplot as plt # plots
import re # nettoyage des données

# repartition des echantillon train et test
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # tokenization
import tensorflow as tf # word to seq



# evaluation des modèles
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import seaborn as sns # representation graphique de la matrice de confusion

import os

# Renseignement du répertoire de travail
os.chdir('/Users/clairegiraud/Desktop/01. SupAgro IA /01. 3A - SDAA/Cours/05. Machine learning/00. Projet')

# paramètres graphiques
plt.style.use('ggplot')

# -----------------------------------------------------------------------------------------

# definition des fonctions

# normalisation on enlève les url, les espaces en trop, etc. 
def normalize(df):
    normalized = []
    for i in df:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

# préparation du jeu d'entrainement

def prep_train(data, max_vocab) : 
    data = normalize(data)
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(data) 
    # tokenize the text into vectors 
    data = tokenizer.texts_to_sequences(data)
    #mise sous forme d'une matrice de longeur 100, les séquences plus courtes sont ralongées avec une valeur 
    data = tf.keras.preprocessing.sequence.pad_sequences(data, 
                                                         padding='post', 
                                                         maxlen=256)
    return (data)
       

def test(data, y_res, mod, max_vocab): # data = le X à tester, res = la var réponse à comparer, model = modèle entrainé
    data = normalize(data)
    # tokenization
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(data)
    # création du vecteur
    data = tokenizer.texts_to_sequences(data)
    # tensorflow
    data = tf.keras.preprocessing.sequence.pad_sequences(data, 
                                                         padding='post', 
                                                         maxlen=256)
    # prediction 
    predicted_value = mod.predict(data)
    binary_predictions = []
    for i in predicted_value:
        if i >= 0.5:
            binary_predictions.append(1)
        else:
            binary_predictions.append(0) 
    #Matrice de confusion
    matrix = confusion_matrix(binary_predictions, y_res, normalize='all')
    ## Valeur d'accuracy et de précision      
    print('Accuracy on testing set:', accuracy_score(binary_predictions, y_res))
    print('Precision on testing set:', precision_score(binary_predictions, y_res))
    return(matrix)

# création des graphs (matrice de confusion)

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

# repartition des echantillon train et test
max_vocab =10000
X_train, X_test, y_train, y_test = train_test_split(df.text, 
                                                    df.category, 
                                                    test_size = 0.2,
                                                    random_state=2)

X_train = prep_train(X_train, max_vocab)


# --------------------------------------------------------------------------------------------
# Méthode de machine learning : random forest

# Modèle de Deep Learning

print('Méthode de deep learning: RNN from scratch')

mod_RNN = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab, 32),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(1, activation='sigmoid'), # fonction sigmoïde car 2 classes 
    tf.keras.layers.Dropout(0.5), # aide à réduire le risque de sur-apprentissage 
])
mod_RNN.summary()

#on stop le modèle quand l'accuracy n'augmente plus
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience=2, 
                                              restore_best_weights=True)

mod_RNN.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # changer l'argument from logits quand sigmoid --> FALSE
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

mod_RNN.fit(X_train, 
            y_train, 
            epochs=10, 
            validation_split=0.1, 
            batch_size=30, 
            shuffle=True, 
            callbacks=[early_stop])

print('##################################################################')
# -------------------------------------------------------------------------------------------
##### Tests du modèle -----------------------------------------------------------------------
print ('test des modèles')

print("méthode : Deep Learning - jeu test : échantillon du jeu de données")
# test du modèle de Deep Learning sur un échantillon issu du jeu de données

test_dataset_DL = test(X_test,y_test,mod_RNN, 10000)
graph(test_dataset_DL)

print('##################################################################')
print("méthode : Deep Learning - jeu test : jeu de données personnelles traduites")
# test sur un jeu de données personnel

own_data = pd.read_excel("notre_base.xlsx", header=None)
categorie = [0] * 16 + [1] * 15
own_data['category'] = categorie
own_data = own_data.rename(columns={0: 'text'})
own_data = own_data.sample(frac = 1)


test_own_data = test(own_data.text,own_data.category,mod_RNN, 10000)
graph(test_own_data)

print('##################################################################')