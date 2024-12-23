# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown] id="zW8kd99368FP"
# # Training session with our whale pod size classifier built on YAMNet
#
# Note: Une grande partie du code ci-dessous à été extrait ou inspiré de la ressource pour utiliser YAMNet. Crédit à https://www.tensorflow.org/hub/tutorials/yamnet et https://www.tensorflow.org/hub/tutorials/yamnet

# %% [markdown] id="QuxFDZLRAUs1"
# ## Préparation

# %% id="xuL_NDp8cfa8"

import os
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

# %% id="NLiUau4lchFG"
# Chargement du modèle YAMNet de Google
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


# %% id="OKTN8Es1c3Vv"
# Fonctions utilitaires pour charger des fichiers audio et s'assurer que la fréquence d'échantillonnage est correcte
@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


# %% colab={"base_uri": "https://localhost:8080/"} id="wNVCbAw-toOT" outputId="fc3aa492-21e7-4989-c4d2-a9d4be766bf7"
print(len(os.listdir()))

# %% id="nyvEHHUSqKmj"
# Chargement de la données
file_path = '/content/annotations_clean_and_noised.xlsx'
df = pd.read_excel(file_path)

# %% colab={"base_uri": "https://localhost:8080/"} id="sIgAQI-Bd78x" outputId="e29a837e-b6f2-42e8-bdff-14caa99d46d7"
# Mettre les données dans le bon format (tensorflow)
filenames = df['filename']
targets = df['target']
folds = df['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec


# %% colab={"base_uri": "https://localhost:8080/"} id="479o5SBVrFux" outputId="cc8f61b5-10c0-4bad-983a-65a8c27a3233"
# Transformer les fichiers audio dans le format appropriés (établie par notre fonction intérieur)
def load_wav_for_map(filename, label, fold):
  return load_wav_16k_mono(filename), label, fold

main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# %% colab={"base_uri": "https://localhost:8080/"} id="vl40-gtcrHtR" outputId="96b20491-c725-460e-bd83-9054cc7fd425"
# Applique le modèle d'extraction d'embeddings à des données audio WAV
def extract_embedding(wav_data, label, fold):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))

# extraction d'embeddings pour chaque donnée
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec

# %% id="6826auhprJ5r"
# Split des données et mise en cache
cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 7)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 7)

# supprimer la colonne fold
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)

# mélange des données, batching, préchargement
train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

# %% id="Ib10Xyrzrh_b"
# Instancier les noms des nouveaux labels
my_classes = ['0', '1','2','3','4']
map_class_to_id = {'0':0, '1':1,'2':2,'3':3,'4':4}

# %% [markdown] id="GP9xuFSCAObK"
# ## Entrainement

# %% colab={"base_uri": "https://localhost:8080/"} id="mMGYCcHSrL1L" outputId="163b5670-b1b0-4587-f3d8-f99b2ca31e0c"
# Entrée : Un vecteur de taille 1024 (représentant un embedding)
# Couche cachée : 3 couche dense avec 64 neurones et une activation ReLU
# Couche sortie : nombre de classe

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(my_classes))
], name='MLP')

model.summary()

# %% id="l6PZ9tuNrkAA"
# prépare le modèle pour l'entraînement
# Crossentropyloss/ softmax pour la classfication
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

# EarlyStopping pour arrêter l'entraînement lorsque le modèle cesse d'améliorer
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=5,
                                            restore_best_weights=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="CEA9Jvqcrlyl" outputId="ae718d35-d128-4ca8-aa75-3bdc3f730ac3"
# Entraînement du modèle (Suivi de l'entrainement sur x epoch, en validation)
history = model.fit(train_ds,
                       epochs=100,
                       validation_data=val_ds,
                       callbacks=callback)


# %% [markdown] id="Mlq72-wc_zSD"
# ## Sauvegarder le modèle entrainé

# %% id="jsvv7ISoAz3k"
class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)


# %% colab={"base_uri": "https://localhost:8080/"} id="zpYr3v-9AcuD" outputId="2332881d-359e-44da-e2de-9f17de45ec2e"
saved_model_path = './Model_on_YAMNet_clean_and_noised'

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='YAMNet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)

# %% colab={"base_uri": "https://localhost:8080/"} id="3PUAHDhKC4bV" outputId="e1ed6884-4c41-44d9-ff2f-6e4bea01275e"
from google.colab import drive
drive.mount('/content/drive')

# %% id="KqgMCRFeC7kf"
# Téléchargement du modèle
# !cp -r /content/Model_on_YAMNet_clean_and_noised /content/drive/MyDrive/
