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

# %% [markdown] id="QsjBhhCFJsOU"
# # Inference session with our whale pod size classifier built on PANN CNN14.

# %% [markdown]
# ## Préparation

# %% id="gpIOxtEdMFkp"
import pandas as pd
import numpy as np
import torchaudio
from panns_inference import AudioTagging
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
from torchsummary import summary

# %% colab={"base_uri": "https://localhost:8080/"} id="FyD_4ynSOs18" outputId="bbe87f37-34bb-4f1a-b79f-aeb9749e4f81"
# Charger le modèle sur CNN14
audio_tagging = AudioTagging(checkpoint_path=None)

# %% id="8hvtMBAJj0Ra"
input_file = 'annotations_test.xlsx'

# Charger le fichier Excel
df = pd.read_excel(input_file)

# Ajouter une colonne pour les embeddings
df["embeddings"] = None

# Ajouter chaque embeddings de chaque instance dans le dataframe
for index, row in df.iterrows():
    try:
      # Charger la donnée audio depuis la source
      wav_path = '/content/' + str(row['filename'])

      # Obtenir le bon format
      waveform, sample_rate = torchaudio.load(wav_path)

      # Applique le modèle d'extraction d'embeddings à des données audio WAV
      _, embedding = audio_tagging.inference(waveform)

      # Ajouter le résultat dans le DataFrame
      df.at[index, "embeddings"] = embedding
    except Exception:
        print("Erreur")


# %% id="h4D3h9avv1QR"
# Dataset des données pour le dataloader
class CustomDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(data["embeddings"].tolist(), dtype=torch.float32)
        self.labels = torch.tensor(data["target"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Création des datasets
test_dataset = CustomDataset(df)

# %% id="f3E9jhVdv_UK"
# Création des DataLoaders
batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %% colab={"base_uri": "https://localhost:8080/"} id="Dee9MV0h_paF" outputId="1da52dfc-92c3-41c4-c79e-6cd1084af496"
# Entrée : Un vecteur de taille 2048 (représentant un embedding)
# Couche cachée : des couches denses avec 128 neurones et une activation ReLU (pleinement connectées)
# Couche sortie : nombre de classe

class MLP(nn.Module):
    def __init__(self, input_size = 2048, num_classes = 5):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)

        self.relu = nn.ReLU()

        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.out(x)
        return x

# Initialisation du modèle
model = MLP(input_size=2048, num_classes= 5).to('cuda')

# Résumé de l'architecture
summary(model, input_size=(2048,))

# %% [markdown] id="RcjNzqBWJyQ4"
# ## Chargement du modèle déjà entrainé

# %% colab={"base_uri": "https://localhost:8080/"} id="-5Bd7r3i_mXS" outputId="0ef1c397-37f7-4713-86dd-a8a5b97b025b"
# Charger le modèle
model = MLP()
model.load_state_dict(torch.load('MLP_on_CNN14.pth'))
model.to('cuda')

# %% [markdown] id="0P0KlP0bJ27t"
# ## Inférence

# %% id="nrFlpxb5NLEt"
input_file = 'annotations_test.xlsx'
output_file = 'test_predictions_CNN14.xlsx'

# Charger le fichier Excel
df = pd.read_excel(input_file)

# Ajouter une colonne pour les embeddings
df["embeddings"] = None

# Ajouter une colonne pour la prédiction
df['Prediction'] = None

model.eval()
with torch.no_grad():
  # Ajouter chaque embeddings de chaque instance dans le dataframe
  for index, row in df.iterrows():
      try:
        # Charger la donnée audio depuis la source
        wav_path = '/content/' + str(row['filename'])

        # Obtenir le bon format
        waveform, sample_rate = torchaudio.load(wav_path)

        # Applique le modèle d'extraction d'embeddings à des données audio WAV
        _, embedding = audio_tagging.inference(waveform)

        embedding_torch = torch.from_numpy(embedding).unsqueeze(0).float().to('cuda')

        # Inférence avec le modèle
        outputs = model(embedding_torch).squeeze(1)

        # Déterminer la classe prédite
        prediction = torch.argmax(outputs, dim=1).item()

        # # Ajouter le résultat dans le DataFrame
        df.at[index, "embeddings"] = embedding
        df.at[index, "Prediction"] = prediction
      except Exception as e:
          print(f"Erreur")

# Sauvegarder le DataFrame avec les prédictions dans un nouveau fichier Excel
df.to_excel(output_file, index=False)

# %% [markdown] id="Pe-puMacNNnt"
# ## Évaluation

# %% colab={"base_uri": "https://localhost:8080/", "height": 572} id="4sEnHd4zBM5W" outputId="b34bc90b-2cd0-48ee-e418-bb2b184b1c86"
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Mettre les labels dans une liste
true_labels = []
predicted_labels = []

model.eval()
with torch.no_grad():
    for features, labels in test_loader:

        # X (embedding), y (target)
        features, labels = features.to('cuda'), labels.to('cuda')

        # Inférence avec le modèle
        outputs = model(features).squeeze(1)

        # Déterminer la classe prédite
        predictions = torch.argmax(outputs, dim=1)

        # Ajouter les labels et prédictions aux listes
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predictions.cpu().numpy())

# Convertir en tableaux numpy
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Afficher la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cbar=True, annot_kws={'size': 12})
plt.title('Matrice de Confusion', fontsize=16)
plt.xlabel('Prédictions', fontsize=12)
plt.ylabel('Véritables Labels', fontsize=12)

# Ajuster les labels
classes = range(len(conf_matrix))
plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)

plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="wAoyzy8kEokq" outputId="be07a2a8-fd7c-41cb-8fc3-d06f5893095c"
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Calcul de l'accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calcul des métriques micro
micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
    true_labels, predicted_labels, average='micro'
)

# Calcul des métriques macro
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    true_labels, predicted_labels, average='macro'
)

# Afficher les résultats
print(f"Accuracy: {accuracy:.2f}")

print("\n=== Micro Scores ===")
print(f"Micro Précision: {micro_precision:.2f}")
print(f"Micro Rappel: {micro_recall:.2f}")
print(f"Micro F1-Score: {micro_f1:.2f}")

print("\n=== Macro Scores ===")
print(f"Macro Précision: {macro_precision:.2f}")
print(f"Macro Rappel: {macro_recall:.2f}")
print(f"Macro F1-Score: {macro_f1:.2f}")
