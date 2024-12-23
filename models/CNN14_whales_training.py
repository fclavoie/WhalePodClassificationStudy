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

# %% [markdown] id="Wyml0xMUKIGI"
# # Training session with our whale pod size classifier built on PANN CNN14.

# %% [markdown]
# ## Préparation

# %% id="gpIOxtEdMFkp"
import os
import pandas as pd
import torchaudio
from panns_inference import AudioTagging
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
from torchsummary import summary

# %% colab={"base_uri": "https://localhost:8080/"} id="FyD_4ynSOs18" outputId="00901fc9-c97d-4859-ad90-0c62e9b35570"
# Charger le modèle sur CNN14
audio_tagging = AudioTagging(checkpoint_path=None)

# %% colab={"base_uri": "https://localhost:8080/"} id="NjkYdHDj68MV" outputId="3e2c62af-3f0a-4dd4-80de-3dae5010f920"
print(len(os.listdir()))

# %% id="8hvtMBAJj0Ra"
input_file = 'annotations_clean_and_noised.xlsx'

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
    except Exception as e:
        print(f"Erreur")


# %% id="iQ5CXEBHvsCH"
# Split des données
def split_data(df):
    train_df = df[df["fold"].isin([1, 2, 3, 4, 5, 6])]
    valid_df = df[df["fold"] == 7]

    return train_df, valid_df

train_df, valid_df = split_data(df)


# %% colab={"base_uri": "https://localhost:8080/"} id="h4D3h9avv1QR" outputId="9f7ca1f7-01c1-4ec0-d9c9-0e6549310412"
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
train_dataset = CustomDataset(train_df)
valid_dataset = CustomDataset(valid_df)

# %% id="f3E9jhVdv_UK"
# Création des DataLoaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# %% [markdown] id="gz31egCMKLpY"
# ## Modèle

# %% colab={"base_uri": "https://localhost:8080/"} id="ZjM9evvZwDLE" outputId="243b5995-efc6-4d9d-ff47-e22ffa3d9f11"
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


# %% [markdown] id="wQwZFioEKRWg"
# ## Entrainement

# %% id="Xhx-iHU6Uv-p"
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, valid_loss):
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# %% id="gkmUIJVcU5Qw"
# Initialisation d'early stopping
early_stopping = EarlyStopping(patience=5)

# %% colab={"base_uri": "https://localhost:8080/"} id="yv9HrivmAOAe" outputId="441d6c86-d025-462b-bf3d-2d617e6f1d94"
# Entrainement
epochs = 100
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):

    # Entraînement
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features.to('cuda')).squeeze(1)

        # Calcul de la loss
        loss = criterion(outputs, labels.to('cuda'))
        loss.backward()
        optimizer.step()

        # Mise à jour des métriques d'entraînement
        total_train_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_train += (predictions.to('cuda') == labels.to('cuda')).sum().item()
        total_train += labels.size(0)

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    # Validation
    model.eval()
    total_valid_loss = 0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for features, labels in valid_loader:
            outputs = model(features.to('cuda')).squeeze(1)

            # Calcul de la loss
            loss = criterion(outputs, labels.to('cuda'))
            total_valid_loss += loss.item()

            # Mise à jour des métriques de validation
            predictions = torch.argmax(outputs, dim=1)
            correct_valid += (predictions == labels.to('cuda')).sum().item()
            total_valid += labels.size(0)

    valid_loss = total_valid_loss / len(valid_loader)
    valid_accuracy = correct_valid / total_valid

    # Métriques pour l'époque en cours
    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {train_loss:}, Train Accuracy: {train_accuracy:}")
    print(f"Valid Loss: {valid_loss:}, Valid Accuracy: {valid_accuracy:}")

    # Early stopping
    early_stopping(valid_loss)
    if early_stopping.early_stop:
        print("Early stopping stopped the training!")
        break

# %% [markdown] id="zamYdidGKag0"
# ## Sauvegarder le modèle entrainé

# %% id="fDPzHiL9eiSM"
torch.save(model.state_dict(), "MLP_on_CNN14_clean_and_noised.pth")
