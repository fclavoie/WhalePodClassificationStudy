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

# %% [markdown]
# # Implementation of whale pod size classifier using SpeechBrain CNN14.
#
# Fonctionnal implementation of classifier using the CNN14 model from SpeechBrain.
# Preliminary tests showed slightly lower score than the implemantation from PANN, possibly due to the corpus used for training.
#
# Further exploration with it could still be interesting, if time allows it.

# %%
import os
import random
from pathlib import Path

import speechbrain as sb
import torch
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.epoch_loop import EpochCounterWithStopper
from speechbrain.utils.metric_stats import ClassificationStats

# %%
folder = Path().resolve().parent
folder


# %%
def prepare_dataset(csv_path, folder):
    dataset = DynamicItemDataset.from_csv(csv_path=csv_path)

    @sb.utils.data_pipeline.takes("filename")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(filename):
        full_path = os.path.join(folder, filename)
        signal = read_audio(full_path)
        yield signal

    dataset.add_dynamic_item(audio_pipeline)

    @sb.utils.data_pipeline.takes("target")
    @sb.utils.data_pipeline.provides("target_label")
    def label_pipeline(target_str):
        label_tensor = torch.tensor(int(target_str), dtype=torch.long)
        yield label_tensor

    dataset.add_dynamic_item(label_pipeline)

    dataset.set_output_keys(["id", "sig", "target_label"])
    return dataset

def split_train_valid(dataset, valid_ratio=0.1):
    data_list = list(dataset.data.items())
    random.shuffle(data_list)

    N = len(data_list)
    valid_size = int(N * valid_ratio)

    valid = dict(data_list[:valid_size])
    train = dict(data_list[valid_size:])

    train_data = sb.dataio.dataset.DynamicItemDataset(data=train)
    valid_data = sb.dataio.dataset.DynamicItemDataset(data=valid)

    train_data.pipeline = dataset.pipeline
    valid_data.pipeline = dataset.pipeline

    return train_data, valid_data


def move_batch_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


# %%

pretrained_model = EncoderClassifier.from_hparams(
    source="speechbrain/cnn14-esc50", savedir="pretrained_models/cnn14-esc50"
)

class WhaleBrain(sb.Brain):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)

        self.train_loss_history = []
        self.valid_loss_history = []
        self.test_loss_history  = []

        self.train_accuracy_history = []
        self.valid_accuracy_history = []
        self.test_classification_summaries  = []

    def on_stage_start(self, stage, epoch=None):
        """Initialisation des métriques."""
        self.classification_stats = ClassificationStats()

    def compute_forward(self, batch, stage):
        """Forward pass: audio -> features -> CNN14 -> classif."""
        batch = move_batch_to_device(batch, self.device)
        wavs = batch["sig"]
        lens = torch.ones(wavs.size(0), device=wavs.device)

        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        emb = self.modules.embedding_model(feats)
        logits = self.modules.new_classifier(emb)
        return logits

    def compute_objectives(self, predictions, batch, stage):
        """Compute the loss + classification stats."""
        batch = move_batch_to_device(batch, self.device)
        targets = batch["target_label"]

        if predictions.dim() == 3 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)

        loss = F.cross_entropy(predictions, targets)

        predicted_class = torch.argmax(predictions, dim=-1)
        pred_str_list = [str(p.item()) for p in predicted_class]
        tgt_str_list  = [str(t.item()) for t in targets]

        self.classification_stats.append(
            ids=[str(i) for i in batch["id"]],
            predictions=pred_str_list,
            targets=tgt_str_list,
        )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """En fin d'étape, on récupère les métriques."""
        if stage == sb.Stage.TRAIN:
            accuracy = self.classification_stats.summarize("accuracy")
            print(f"End of epoch {epoch} | train loss = {stage_loss:.4f} | train acc = {accuracy:.4f}")
            
            self.train_loss_history.append(stage_loss)
            self.train_accuracy_history.append(accuracy)

        elif stage == sb.Stage.VALID:
            accuracy = self.classification_stats.summarize("accuracy")
            print(f"End of epoch {epoch} | valid loss = {stage_loss:.4f} | valid acc = {accuracy:.4f}")
            
            self.valid_loss_history.append(stage_loss)
            self.valid_accuracy_history.append(accuracy)

        elif stage == sb.Stage.TEST:
            summary = self.classification_stats.summarize()
            print("Classwise accuracy:", summary["classwise_accuracy"])

            self.test_classification_summaries.append(summary)



# %%
train_csv = folder / "dataset_16khz/train/annotations.csv"
test_csv = folder / "dataset_16khz/test/annotations.csv"
train_folder = folder / "dataset_16khz/train"
test_folder = folder / "dataset_16khz/test"

train_data = prepare_dataset(train_csv, train_folder)
train_data, valid_data = split_train_valid(train_data, valid_ratio=0.1)
test_data = prepare_dataset(test_csv, test_folder)

modules = {
    "compute_features": pretrained_model.mods["compute_features"],
    "mean_var_norm": pretrained_model.mods["mean_var_norm"],
    "embedding_model": pretrained_model.mods["embedding_model"],
    "new_classifier": sb.nnet.linear.Linear(input_size=2048, n_neurons=5),
}

freeze_cnn = False
if freeze_cnn:
    for p in modules["embedding_model"].parameters():
        p.requires_grad = False

brain = WhaleBrain(
    modules=modules,
    opt_class=lambda x: torch.optim.Adam(x, lr=0.0001),
    run_opts={"device": "cuda"},
    hparams={"batch_size": 8},
)

epoch_counter = EpochCounterWithStopper(
    limit=100,
    limit_to_stop=5,
    direction="min",
    limit_warmup=0,
)

checkpointer = Checkpointer(
    folder / "checkpoints/speechbrain_cnn14",
    recoverables={"brain": brain, "epochs": epoch_counter},
)
brain.checkpointer = checkpointer

train_loader = SaveableDataLoader(train_data, batch_size=8, shuffle=True)
valid_loader = SaveableDataLoader(valid_data, batch_size=8, shuffle=False)
test_loader = SaveableDataLoader(test_data, batch_size=8, shuffle=False)


brain.fit(epoch_counter=epoch_counter, train_set=train_loader, valid_set=valid_loader)

# %%
brain.evaluate(test_loader)
