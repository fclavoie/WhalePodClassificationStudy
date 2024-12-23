# Whale Pod Classification Study

This repository contains the code and resources for the classification of whale pod sizes in noisy environments. The project leverages pre-trained models and audio data from the **NOAA Pacific Islands Passive Acoustic Network (PIPAN)** dataset.

## Overview

The study aims to develop a robust system for classifying the number of whales in a group based on their vocalizations, even in the presence of maritime noise. By utilizing embeddings extracted from pre-trained models such as **YAMNet** and **CNN14**, we implemented a Multi-Layer Perceptron (MLP) for classification tasks. This system can potentially contribute to passive acoustic monitoring solutions to prevent ship-whale collisions and mitigate the impact of anthropogenic noise on whale communication and behavior in busy waterways like the Saint Lawrence River.

### Key Features:
- **Data Augmentation**: Simulated synthetic data by superimposing whale vocalizations and maritime noise.
- **Model Comparison**: Evaluated YAMNet and CNN14 for performance in noisy and clean audio contexts.
- **Real Data Integration**: Used real acoustic data annotated manually to enhance model robustness.

### Dataset:
We use the **Pacific Islands Passive Acoustic Network (PIPAN) 10kHz Data**:

```
NOAA Pacific Islands Fisheries Science Center. 2021.
Pacific Islands Passive Acoustic Network (PIPAN) 10kHz Data.
NOAA National Centers for Environmental Information.
https://doi.org/10.25921/Z787-9Y54
(Accessed on 2024-10-20)
```

[Dataset Information and Access](https://data.noaa.gov/metaview/page?xml=NOAA/NESDIS/NGDC/MGG/passive_acoustic//iso/xml/PIFSC_HARP_10kHzDecimated.xml&view=getDataView)

### Code

Various scripts and notebooks are provided to assist with the extraction and processing of relevant audio segments for this study.
Note that in many of them, path to local folders should be adjusted.

## Reproduce the study

The minimal steps to reproduce the experimentation would be to:
- Start by downloading the **PIPAN** dataset from Google Cloud Storage (we used the Hawaii subset);
- Extract the metadata using *metadata_extraction.py*;
- Use the *flac_extraction* notebook to get annotated segments from the **PIFSC** research team;
- You can generate random samples in a given window of time using *audio_segmenter.py* and annotate them manually;
- The sample rate must be changed from 10kHz to 16kHz with *audio_resampler.py*;
- Create augmented files with *augmentations.py*;
- You are now ready to train your classifier!
