# FullSubNet+ Speech Enhancement

A deep learning project for **speech enhancement and noise suppression** using the **FullSubNet+** architecture and the **VoiceBank-DEMAND 16 kHz** dataset.

The goal of the project is to transform noisy speech into cleaner speech while preserving the important characteristics of the original speaker's voice.

## Overview

Speech recorded in real-world environments often contains unwanted background noise such as:

* Traffic
* Fans
* Machinery
* Room noise
* Electronic noise
* Environmental sounds

This project explores speech enhancement using a neural-network-based approach. The notebook currently focuses on loading the VoiceBank-DEMAND dataset, inspecting noisy and clean speech signals, visualizing their waveforms and spectrograms, and preparing PyTorch datasets and data loaders.

## Project Pipeline

```text
VoiceBank-DEMAND Dataset
          │
          ▼
   Noisy / Clean Audio
          │
          ▼
    Audio Inspection
          │
          ├── Waveform Visualization
          │
          └── Spectrogram Visualization
          │
          ▼
   PyTorch Dataset
          │
          ▼
     DataLoader
          │
          ▼
     FullSubNet+
          │
          ▼
  Enhanced Speech
          │
          ▼
   Quality Evaluation
```

## Dataset

The project uses:

**VoiceBank-DEMAND 16 kHz**

Hugging Face dataset:

`JacobLinCool/VoiceBank-DEMAND-16k`

The dataset provides paired:

* **Noisy speech**
* **Clean speech**

The notebook loads the dataset using:

```python
from datasets import load_dataset

ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
```

The audio samples contain both the waveform array and sampling rate.

## Current Implementation

### 1. Install Dependencies

The notebook installs the following packages:

```bash
pip install -q datasets transformers librosa soundfile torchaudio
```

Main libraries:

| Library               | Purpose                        |
| --------------------- | ------------------------------ |
| PyTorch               | Deep learning and data loading |
| Hugging Face Datasets | Dataset loading                |
| Librosa               | Audio processing               |
| SoundFile             | Audio I/O                      |
| Torchaudio            | PyTorch-based audio processing |
| NumPy                 | Numerical computation          |
| Matplotlib            | Visualization                  |

### 2. Load Dataset

```python
from datasets import load_dataset

ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
```

The notebook then examines the available dataset structure and features.

### 3. Extract Audio

For each sample, the notebook extracts noisy and clean speech:

```python
clean_audio = sample["clean"]["array"]
clean_sr = sample["clean"]["sampling_rate"]

noisy_audio = sample["noisy"]["array"]
noisy_sr = sample["noisy"]["sampling_rate"]
```

The project therefore follows the supervised-learning formulation:

```text
Noisy Speech ──► Neural Network ──► Enhanced Speech
                                      │
                                      ▼
                              Compare with Clean Speech
```

### 4. Audio Visualization

The notebook plays both noisy and clean speech and plots their waveforms.

This makes it possible to visually inspect how noise affects the speech signal.

### 5. Spectrogram Analysis

The noisy signal is converted into a Short-Time Fourier Transform (STFT):

```python
spec = librosa.stft(noisy_audio)
spec_db = librosa.amplitude_to_db(np.abs(spec))
```

The resulting spectrogram represents the distribution of signal energy across:

* Time
* Frequency

This is particularly important for FullSubNet+, because speech enhancement is performed primarily in the time-frequency domain.

## PyTorch Dataset

A custom PyTorch dataset is implemented:

```python
class VoiceBankDataset(Dataset):

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        noisy = sample["noisy"]["array"]
        clean = sample["clean"]["array"]

        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)

        return noisy, clean
```

This converts the Hugging Face dataset into a format compatible with PyTorch training.

Each training sample is:

```text
(noisy waveform, clean waveform)
```

## DataLoaders

Training and testing datasets are created:

```python
train_dataset = VoiceBankDataset(ds["train"])
test_dataset = VoiceBankDataset(ds["test"])
```

They are then wrapped in PyTorch DataLoaders:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2
)
```

### Training configuration

* Batch size: `8`
* Training data: shuffled
* Test data: not shuffled
* DataLoader workers: `2`

## FullSubNet+

The intended enhancement model is **FullSubNet+**.

FullSubNet+ is designed for monaural speech enhancement and operates in the time-frequency domain.

Conceptually:

```text
Noisy Waveform
      │
      ▼
      STFT
      │
      ▼
Noisy Complex Spectrum
      │
      ▼
 ┌───────────────┐
 │  FullSubNet+  │
 └───────────────┘
      │
      ▼
Estimated Clean Spectrum
      │
      ▼
     iSTFT
      │
      ▼
Enhanced Speech
```

The model learns the relationship between noisy and clean speech from paired training examples.

## Why FullSubNet+?

FullSubNet+ combines:

* Full-band processing
* Sub-band processing
* Spectral information
* Temporal information
* Local frequency information
* Global frequency information

This allows the model to handle both broad spectral characteristics and detailed frequency-band variations in speech.

## Training Objective

The fundamental supervised-learning objective is:

```text
Input:
    Noisy Speech

Target:
    Clean Speech

Model:
    FullSubNet+

Output:
    Enhanced Speech
```

The model parameters are optimized by minimizing the difference between the enhanced speech and the clean reference.

A suitable implementation can use spectral-domain and/or waveform-domain losses depending on the final FullSubNet+ implementation.

## Evaluation

After training, the enhanced speech can be compared against the noisy input and clean reference.

Useful speech-enhancement metrics include:

* **PESQ** — Perceptual Evaluation of Speech Quality
* **STOI** — Short-Time Objective Intelligibility
* **SI-SDR** — Scale-Invariant Signal-to-Distortion Ratio
* **SNR improvement**
* **DNSMOS**, where applicable

A useful comparison is:

```text
Noisy Speech
     │
     ├──► PESQ
     ├──► STOI
     └──► SI-SDR

Enhanced Speech
     │
     ├──► PESQ
     ├──► STOI
     └──► SI-SDR
```

The objective is to demonstrate that the enhanced signal provides better speech quality and intelligibility than the original noisy signal.

## Project Structure

A recommended project structure is:

```text
fullsubnet-plus/
│
├── fullsubnet+.ipynb
├── README.md
│
├── models/
│   └── fullsubnet_plus.py
│
├── datasets/
│   └── voicebank_dataset.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── evaluate.py
│
├── inference/
│   └── enhance.py
│
├── checkpoints/
│   └── best_model.pth
│
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd fullsubnet-plus
```

Install the dependencies:

```bash
pip install datasets transformers librosa soundfile torchaudio torch numpy matplotlib
```

## Running the Notebook

The notebook can be run using Google Colab or Jupyter Notebook.

```bash
jupyter notebook
```

Then open:

```text
fullsubnet+.ipynb
```

The notebook currently performs:

1. Dependency installation
2. Dataset loading
3. Dataset inspection
4. Audio extraction
5. Noisy/clean audio playback
6. Waveform visualization
7. Spectrogram visualization
8. PyTorch dataset creation
9. DataLoader creation

## Current Status

### Completed

* [x] Install audio-processing dependencies
* [x] Load VoiceBank-DEMAND 16 kHz dataset
* [x] Inspect dataset features
* [x] Extract noisy speech
* [x] Extract clean speech
* [x] Play noisy audio
* [x] Play clean audio
* [x] Plot waveforms
* [x] Generate noisy spectrogram
* [x] Create PyTorch Dataset
* [x] Create training DataLoader
* [x] Create testing DataLoader

### Next Steps

* [ ] Implement STFT preprocessing for model input
* [ ] Implement FullSubNet+ architecture
* [ ] Implement full-band network
* [ ] Implement sub-band network
* [ ] Implement complex spectral masking/estimation
* [ ] Implement training loop
* [ ] Add validation
* [ ] Train the model
* [ ] Save model checkpoints
* [ ] Implement inference
* [ ] Reconstruct enhanced waveform using iSTFT
* [ ] Calculate PESQ
* [ ] Calculate STOI
* [ ] Calculate SI-SDR
* [ ] Compare noisy vs. enhanced speech
* [ ] Add real-time or microphone-based enhancement

## Expected Result

The final system should take a noisy speech signal:

```text
Noisy Speech
     │
     ▼
   STFT
     │
     ▼
 FullSubNet+
     │
     ▼
Enhanced Spectrum
     │
     ▼
   iSTFT
     │
     ▼
Enhanced Speech
```

The expected result is speech with significantly reduced background noise while maintaining speech intelligibility and speaker characteristics.

## Technologies

* Python
* PyTorch
* Hugging Face Datasets
* Librosa
* Torchaudio
* NumPy
* Matplotlib
* FullSubNet+
* VoiceBank-DEMAND

## Applications

Speech enhancement systems can be used in:

* Voice assistants
* Video conferencing
* Mobile communication
* Hearing assistance
* Automatic speech recognition
* Audio recording
* Telecommunication
* Robotics
* Automotive voice systems
* Noise-robust speech recognition

## References

This project is based on the FullSubNet/FullSubNet+ approach for monaural speech enhancement and uses the VoiceBank-DEMAND 16 kHz dataset through Hugging Face.

## Author

**Anoop**

Engineering / AI-ML Project

---

### Note

The current notebook establishes the **dataset and preprocessing pipeline**, but it does **not yet contain the FullSubNet+ neural-network implementation or training loop**. Therefore, those sections in this README describe the intended next stage rather than claiming they are already implemented in the uploaded notebook.
