# 1d Audio VQ-GAN

## What exactly is this thing bro?

I first train a VAE to compress the raw wav data (**not** the spectogram but the individual samples).
The VQ-VAE is a VAE using a code book to make the VAE not only more robust to noise, but also to be able to give give
out indices,
which can be the input of transformer for the classification of the audio sample.

## Why not using a spectrogram ???

1. (MEL-)Spectrogram give a destructive representation of the data
2. Using a CNN on spectrogram data does not make sense, as one would compress the audio over multiple sample **over a
   fixed frequency domain!**

   Not using a spectrogram allows me to use a (1D-)CNN to compress the individual raw samples of the audio

## Run

**Train VQ-GAN**: `py train_spectrosformer_single_speaker.py`
**Train transformer**: `py train_spectrosformer_single_speaker.py`

Ignore the `main.py` (legacy code)

**Note** that there is a bug that i am avoiding where the input **has** the be **divisible by 16** or else the
reconstruction is of different size.

### System
I used python 3.9.9 with the packages in the requirements.txt.
Note that I assume that you have a gpu to run the code