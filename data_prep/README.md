This document describes the data downloading and pre-processing steps carried out for my project: Spectrum4Geo.

1. The SoundingEarth data as described in the paper: [Self-supervised Audiovisual Representation Learning for Remote Sensing Data](https://arxiv.org/abs/2108.00688) can be downloaded from https://zenodo.org/record/5600379. The downloded data will have high-resolution GoogleEarth imagery along with a `metadata.csv` containing details of the audio recording corresponding to the overhead imagery. Now to download raw audio data use the script `./get_raw_audio.sh` provided. 
- Download the dataset, you only need the metadata.csv and the GoogleEarth imagery 
- not all sounds in metadata.csv are avaiable!

2. Once all data is downloaded a quick sanity check can be run using the script `./sanity` which will just try to read the audio files and save those as torch tensors `.pt` in an temporary folder (tempfile). Moreover, it also saves the id of the audio samples that failed to be read into a file: `corrupt_ids_final.csv`.

3. `get_corrupted_raw_audio.py` can be used to redownload the samples 
inside of `corrupt_ids_final.csv`, but if an sample isnt avaiable anymore it is also not able to download these sample.

4. `clean_metadata.py`: This script performs simple pre-processing of the `description` column of `metadata.csv`. Moreover, it uses `geopy` to perform reverse geo-coding of the address from the given lattitude-longitude of the audio sample and adds that address along with the cleaned description of the audio. Finally, this script yields a file: `final_metadata.csv` containing pre-processed captions along with all other metadata for audio samples in our data.
- it will ignore the samples which are listed in `corrupt_ids_final.csv`
- it will ignore samples with an sample rate lesser than 16kHz and write the inside `sr_less_16k_ids_final.csv`
- it will create an `no_address_ids_final.csv` with samples without address information, but will not remove these
=> Resulting in an cleaned csv-file: `final_metadata.csv`

5. `data_split.py`: This script splits `final_metadata.csv` into train/val/test with ratio $70:10:20$ yielding `csv` files: `train_df.csv`, `validate_df.csv`, `test_df.csv` with size $35554$, $5079$, and $10159$ respectively.

6. Use `create_mel_spectrograms.py` to create spectrogramms of the samples and save these into the folder structure inside of /data
- the spectrogramm data ist stored in .npy files, to look at these spectrogramms, you can convert these to images by using the tools inside /tools
