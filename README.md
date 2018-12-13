# Overview
Implementation for [QANet](https://arxiv.org/abs/1804.09541) using Keras with Tensorflow backend.

# Preparation (QANet implementation)
Setting variables are defined in `src/squad/config.py`

0. Preparation
    * Install Python packages in requirements.txt
    * Install English corpus for Spacy: `python -m spacy download en`

1. Download glove and extract the file `glove.6B.300d.txt` to `data/glove`
   (Setting variable: `EMBEDDING_FILE`)

2. Download SQuAD data v1.1 and extract the files `train-v1.1.json` & `dev-v1.1.json` to `data/SQUAD_Data/v1.1`
   (Setting variable: `TRAIN_JSON` & `DEV_JSON`)


# For inference using pre-trained model
* Download the model file `qanet_ep20.h5` from `https://github.com/nptdat/qanet/releases/download/v1.0/qanet_ep20.h5` and put it into `model` folder.
  (Setting variable: `INFERENCE_MODEL_PATH`)

* If you use the above model, I recommend you to download the following files from `https://github.com/nptdat/qanet/releases/download/v1.0` to ensure the data consistence:
  * `squad_processed-v1.1.pkl.zip`: unzip and move the pickle file to `data/SQUAD_Data/v1.1/`
  * `numpy_files.zip`: unzip and move all the .npy files to `data/SQUAD_Data/v1.1/numpy/`
  * Data from these files will overwrite those generated from `build_squad_data.py`

* Run
```
$ FLASK_APP=demo_qanet.py flask run --host=0.0.0.0 --port=8080
```
Then access `http://localhost:8080/qanet` via browser.

# For training
1. Run `build_squad_data.py` to load SQuAD data from json files, transform the data and save to .pkl files

```
$ python build_squad_data.py
```

2. Run `train.py`
```
$ python train.py
```

* Model files will be saved to `model` folder, 1 model per epoch
* Tensorboard log data will be saved to `log/tensorboard`
* Please take a look at config.py for further setting

# Unit Test
Please read `src/squad/test/README.md`
