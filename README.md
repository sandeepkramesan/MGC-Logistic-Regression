## Music Genre Classification using Logistic Regression

#### Dataset Heirarchy
```
gtzan_dataset
	\training_data
	\test_data
```

#### Run order of files:

```
python3 wav_convert.py
```
Converts music files to the needed .wav format and store into ./converted  

```
python3 wav_extract_fft.py
```
Extracts fft values and saves numpy files to ./fft_extracted  

```
python3 log_reg.py fft_extracted/
```
Applies Logistic Regression to train and classify; gives prediction accuracy, saves confusion matrix image and model  

```
python3 test.py ~/path/to/your/song/file
```
Loads saved model to predict genre of new file  
