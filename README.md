# Music-Genre-Classification-CNN

## Run order of files:

### wav_convert.py  
Converts music files to the needed .wav format and store into ./converted  
### wav_extract.py
Extracts fft values and saves numpy files to ./fft_extracted  
### log_reg.py
Applies Logistic Regression to train and classify; gives prediction accuracy, saves confusion matrix image and model  
### test.py
Loads saved model to predict genre of new file  
