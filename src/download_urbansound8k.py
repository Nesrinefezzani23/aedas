import soundata

dataset = soundata.initialize('urbansound8k', data_home='data/raw/urbansound8k')
dataset.download()
dataset.validate()