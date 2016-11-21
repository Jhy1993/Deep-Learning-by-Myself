import h5py 
from keras.models import model_from_json 
model = model_from_json(open('model0813.json').read())  
model.load_weights('model0813.h5') 