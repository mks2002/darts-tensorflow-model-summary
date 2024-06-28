from tensorflow.keras.models import load_model, save_model

# Load your existing model
model = load_model('model_structure.h5')

# Save the model again
save_model(model, 'new_model_structure.h5')

'''
this file is used for just to confirm , whether this auto conversion of old h5 file into new h5 file , can able to resolve the error which comes while model loading or not --->
    
error --->
No model config found in the file at <tensorflow.python.platform.gfile.GFile object at 0x7fc6e6636b20>

cant able to handle, even it cant able to save new model and gives the same error ...

'''