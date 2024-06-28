# this is using tensorflow ...

'''
import tensorflow as tf
from tensorflow.keras.models import load_model

# Path to your HDF5 model file
h5_model_file = 'model_structure.h5'

# Load the HDF5 model
try:
    model = load_model(h5_model_file)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Print model summary
model.summary()
'''


import tensorflow as tf
from tensorflow.keras.models import load_model

# Path to your HDF5 model file
h5_model_file = 'model_structure.h5'

# Load the HDF5 model
try:
    model = load_model(h5_model_file)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Print model summary
model.summary()


# this is using h5 library ...

'''
import h5py
from tensorflow.keras.models import load_model

h5_model_file = 'model_structure.h5'

try:
    with h5py.File(h5_model_file, 'r') as f:
        # Check if the model config exists in the file
        if 'model_config' not in f.attrs:
            raise ValueError('No model config found in the HDF5 file.')

    # Load the model
    model = load_model(h5_model_file)
    model.summary()

    # Proceed with visualization or further operations
except Exception as e:
    print(f"Error loading the model: {e}")
'''    