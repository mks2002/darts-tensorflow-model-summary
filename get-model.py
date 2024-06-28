# this is version 1 of the code , here we save it into txt file , and then from that we try to create h5 file ....

import tensorflow as tf
# Required for HDF5 support
import h5py 

def extract_model_structure(meta_file, checkpoint_dir, output_txt_file):
    # Start a new session
    with tf.compat.v1.Session() as sess:
        # Load the meta graph and weights
        saver = tf.compat.v1.train.import_meta_graph(meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # Get the default graph
        graph = tf.compat.v1.get_default_graph()

        # Extract model structure ...
        model_structure = []
        for op in graph.get_operations():
            model_structure.append({
                'name': op.name,
                'type': op.type,
                'input_shape': [str(i.get_shape()) for i in op.inputs],
                'output_shape': [str(o.get_shape()) for o in op.outputs]
            })

        # Write model structure to a text file
        with open(output_txt_file, 'w') as f:
            for layer in model_structure:
                f.write(f"Layer Name: {layer['name']}\n")
                f.write(f"Type: {layer['type']}\n")
                f.write(f"Input Shape: {layer['input_shape']}\n")
                f.write(f"Output Shape: {layer['output_shape']}\n")
                f.write("\n")

        return model_structure
    
# Now create an HDF5 representation of the model structure
def create_h5_model_structure(model_structure, output_h5_file):
    with h5py.File(output_h5_file, 'w') as hf:
        for idx, layer in enumerate(model_structure):
            grp = hf.create_group(f'layer_{idx}')
            grp.create_dataset('name', data=layer['name'])
            grp.create_dataset('type', data=layer['type'])
            grp.create_dataset('input_shape', data=layer['input_shape'])
            grp.create_dataset('output_shape', data=layer['output_shape'])


# Define the paths to your saved model files
meta_file = 'outputs/train_model/output-epochs2/model-3111.meta'
checkpoint_dir = 'outputs/train_model/output-epochs2/'
output_txt_file = 'model_structure.txt'

# Extract model structure to text file
model_structure = extract_model_structure(meta_file, checkpoint_dir, output_txt_file)

# Print the extracted model structure
for layer in model_structure:
    print(layer)

output_h5_file = 'model_structure.h5'
create_h5_model_structure(model_structure, output_h5_file)

print(f"Model structure saved to {output_txt_file} and {output_h5_file}")