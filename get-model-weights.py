# this code saves the model structure as well as model weights , into h5 format ...

import tensorflow as tf
import h5py  # Required for HDF5 support

def extract_model_structure(meta_file, checkpoint_dir, output_txt_file):
    # Start a new session
    with tf.compat.v1.Session() as sess:
        # Load the meta graph and weights
        saver = tf.compat.v1.train.import_meta_graph(meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # Get the default graph
        graph = tf.compat.v1.get_default_graph()

        # Extract and print model structure
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
    
# function to create h5 file from model_structure ...
def create_h5_model_structure(model_structure, output_h5_file):
    with h5py.File(output_h5_file, 'w') as hf:
        for idx, layer in enumerate(model_structure):
            grp = hf.create_group(f'layer_{idx}')
            grp.create_dataset('name', data=layer['name'])
            grp.create_dataset('type', data=layer['type'])
            grp.create_dataset('input_shape', data=layer['input_shape'])
            grp.create_dataset('output_shape', data=layer['output_shape'])    

# function to create another h5 file which contains model weights ...
def save_model_weights(meta_file,checkpoint_dir, output_weights_file):
    with tf.compat.v1.Session() as sess:
        # Load the meta graph and restore the variables
        saver = tf.compat.v1.train.import_meta_graph(meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # Get all trainable variables
        variables = tf.compat.v1.trainable_variables()
        
        # Create a dictionary to store variable names and their values
        weights_dict = {}
        for var in variables:
            weights_dict[var.name] = sess.run(var)

    # Save weights to HDF5 file
    with h5py.File(output_weights_file, 'w') as hf:
        for name, value in weights_dict.items():
            hf.create_dataset(name, data=value)

    print(f"Weights saved to {output_weights_file}")

# Define the paths to your saved model files
meta_file = 'outputs/train_model/output-epochs2/model-3111.meta'
checkpoint_dir = 'outputs/train_model/output-epochs2/'

output_txt_file = 'model_structure.txt'
output_h5_file = 'model_structure.h5'
output_weights_file = 'model_weights.h5'

# Extract model structure to text file
model_structure = extract_model_structure(meta_file, checkpoint_dir, output_txt_file)

# save model_structure to the HDF5 file ...
create_h5_model_structure(model_structure, output_h5_file)

# Save model weights to HDF5 file ...
save_model_weights(meta_file, checkpoint_dir, output_weights_file)

# Print information about saved files
print(f"Model structure saved to {output_txt_file} and {output_h5_file}")
print(f"Weights saved to {output_weights_file}")
