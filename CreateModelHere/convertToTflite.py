import tensorflow as tf

# Step 1: Load your TensorFlow model (example with a Keras model)
# Replace 'your_model.h5' with your model file
model = tf.keras.models.load_model('model_au_normal.h5')

# If you have a SavedModel, use:
# model = tf.saved_model.load('path_to_saved_model')

# Step 2: Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# If using a SavedModel, use:
# converter = tf.lite.TFLiteConverter.from_saved_model('path_to_saved_model')

# Optional: Set optimization options (e.g., for smaller size or faster performance)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# You can also set other optimization options like:
# converter.target_spec.supported_types = [tf.float16]
# converter.representative_dataset = your_representative_dataset_generator

# Convert the model
tflite_model = converter.convert()

# Step 3: Save the converted model to a .tflite file
with open('normal_lite_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved to 'model.tflite'")
