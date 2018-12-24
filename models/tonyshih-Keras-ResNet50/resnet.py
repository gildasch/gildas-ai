import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
from keras import backend as K
import os
import sys

if not os.path.isfile("resnet50.h5"):
    os.system("curl -L -o resnet50.h5 http://modeldepot.io/assets/uploads/models/models/2fefdb45-9b31-45c6-a714-dc76f8576c6b_resnet50_weights_tf_dim_ordering_tf_kernels.h5")

if len(sys.argv[1:]) > 0:
    modelName = sys.argv[1]
else:
    modelName = "resnet"

sess = tf.Session()
K.set_session(sess)

model = ResNet50(weights='resnet50.h5')

img_path = 'gorge.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

# Use TF to save the graph model instead of Keras save model to load it in Golang
builder = tf.saved_model.builder.SavedModelBuilder(modelName)
# Tag the model, required for Go
builder.add_meta_graph_and_variables(sess, ["myTag"])
builder.save()
sess.close()
