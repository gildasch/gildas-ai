import keras
from keras import backend as K
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import json
import os
import sys

if not os.path.isfile("Xception.h5"):
    os.system("curl -L -o Xception.h5 http://modeldepot.io/assets/uploads/models/models/0fc18241-ecc1-4c84-97f8-efc4e4f9dad8_Xception.h5")

if len(sys.argv[1:]) > 0:
    modelName = sys.argv[1]
else:
    modelName = "xception"

sess = tf.Session()
K.set_session(sess)

model = Xception(weights="Xception.h5")
sess.run(tf.global_variables_initializer())

img = image.load_img('gorge.jpg', target_size=(299,299)) #note the input size
img_arr = np.expand_dims(image.img_to_array(img), axis=0)
x = preprocess_input(img_arr)
preds = model.predict(x)

# print('Predicted:', decode_predictions(preds, top=3)[0])
# print([n.name for n in tf.get_default_graph().as_graph_def().node if n.name.startswith("predictions")])

# Use TF to save the graph model instead of Keras save model to load it in Golang
builder = tf.saved_model.builder.SavedModelBuilder(modelName)
# Tag the model, required for Go
builder.add_meta_graph_and_variables(sess, ["myTag"])
builder.save()
sess.close()
