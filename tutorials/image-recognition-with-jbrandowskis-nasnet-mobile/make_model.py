import keras
from keras.applications.nasnet import NASNetMobile
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
from keras import backend as K

sess = tf.Session()
K.set_session(sess)

model = NASNetMobile(weights="09a9e3fd-ebf0-46d4-bd5d-8be69d80cf44_NASNet-mobile.h5")
img = image.load_img('cat.jpg', target_size=(224,224)) #note the input size
img_arr = np.expand_dims(image.img_to_array(img), axis=0)
x = preprocess_input(img_arr)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

print('input layer: ', model.input)
print('output layer: ', model.output)

# Use TF to save the graph model instead of Keras save model to load it in Golang
builder = tf.saved_model.builder.SavedModelBuilder("myModel")
# Tag the model, required for Go
builder.add_meta_graph_and_variables(sess, ["myTag"])
builder.save()
sess.close()
