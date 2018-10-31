With Tensorflow installed for Go and all the project dependencies:

```
$ go run main.go https://img.buzzfeed.com/buzzfeed-static/static/2014-10/28/23/campaign_images/webdr02/33-awesome-facts-about-dogs-2-1561-1414553390-14_dblbig.jpg
2018-10-31 01:29:36.266384: I tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: myModel
2018-10-31 01:29:36.281266: I tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { myTag }
2018-10-31 01:29:36.305518: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-10-31 01:29:36.377970: I tensorflow/cc/saved_model/loader.cc:162] Restoring SavedModel bundle.
2018-10-31 01:29:36.508289: I tensorflow/cc/saved_model/loader.cc:138] Running MainOp with key legacy_init_op on SavedModel bundle.
2018-10-31 01:29:36.508322: I tensorflow/cc/saved_model/loader.cc:259] SavedModel load for tags { myTag }; Status: success. Took 241948 microseconds.
Results:
Pembroke (0.404670)
Shetland_sheepdog (0.168692)
Cardigan (0.041638)
collie (0.031163)
papillon (0.018812)
Blenheim_spaniel (0.018086)
Pomeranian (0.011312)
Samoyed (0.007005)
Bernese_mountain_dog (0.006889)
chow (0.004397)
```
