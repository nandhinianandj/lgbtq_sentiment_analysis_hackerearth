# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : tfhub_ex.py
#
#* Purpose :
#
#* Creation Date : 23-08-2019
#
#* Last Modified : Friday 23 August 2019 06:23:54 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

import tensorflow as tf
import tensorflow_hub as hub

tf.enable_eager_execution()

module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
embed = hub.KerasLayer(module_url)
embeddings = embed(["A long sentence.", "single-word",
                    "http://example.com"])
print(embeddings.shape)  #(3,128)

