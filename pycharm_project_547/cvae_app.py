import traceback
import os
from flask import Flask, request, jsonify
import json
import tensorflow as tf

import vae_model
from app_utils import generate_image_cvae
from app_utils import numpy2base, check_params


app = Flask(__name__)

"""
加载模型
"""
# 加载 CVAE
run_config = tf.ConfigProto()
cvae_out_dir = "./out/cvae_ckpt"
cvae_checkpoint_dir = os.path.join(cvae_out_dir, "checkpoint")
dim_img = 784
dim_z = 2
num_labels = 10
n_hidden = 500

x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
y = tf.placeholder(tf.float32, shape=[None, 10], name='target_labels')

keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
fack_id_in = tf.placeholder(tf.float32, shape=[None, 10], name='latent_variable') # condition

# network architecture
x_, z, loss, neg_marginal_likelihood, KL_divergence, loss_sum, neg_marginal_likelihood_sum, KL_divergence_sum = \
    vae_model.autoencoder(x_hat, x, y, dim_img, dim_z, n_hidden, keep_prob)

cvae_sess = tf.Session(config=run_config)
cvae_sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})
cvae_saver = tf.train.Saver(max_to_keep=1)
load_success, checkpoint_counter = vae_model.load(cvae_sess, cvae_checkpoint_dir, cvae_saver)
decoded = vae_model.decoder(z_in, fack_id_in, dim_img, n_hidden)
if not load_success:
    raise Exception("Checkpoint not found in " + "checkpoint")


"""
定义路由
"""


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/cvaeMnist', methods=['POST'])
def gan_mnist():
    try:
        post_json = request.get_json()
        print("POST: ", post_json)
        check_params(["label"], post_json)
        label = int(post_json["label"])
        samples = generate_image_cvae(cvae_sess, decoded, label, 1, z_in, fack_id_in, keep_prob)
        if samples is None:
            raise ValueError("An error happens when generating the image")
        sample255 = samples[0]*255
        ret_base64 = numpy2base(sample255)

        return jsonify({
            'isSuccess': True,
            'message': None,
            'data': ret_base64
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            'isSuccess': False,
            'message': e.args,
            'data': None
        })


if __name__ == '__main__':
    app.run(debug=True, port=5001)
