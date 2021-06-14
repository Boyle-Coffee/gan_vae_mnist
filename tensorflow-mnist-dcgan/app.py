import traceback
import os
from flask import Flask, request, jsonify
import json
import tensorflow as tf

from dcgan_model import DCGAN
from gan_utils import generate_image_gan
from gan_utils import numpy2base, base2numpy, check_params

app = Flask(__name__)

"""
加载模型
"""
# 加载 DCGAN
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True

gan_out_dir = "./gan_out"
gan_checkpoint_dir = os.path.join(gan_out_dir, "checkpoint")

gan_sess = tf.Session(config=run_config)
gan_batch_size = 1
dcgan = DCGAN(
    gan_sess,
    input_width=28,
    input_height=28,
    output_width=28,
    output_height=28,
    batch_size=gan_batch_size,
    sample_num=1,
    y_dim=10,
    z_dim=100,
    dataset_name="mnist",
    crop=False,
    checkpoint_dir=gan_checkpoint_dir,
    sample_dir="samples",
    data_dir="./data",
    out_dir=gan_out_dir,
    max_to_keep=1)

load_success, load_counter = dcgan.load(gan_checkpoint_dir)
if not load_success:
    raise Exception("Checkpoint not found in " + "checkpoint")

"""
定义路由
"""


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/ganMnist', methods=['POST'])
def gan_mnist():
    try:
        post_json = request.get_json()
        print("POST: ", post_json)
        check_params(["label"], post_json)
        label = int(post_json["label"])
        samples = generate_image_gan(gan_sess, dcgan, label, gan_batch_size)
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
    app.run(host='0.0.0.0', debug=True, port=5000)
