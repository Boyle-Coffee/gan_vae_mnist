import base64
import cv2
import numpy as np


def check_params(req_params, post_json):
    post_params = post_json.keys()
    for param in req_params:
        if param not in post_params:
            raise ValueError("missing param '{param}'")
    for param in post_params:
        if param not in req_params:
            raise ValueError("{param} is not a request param")


def base2numpy(image_base64):
    image_data = base64.b64decode(image_base64)
    nparr = np.fromstring(image_data, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image_np


def numpy2base(image_np):
    retval, buffer = cv2.imencode('.jpg', image_np)
    pic_str = base64.b64encode(buffer)
    image_base64 = pic_str.decode()

    return image_base64


def generate_image_gan(sess, dcgan, y, batch_size):
    try:
        z_sample = np.random.uniform(-1, 1, size=(batch_size, dcgan.z_dim))
        y_one_hot = np.zeros((batch_size, 10))
        y_one_hot[np.arange(batch_size), y] = 1
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
        return samples
    except IndexError as e:
        return None


def generate_image_cvae(sess, decoded, y, batch_size, z_in, fack_id_in, keep_prob, z_dim=2):
    try:
        z_sample = np.random.uniform(-1, 1, size=(batch_size, z_dim))
        y_one_hot = np.zeros((batch_size, 10))
        y_one_hot[np.arange(batch_size), y] = 1
        samples = sess.run(decoded, feed_dict={z_in: z_sample, fack_id_in:  y_one_hot, keep_prob : 1})
        samples_ = samples.reshape(batch_size, 28, 28)
        return samples_
    except IndexError as e:
        return None
