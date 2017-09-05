import os.path
import tensorflow as tf
import helper
import warnings
import time
import numpy as np
import scipy.misc
import cv2
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip

def main():

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['vgg_fcn'], './outputmodel')
        logits = sess.graph.get_tensor_by_name('logits:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        image_pl = sess.graph.get_tensor_by_name('image_input:0')

        def process_image(image):
            nonlocal sess, logits

            # must be multiples of 32 for this model to work correctly
            # 1280x720 -> 576x320
            image = cv2.resize(image, (576, 320))
            image_shape = image.shape
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})

            # im_softmax contains two columns - 0: road, 1: non-road
            # Create mask from road column
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])

            # boolean mask with 3 dims
            mask = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)

            return np.clip(image + (mask.astype(np.uint16)*[128,0,128]), 0, 255).astype(image.dtype)

        # OPTIONAL: Apply the trained model to a video
        clip1 = VideoFileClip("./data/project_video.mp4")
        modifiedClip = clip1.fl_image(process_image)
        modifiedClip.write_videofile('./data/project_video_annotated.mp4', audio=False)


if __name__ == '__main__':
    main()
