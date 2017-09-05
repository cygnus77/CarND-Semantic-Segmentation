import os.path
import tensorflow as tf
import helper
import warnings
import time
import numpy as np
import cv2
from distutils.version import LooseVersion
import project_tests as tests
import shutil
from glob import glob
import re

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

regularizer = tf.contrib.layers.l2_regularizer(1e-3)

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    return (sess.graph.get_tensor_by_name(vgg_input_tensor_name),
      sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name),
      sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name),
      sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name),
      sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name))

tests.test_load_vgg(load_vgg, tf)

def print_tensor_shape(op, msg, tensor):
  """
  For debugging: returns print operation to be inserted in graph
  """
  return tf.Print(op, data=[tf.shape(tensor)], message=msg, first_n=5, summarize=5)

def conv_transpose(layer, name, filters, strides=(2,2)):
  """
  Return a transpose convolution operation that upscales previous layer by factor of 2
  Uses a 4x4 kernel, 2x2 stride and globally defined regularizer
  Padding SAME is essential for maintaining correct size
  """
  layer = tf.layers.conv2d_transpose(layer, filters=filters, kernel_size=(4,4), strides=strides, padding="SAME", name=name, kernel_regularizer=regularizer)
  # Tracing: print shape of tensor passed into graph
  #layer = print_tensor_shape(layer, name, layer)
  return layer

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
  """
  Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
  :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
  :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
  :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
  :param num_classes: Number of classes to classify
  :return: The Tensor for the last layer of output
  """
  #
  # Tracing: print shapes of layers being passed in from VGG
  #
  #vgg_layer7_out = print_tensor_shape(vgg_layer7_out, "shape vgg_layer3_out: ", vgg_layer3_out)
  #vgg_layer7_out = print_tensor_shape(vgg_layer7_out, "shape vgg_layer4_out: ", vgg_layer4_out)
  #vgg_layer7_out = print_tensor_shape(vgg_layer7_out, "shape vgg_layer7_out: ", vgg_layer7_out)

  # 1x1 convolution to connect decoder to encoder part of FCN
  layer8 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='SAME', kernel_regularizer=regularizer)
  layer9 = conv_transpose(layer8, "layer9", filters=512)
  layer9 = tf.add(layer9, vgg_layer4_out)
  layer10 = conv_transpose(layer9, "layer10", filters=256)
  layer10 = tf.add(layer10, vgg_layer3_out)
  layer11 = conv_transpose(layer10, "layer11", filters=num_classes, strides=(8,8))
  return layer11

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
  """
  Build the TensorFLow loss and optimizer operations.
  :param nn_last_layer: TF Tensor of the last layer in the neural network
  :param correct_label: TF Placeholder for the correct label image
  :param learning_rate: TF Placeholder for the learning rate
  :param num_classes: Number of classes to classify
  :return: Tuple of (logits, train_op, cross_entropy_loss)
  """
  # operation to produce logits as output of FCN
  logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')

  # loss function
  cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

  # Optimize with gradient descent or Adam optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # Training operation
  train_op = optimizer.minimize(cross_entropy_loss)

  return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, keep_prob_value=1.0):
  """
  Train neural network and print out the loss during training.
  :param sess: TF Session
  :param epochs: Number of epochs
  :param batch_size: Batch size
  :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
  :param train_op: TF Operation to train the neural network
  :param cross_entropy_loss: TF Tensor for the amount of loss
  :param input_image: TF Placeholder for input images
  :param correct_label: TF Placeholder for label images
  :param keep_prob: TF Placeholder for dropout keep probability
  :param learning_rate: TF Placeholder for learning rate
  """

  # repeat learning step for specified number of epochs
  for epoch in range(epochs):

    print("Starting epoch: %d" % (epoch))
    start_time = time.time()

    step = 1
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    for X_train, y_train in get_batches_fn(batch_size):
    
      feed_dict = {
          input_image: X_train,
          correct_label: y_train,
          keep_prob: keep_prob_value
      }

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      # Print status to stdout.
      print('   Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
      step += 1

tests.test_train_nn(train_nn)

def readImage(img_fname):
  # retrive image data from disk
  # returns image in RGB format
  img_path = os.path.join(DATA_DIR,img_fname.strip())
  img = cv2.imread(img_path)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def addShadow(img):
  # adds a random shadow to a given image
  # pick a random shadow coloration
  shadow_shade = np.random.randint(60,120)
  # convert image to YUV space to get the luma (brightness) channel
  y,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))
  y = y.astype(np.int32)
  # create mask image with same shape as input image
  #mask = np.zeros(y.shape, dtype=np.int32)
  # compute a random line in slope, intercept form
  # random x1,x2 values (y1=0, y2=height)
  x1 = np.random.uniform() * y.shape[1]
  x2 = np.random.uniform() * y.shape[1]
  slope = float(y.shape[0]) / (x2 - x1)
  intercept = -(slope * x1)
  # assign pixels of mask below line
  for j in range(y.shape[0]):
      for i in range(y.shape[1]):
          if j > (i*slope)+intercept:
              y[j,i] -= shadow_shade
  # apply mask
  #y += mask
  # ensure values are within uint8 range to avoid artifacts
  y = np.clip(y, 0,255).astype(np.uint8)
  # convert back to RGB
  return cv2.cvtColor(cv2.merge((y,u,v)), cv2.COLOR_YCrCb2RGB)

def adjustBrightness(img, m):
  # adjust brightness of given image (img) by multiplyling V (brightness)
  h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
  v = np.clip(v * m, 0, 255).astype(np.uint8)
  return cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2RGB)

def augment_dataset(data_folder):
  """
  Creates transformation for each image on disk
  - one copy with arbitrary shadow
  - 3 copies at varying brightness - .25, .5 and .75
  - copies have an "aug" inserted in the name like: um_aug00124.png
  - copy label file with matching filenames
  """
  image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
  label_paths = {
    re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
    for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
  background_color = np.array([255, 0, 0])

  for image_file in image_paths:
    print("Augmentation: processing: {0}".format(image_file) )
    gt_image_file = label_paths[os.path.basename(image_file)]
    img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    aug = [addShadow(img), 
      adjustBrightness(img, .75),
      adjustBrightness(img, .5),
      adjustBrightness(img, .25)]
    for idx, augimg in enumerate(aug):
      cv2.imwrite("{0}_aug{2}{1}".format(*(list(os.path.splitext(image_file))+[idx])), augimg)
      shutil.copy(gt_image_file, "{0}_aug{2}{1}".format(*(list(os.path.splitext(gt_image_file))+[idx])))

def run():
  num_classes = 2
  image_shape = (160, 576)
  data_dir = './data'
  runs_dir = './runs'
  tests.test_for_kitti_dataset(data_dir)
  learning_rate = 0.0001
  batch_size = 10
  epochs = 10

  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)

  # Path to vgg model
  vgg_path = os.path.join(data_dir, 'vgg')
  # Create function to get batches
  get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

  # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
  # You'll need a GPU with at least 10 teraFLOPS to train on.
  #  https://www.cityscapes-dataset.com/

  with tf.Session() as sess:
    # Augmentation - takes a long time, better run as separate script
    #augment_dataset(os.path.join(data_dir, 'data_road/training'))

    # Placeholder for labels - batch, width, height, num_classes
    labels_placeholder = tf.placeholder(tf.int32, shape=(None, None, None, num_classes))

    # Build NN using load_vgg, layers, and optimize function
    input_image_placeholder, keep_prob_placeholder, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
    last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

    # operations    
    logits, train_op, cross_entropy_loss = optimize(last_layer, labels_placeholder, learning_rate, num_classes)

    # initialize to variables to random values
    sess.run(tf.global_variables_initializer())

    # Train NN using the train_nn function
    train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image_placeholder,
         labels_placeholder, keep_prob_placeholder, learning_rate, keep_prob_value=0.6)

    # Save model to disk
    builder = tf.saved_model.builder.SavedModelBuilder('./outputmodel')
    builder.add_meta_graph_and_variables(sess,["vgg_fcn"])
    builder.save()

    # Save inference data using helper.save_inference_samples
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_placeholder, input_image_placeholder)

    def process_video_clip(image):
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

      return image + (mask.astype(image.dtype)*[128,0,0])

    # OPTIONAL: Apply the trained model to a video
    clip1 = VideoFileClip(os.path.join(data_dir, 'project_video.mp4'))
    modifiedClip = clip1.fl_image(process_video_clip)
    modifiedClip.write_videofile(os.path.join(data_dir, 'project_video_annotated.mp4'), audio=False)

if __name__ == '__main__':
    run()
