import os.path
import tensorflow as tf
import helper
import warnings
import time
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

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
    
    return sess.graph.get_tensor_by_name(vgg_input_tensor_name), \
        sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name), sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name), \
        sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name), sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # layer8 1x1x4096 -> 14x14x512
    layer8 = tf.layers.conv2d_transpose(vgg_layer7_out, filters=512, kernel_size=(2,2), strides=(14,14), padding="SAME")
    #print(layer8.get_shape().as_list())

    # layer9 14x14x512 -> 28x28x512
    layer9a = tf.layers.conv2d_transpose(layer8, filters=512, kernel_size=(2,2), strides=(2,2), padding="SAME")
    #print(layer9a.get_shape().as_list())

    layer9 = tf.add(layer9a, vgg_layer4_out)

    # layer10 28x28x512 -> 56x56x256
    layer10a = tf.layers.conv2d_transpose(layer9, filters=256, kernel_size=(2,2), strides=(2,2), padding="SAME")
    #print(layer10a.get_shape().as_list())

    layer10 = tf.add(layer10a, vgg_layer3_out)

    # layer11 56x56x256 -> 112x112x128
    layer11 = tf.layers.conv2d_transpose(layer10, filters=128, kernel_size=(2,2), strides=(2,2), padding="SAME")
    #print(layer11.get_shape().as_list())

    # layer11 112x112x128 -> 224,224,64
    layer12 = tf.layers.conv2d_transpose(layer11, filters=64, kernel_size=(2,2), strides=(2,2), padding="SAME")
    #print(layer12.get_shape().as_list())

    # layer11 224,224x64 -> 224,224,num_classes
    layer13 = tf.layers.conv2d_transpose(layer12, filters=num_classes, kernel_size=(2,2), strides=(1,1), padding="SAME")
    #print(layer13.get_shape().as_list())

    return layer13
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
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
    epoch = 1
    while epoch < epochs:

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
        
        epoch += 1


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    learning_rate = 0.0001
    batch_size = 50
    epochs = 15

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
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, image_shape[0], image_shape[1], num_classes))
 
        # Build NN using load_vgg, layers, and optimize function
        input_image_placeholder, keep_prob_placeholder, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        logits, train_op, cross_entropy_loss = optimize(last_layer, labels_placeholder, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image_placeholder,
             labels_placeholder, keep_prob_placeholder, learning_rate, keep_prob_value=0.75)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
