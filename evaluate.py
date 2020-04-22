# -*- coding: utf-8 -*-
import tensorflow as tf
from configuration import cfg
from dataset_process import input_pipeline

test_tfrecord = cfg.MODEL.MODEL.SAVE_DIR
save_model_dir = cfg.MODEL.TFRCORD_TEST

if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    test_dataset = input_pipeline(test_tfrecord)
    # load the model
    # model = get_model()
    # model.load_weights(filepath=save_model_dir)
    model = tf.saved_model.load(save_model_dir)

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))
