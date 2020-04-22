# -*- coding: utf-8 -*-
import tensorflow as tf
from models.inception.Inception_resnet_V1 import InceptionResNet_V1
from dataset_process import input_pipeline
from configuration import cfg

train_tfrecord = cfg.MODEL.TFRCORD_TRAIN
valid_tfrecord = cfg.MODEL.TFRCORD_VALID
IMAGE_HEIGHT   = cfg.MODEL.INP_HEIGHT
IMAGE_WIDTH    = cfg.MODEL.INP_WIDTH
CHANNELS       = cfg.MODEL.CHANELS
EPOCHS         = cfg.MODEF.EPOCHS
save_model_dir = cfg.MODEL.SAVE_DIR


if __name__ == '__main__':
    # GPU setting
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get dataset
    train_dataset = input_pipeline(train_tfrecord)
    valid_dataset = input_pipeline(valid_tfrecord)

    # create model
    model = InceptionResNet_V1()
    model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    model.summary()
    # tf.keras.utils.plot_model(model=model, to_file='model.png')

    # define loss and optimizer
    loss_object = tf.keras.losses.sparse_categorical_crossentropy
    optimizer = tf.keras.optimizers.Adagrad()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    summary_writer = tf.summary.create_file_writer(logdir='tensorboard')  # 实例化记录器
    tf.summary.trace_on(profiler=True)

    @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_pred=predictions, y_true=label_batch)


    @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # running train step
    for epoch in range(EPOCHS):
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step:{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                    EPOCHS,
                                                                                    step,
                                                                                    train_loss.result().numpy(),
                                                                                    train_accuracy.result().numpy()))
        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_accuracy.result().numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  valid_accuracy.result().numpy()))
        with summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result().numpy(), step=step)

        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        if (epoch+1) % cfg.MODEL.SAVE_FREQ == 0:
            model.save_weights(filepath=save_model_dir + 'epoch-{}'.format(epoch), save_format='tf')

    with summary_writer.as_default():
        tf.summary.trace_export(name='model_trace', step=0, profiler_outdir='tensorboard')

    # save the whole model
    tf.saved_model.save(model, save_model_dir)
