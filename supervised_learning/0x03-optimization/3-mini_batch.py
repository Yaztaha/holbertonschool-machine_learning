#!/usr/bin/env python3
""" Mini-batch training """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ training function for mini batch GD """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        vars = ['x', 'y', 'accuracy', 'loss', 'train_op', 'y_pred']
        for var in vars:
            globals()[var] = tf.get_collection(var)[0]

        for epoch in range(epochs + 1):

            loss_t = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_t = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            loss_v = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_v = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(loss_v))
            print("\tValidation Accuracy: {}".format(acc_v))

            if epoch < epochs:

                X_shuff, Y_shuff = shuffle_data(X_train, Y_train)

                batches_float = X_train.shape[0] / batch_size
                batches_int = int(X_train.shape[0] / batch_size)

                step = 0

                for i in range(0, batches_int + 1):
                    step += 1

                    if i == batches_int:
                        if batches_float > batches_int:
                            X_batch = X_shuff[i * batch_size:]
                            Y_batch = Y_shuff[i * batch_size:]
                        else:
                            break
                    else:
                        X_batch = X_shuff[i * batch_size: (i + 1) * batch_size]
                        Y_batch = Y_shuff[i * batch_size: (i + 1) * batch_size]

                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if step % 100 == 0:
                        loss_b = sess.run(loss, feed_dict={
                            x: X_batch, y: Y_batch})
                        acc_b = sess.run(accuracy, feed_dict={
                            x: X_batch, y: Y_batch})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(loss_b))
                        print("\t\tAccuracy: {}".format(acc_b))

        save_path = loader.save(sess, save_path)
    return save_path
