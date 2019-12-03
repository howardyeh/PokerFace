import tensorflow as tf
from tensorflow.contrib.slim import nets
from preprocess import Preprocess 
slim = tf.contrib.slim
 

# Parameter setup 
training_epoch = 100
total_data_num = 300
train_data_num = total_data_num * 0.7
valid_data_num = total_data_num - train_data_num
batch_size = 10
steps_per_epoch = train_data_num / batch_size
train_total_step = training_epoch * (steps_per_epoch)

# Training schedule
LONG_SCHEDULE = {
    'step_values': [1050, 1470, 1680, 1890], # 50ep, 70ep, 80ep, 90ep 
    'learning_rates': [0.001, 0.0005, 0.00025, 0.000125, 0.0000625],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 2100,
}
        
class Model(object):
    def __init__(self, is_training):
        self._is_training = is_training
        self.global_step = 0
        self.epoch = 0
    
    def predict(self, image):
        # use ResNet_v1 as model
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, endpoints = nets.resnet_v1.resnet_v1_50(image, 
                                                        num_classes=None,
                                                        is_training=self._is_training)
        net = tf.squeeze(net, axis=[1, 2])

        # The output will be a single number indicating possibility
        logits = slim.fully_connected(net, num_outputs= 1,
                                      activation_fn=tf.nn.sigmoid, scope='Predict')
        return logits

    
    def loss(self, prediction, expectation, groundtruth):
        # The weighting between expectation and prediction need to be tuned
        logits = 0.5 * prediction + 0.5 * expectation
        prob_loss = tf.nn.l2_loss(groundtruth - logits)
        slim.losses.add_loss(l2_loss)
        loss = slim.losses.get_total_loss()
        loss_dict = {'prob_loss': prob_loss,'loss': loss}
        return loss_dict

    def error_percent(self, prediction, groundtruth):
        error = abs(groundtruth - prediction)
        return error

    def train(self, checkpoints=None):
        
        image = tf.placeholder(tf.float32, [None, 224, 224, 3])
        image_tb = tf.summary.image("image", image, max_outputs=2)
        expectation = tf.placeholder(tf.float32, [None, 1])
        groundtruth = tf.placeholder(tf.float32, [None, 1])


        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])
        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])


        prediction = self.predict(image)
        total_loss = self.loss(prediction, expectation, groundtruth)
        loss_tb = tf.summary.scalar('loss', total_loss['loss'])


        # for recording validation loss
        valid_loss_placeholder = tf.placeholder(tf.float32, [1])
        valid_loss_tb = tf.summary.scalar('valid_loss', valid_loss_placeholder)

        train_summary = [image_tb, loss_tb]
        valid_summary = [valid_loss_tb]


        saver = tf.train.Saver()


        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)


        dataloader = Preprocess("./../text.txt", "./../image/", batch_size)

        with tf.Session() as sess:
            # merged = tf.summary.merge_all()
            train_merged = tf.summary.merge(train_summary)
            valid_merged = tf.summary.merge(valid_summary)
            writer = tf.summary.FileWriter("logs/", sess.graph)
            sess.run(tf.global_variables_initializer())

            if checkpoints:
                saver.restore(sess, "logs/weighting.ckpt")
            

            for step in range( train_total_step ):
                self.global_step += 1
                if step % steps_per_epoch == 0:
                    self.epoch += 1

                image_batch, expectation_batch, gt_batch = dataloader.create_batch(step, 'train')
                _, loss_ = sess.run([train_op, total_loss['loss']], feed_dict = {image: image_batch, expectation: expectation_batch, groundtruth: gt_batch})
                

                # Print current loss every 10 steps
                if step % 10 == 0:
                    print('Step:', step, '| train loss: %.4f' % loss_)


                # Validate every 10 epoch
                if self.epoch % 10 == 0:
                    validation_total_loss = 0
                    for val_step in range(valid_data_num / batch_size):
                        val_image_batch, val_expectation_batch, val_gt_batch = dataloader.create_batch(val_step, 'validation')
                        valid_loss = sess.run(total_loss['prob_loss'], feed_dict = {image: val_image_batch, expectation: val_expectation_batch, groundtruth: val_gt_batch})
                        validation_total_loss += valid_loss
                    validation_total_loss /= (valid_data_num / batch_size)
                    print('Save validation loss to Tensorboard...')
                    val_result = sess.run(valid_merged, feed_dict = {valid_loss_placeholder: validation_total_loss})
                    writer.add_summary(val_result, step)
                    print('Finish saving.')


                # Save result to tensorboard every 50 steps
                if step % 50 == 0:
                    print('Save to Tensorboard...')
                    result = sess.run(train_merged, feed_dict = {image: image_batch, expectation: expectation_batch, groundtruth: gt_batch})
                    writer.add_summary(result, step)
                    print('Finish saving.')

            save_path = saver.save(sess, "weight/weighting.ckpt")
            print("Save to path: ", save_path)

        