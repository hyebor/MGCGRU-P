import os
import sys
import time
import random
import shutil
import numpy as np
from os.path import join, exists
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import yaml


from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import neg_log_gauss, masked_mae_loss

from model.prnn_model import PRNNModel


class PRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, adj_mx, args, inference=False, pretrained_dir=None, **kwargs):

        SEED = args.trial_id
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        random.seed(SEED)

        self._dataset_name = args.dataset
        self.pretrained_dir = pretrained_dir

        kwargs["data"]["dataset_dir"] = join(args.data_dir, args.dataset)
        kwargs["data"]["dataset_name"] = args.dataset
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self._rnn_type = kwargs.get("model")["rnn_type"]
        self._cost = kwargs.get("train")["cost"]

        # original log_dir setting
        # self._log_dir = self._get_log_dir(kwargs)
        # new log_dir setting:
        self._log_dir = self._get_log_dir_by_trial(args, inference=inference, kwargs=kwargs)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        # Data preparation
        self._data = utils.load_dataset(**self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                self._train_model = PRNNModel(is_training=True, scaler=scaler,
                                              batch_size=self._data_kwargs.get('batch_size', 64),
                                              adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._test_model = PRNNModel(is_training=False, scaler=scaler,
                                             batch_size=self._data_kwargs.get('test_batch_size', 64),
                                             adj_mx=adj_mx, **self._model_kwargs)

        # learning
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        cost = self._train_kwargs.get('cost', 'mae').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr, )
        elif optimizer_name == 'amsgrad':
            optimizer = AMSGrad(self._lr, epsilon=epsilon)

        # Calculate loss
        output_dim = self._model_kwargs.get('output_dim', 1)
        preds = self._train_model.outputs
        preds_samples = self._train_model.outputs_samples
        preds_mu = self._train_model.outputs_mu
        preds_sigma = self._train_model.outputs_sigma
        labels = self._train_model.labels[..., :output_dim]

        null_val = 0.
        if cost == 'mae':
            self._loss_fn = masked_mae_loss(scaler, null_val)
            self._train_loss = self._loss_fn(preds=preds, labels=labels)
        else:
            self._loss_fn = neg_log_gauss()
            self._train_loss = self._loss_fn(preds_mu=preds_mu, preds_sigma=preds_sigma, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._epoch = 0
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    def _get_log_dir_by_trial(self, args, inference, kwargs):
        # log_dir = kwargs['train'].get('log_dir')
        if not inference:
            self.pretrained_dir = None
        if self.pretrained_dir is None:
            if not exists(args.ckpt_dir):
                os.mkdir(args.ckpt_dir)
            log_dir = join(args.ckpt_dir, args.dataset)
            if not exists(log_dir):
                os.mkdir(log_dir)

            log_dir = join(log_dir, self._rnn_type + "_" + self._cost)
            if not exists(log_dir):
                os.mkdir(log_dir)

            log_dir = join(log_dir, "trial_{}".format(args.trial_id))

            if inference:
                log_dir = log_dir + "_infer"
                if not exists(log_dir):
                    os.mkdir(log_dir)
            else:
                if exists(log_dir):
                    shutil.rmtree(log_dir)
                os.mkdir(log_dir)
        else:
            log_dir = self.pretrained_dir
            log_dir = log_dir + "_infer"
            if not exists(log_dir):
                os.mkdir(log_dir)

        kwargs['train']['log_dir'] = log_dir

        return log_dir

    def run_epoch_generator(self, sess, model, data_generator, return_output=False, training=False, writer=None):
        losses = []
        maes = []
        outputs = []
        outputs_samples = []
        output_dim = self._model_kwargs.get('output_dim', 1)
        preds = model.outputs
        preds_samples = model.outputs_samples
        preds_mu = model.outputs_mu
        preds_sigma = model.outputs_sigma
        labels = model.labels[..., :output_dim]
        cost = self._train_kwargs.get('cost', 'mae').lower()
        scaler = self._data['scaler']
        null_val = 0.
        if cost == 'mae':
            loss = self._loss_fn(preds=preds, labels=labels)
            mae = loss
        else:
            loss = self._loss_fn(preds_mu=preds_mu, preds_sigma=preds_sigma, labels=labels)
            mae_fun = masked_mae_loss(scaler, null_val)
            mae = mae_fun(preds=preds, labels=labels)

        fetches = {
            'loss': loss,
            'mae': mae,
            'global_step': tf.train.get_or_create_global_step()
        }
        if training:
            fetches.update({
                'train_op': self._train_op
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs,
                'outputs_samples': model.outputs_samples
            })

        for _, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            maes.append(vals['mae'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])
                outputs_samples.append(vals['outputs_samples'])

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }
        if return_output:
            results['outputs'] = outputs
            results['outputs_samples'] = outputs_samples
        return results

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=10, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = train_kwargs.get('model_filename')
        # print(model_filename)
        if model_filename is not None:
            saver.restore(sess, model_filename)
            self._epoch = epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
        self._logger.info('Start training {} on {}...'.format(self._rnn_type, self._dataset_name))

        while self._epoch <= epochs:
            # Learning rate schedule.
            # new_lr = max(min_learning_rate, base_lr * ((lr_decay_ratio/9) ** np.sum(self._epoch >= np.array(steps))))
            new_lr = max(min_learning_rate, base_lr * ((lr_decay_ratio) ** (self._epoch / 1000)))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True,
                                                     writer=self._writer)
            train_loss, train_mae = train_results['loss'], train_results['mae']
            if train_loss > 1e5:
                self._logger.warning('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self.run_epoch_generator(sess, self._test_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            val_loss, val_mae = np.asscalar(val_results['loss']), np.asscalar(val_results['mae'])

            utils.add_simple_summary(self._writer,
                               ['loss/train_loss', 'metric/train_mae', 'loss/val_loss', 'metric/val_mae'],
                               [train_loss, train_mae, val_loss, val_mae], global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_mae: {:.4f} lr:{:.6f} {:.1f}s'.format(
                self._epoch, epochs, global_step, train_loss, val_mae, new_lr, (end_time - start_time))
            self._logger.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.evaluate(sess)
            if val_mae <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                self._logger.info(
                    'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_mae, model_filename))
                min_val_loss = val_mae
            else:
                wait += 1
                if wait > patience * 2:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def evaluate(self, sess, **kwargs):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, self._test_model,
                                                self._data['test_loader'].get_iterator(),
                                                return_output=True,
                                                training=False)

        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds, y_preds_samples = test_results['loss'], test_results['outputs'], test_results[
            'outputs_samples']
        utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)

        y_preds = np.concatenate(y_preds, axis=0)
        y_preds_samples = np.concatenate(y_preds_samples, axis=0)
        scaler = self._data['scaler']
        predictions = []
        predictions_samples = []
        y_truths = []
        for horizon_i in range(self._data['y_test'].shape[1]):
            y_truth = scaler.inverse_transform(self._data['y_test'][:, horizon_i, :, 0])
            y_truths.append(y_truth)

            y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0], horizon_i, :, 0])
            predictions.append(y_pred)

            y_pred_samples = scaler.inverse_transform(y_preds_samples[:y_truth.shape[0], horizon_i, :, :])
            predictions_samples.append(y_pred_samples)

            mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0)
            mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
            rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
            # self._logger.info(
            #     "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
            #         horizon_i + 1, mae, mape, rmse
            #     )
            # )
            utils.add_simple_summary(self._writer,
                               ['%s_%d' % (item, horizon_i + 1) for item in
                                ['metric/rmse', 'metric/mape', 'metric/mae']],
                               [rmse, mape, mae],
                               global_step=global_step)
        outputs = {
            'predictions': predictions,
            'groundtruth': y_truths,
            'predictions_samples': predictions_samples,
        }
        return outputs

    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix, global_step=global_step,
                                                             write_meta_graph=False)
        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']


