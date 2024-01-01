# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from os.path import join, exists
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model.prnn_supervisor import PRNNSupervisor

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="split_bj", help="name of datasets")

    parser.add_argument("--data_dir", default="../input/data", help="the dir storing dataset")

    parser.add_argument("--ckpt_dir", default="/output/ckpt", help="the dir to store checkpoints")
    parser.add_argument("--graph_dir", default=None,
                        help="the dir storing the graph information; if None, will be set as the '{data_dir}/sensor_graph'.")
    parser.add_argument('--config_dir', default=None, type=str,
                        help="The dir storing the detailed configuration of model (including hyperparameters); if None, will be set as the '{data_dir}/model_config'.")
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id to use; by default using 0')
    parser.add_argument('--use_cpu_only', default=False, action="store_true",
                        help='Add this if want to train in cpu only')
    parser.add_argument("--trial_id", type=int, default=0,
                        help="id of the trial. Used as the random seed in multiple trials training")
    parser.add_argument("--rnn_type", type=str, default='dcgru',
                        help="The type of rnn architecture of rnn_flow; if None, following the setting in the config file")
    parser.add_argument("--cost", type=str, default='mae',
                        help="The type of loss function (e.g., [mae], [nll]); if None, following the setting of the config file")

    args = parser.parse_args(args=[])

    if args.graph_dir is None:
        args.graph_dir = join(args.data_dir, "sensor_graph")

    if args.config_dir is None:
        args.config_dir = join(args.data_dir, "model_config")

    if args.use_cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.graph_dir is None:
        args.graph_dir = join(args.data_dir, "sensor_graph")

    args.dataset = args.dataset.lower()


def load_adj():
    jck_adj = pd.read_csv('../input/data/jck_adj.csv', header=None, index_col=None)  # 邻接矩阵
    jck_adj = np.mat(jck_adj)

    simi_adj = pd.read_csv('../input/data/simi_adj.csv', header=None, index_col=None)  # 历史流量
    simi_adj = np.mat(simi_adj)

    dis_adj = pd.read_csv('../input/data/dis_adj.csv', header=None, index_col=None)  # 距离
    dis_adj = np.mat(dis_adj)

    cont_adj = pd.read_csv('../input/data/cont_adj.csv', header=None, index_col=None)  # 合乘数
    cont_adj = np.mat(cont_adj)

    func_adj = pd.read_csv('../input/data/func_adj.csv', header=None, index_col=None)  # 功能相似性
    func_adj = np.mat(func_adj)

    stage_adj = pd.read_csv('../input/data/stage_adj.csv', header=None, index_col=None)  # 合乘阶段
    stage_adj = np.mat(stage_adj)

    return jck_adj, simi_adj, dis_adj, cont_adj, func_adj, stage_adj


def main(args):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    args.config_filename = join(args.config_dir, "prnn_" + args.dataset + ".yaml")
    with open(args.config_filename) as f:

        supervisor_config = yaml.safe_load(f)

        jck_adj, simi_adj, dis_adj, cont_adj, func_adj, stage_adj = load_adj()

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(device_count={'GPU': 1})

        tf_config.gpu_options.allow_growth = True

        adj_mx = [jck_adj, simi_adj, cont_adj]  # jck_adj, simi_adj, dis_adj, cont_adj, func_adj, stage_adj

        with tf.Session(config=tf_config) as sess:
            supervisor = PRNNSupervisor(adj_mx=adj_mx, args=args, inference=False, pretrained_dir=None,
                                        **supervisor_config)
            args.pretrained_dir = supervisor._log_dir
            supervisor.train(sess=sess)

    print("the checkpoint files are saved in :", args.pretrained_dir)


if __name__ == '__main__':
    main(args)

