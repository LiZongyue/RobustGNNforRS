'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--groc_batch_size', type=int, default=128, help='BS.')
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-1m]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--groc_epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--train_groc_casade', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='use scheduler for learning rate decay')
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--modified_adj_id', type=list, default=0,
                        help='select adj matrix from modified adj matrix ids')
    parser.add_argument('--train_groc', type=bool, default=False, help='control if train the groc')
    parser.add_argument('--loss_weight_bpr', type=float, default=0.9, help='control loss form')
    parser.add_argument('--modified_models_id', type=int, default=0,
                        help='select model matrix from modified model matrix ids')
    parser.add_argument('--T_groc', type=float, default=0.7, help='param temperature for GROC')
    parser.add_argument('--embedding_attack', type=bool, default=False, help='PDG attack and evaluate')
    parser.add_argument('--groc_embed_mask', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--pdg_attack', type=bool, default=False, help='PDG attack and evaluate')
    parser.add_argument('--random_perturb', type=bool, default=False, help='perturb adj randomly and compare to PGD')
    parser.add_argument('--pgd_attack', type=bool, default=False, help='PGD attack and evaluate')
    parser.add_argument('--groc_rdm_adj_attack', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--groc_with_bpr', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--gcl_with_bpr', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--groc_with_bpr_cat', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--use_IntegratedGradient', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--use_groc_pgd', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--model_lightgcn', type=bool, default=False, help='mask embedding of users/items of GCN')
    parser.add_argument('--model_ngcf', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
    parser.add_argument('--baseline_single_loss', type=bool, default=False, help='bsl single loss')
    parser.add_argument('--k', type=float, default=0.01, help='mask embedding of users/items of GCN')
    parser.add_argument('--valid_freq', type=int, default=1, help='valid freq')
    parser.add_argument('--save_to', type=str, default='tmp', help='save path of ckpt and log')
    parser.add_argument('--val_batch_size', type=int, default=2048, help='val BS.')
    parser.add_argument('--batch_size', type=int, default=2048, help='val BS.')
    parser.add_argument('--train_baseline', type=bool, default=False, help='train baseline.')
    parser.add_argument('--prepare_adj_data', type=bool, default=False, help='BS.')
    parser.add_argument('--use_dcl', type=bool, default=False, help='mask embedding of users/items of GCN')
    parser.add_argument('--model_gccf', type=bool, default=False, help='mask embedding of users/items of GCN')
    parser.add_argument('--model_gcmc', type=bool, default=False, help='mask embedding of users/items of GCN')
    parser.add_argument('--with_bpr', type=bool, default=False, help='BS.')
    parser.add_argument('--train_groc_pipeline', type=bool, default=False, help='GROC training')
    parser.add_argument('--double_loss_baseline', type=bool, default=False,
                        help='GROC training. When False, GCL_DCL training ')
    parser.add_argument('--double_loss', type=bool, default=False, help='CL for RS double loss ')
    parser.add_argument('--train_with_bpr_perturb', type=bool, default=False,
                        help='GROC training controller. GCL_DCL training ')
    parser.add_argument('--only_user_groc', type=bool, default=False, help='GROC training anchor node only from users ')
    parser.add_argument('--with_bpr_gradient', type=bool, default=False,
                        help='GROC adj insert/remove with bpr gradient signals.')
    parser.add_argument('--insert_prob_1', type=float, default=0.4, help='mask embedding of users/items of GCN')
    parser.add_argument('--insert_prob_2', type=float, default=0.4, help='mask embedding of users/items of GCN')
    parser.add_argument('--remove_prob_1', type=float, default=0.2, help='mask embedding of users/items of GCN')
    parser.add_argument('--remove_prob_2', type=float, default=0.4, help='mask embedding of users/items of GCN')

    return parser.parse_args()
