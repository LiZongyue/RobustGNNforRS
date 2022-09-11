import gc
import torch
import numpy as np
import argparse
import os
import lightgcn
import ngcf_ori
from register import dataset
import datetime
from topology_attack import PGDAttack
import utils
from utils_attack import attack_model, attack_randomly, attack_embedding
import Procedure
from scipy.sparse import csc_matrix, coo_matrix
from groc_loss import GROC_loss

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--warmup_steps', type=int, default=10000, help='Warm up steps for scheduler.')
parser.add_argument('--batch_size', type=int, default=2048, help='BS.')
parser.add_argument('--groc_batch_size', type=int, default=128, help='BS.')
parser.add_argument('--groc_epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1-keep probability).')
parser.add_argument('--train_groc', type=bool, default=False, help='control if train the groc')
parser.add_argument('--pgd_attack', type=bool, default=False, help='PGD attack and evaluate')
parser.add_argument('--embedding_attack', type=bool, default=False, help='PGD attack and evaluate')
parser.add_argument('--random_perturb', type=bool, default=False, help='perturb adj randomly and compare to PGD')
parser.add_argument('--groc_with_bpr', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
parser.add_argument('--groc_rdm_adj_attack', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
parser.add_argument('--groc_embed_mask', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
parser.add_argument('--gcl_with_bpr', type=bool, default=False, help='train a pre-trained GCN on GROC loss')
parser.add_argument('--use_scheduler', type=bool, default=False, help='Use scheduler for learning rate decay')
parser.add_argument('--use_IntegratedGradient', type=bool, default=False, help='Use scheduler for learning rate decay')
parser.add_argument('--groc_with_bpr_cat', type=bool, default=False, help='Use scheduler for learning rate decay')
parser.add_argument('--use_groc_pgd', type=bool, default=False, help='Use scheduler for learning rate decay')
parser.add_argument('--loss_weight_bpr', type=float, default=0.9,
                    help='train loss with learnable weight between 2 losses')
parser.add_argument('--dataset', type=str, default='ml-1m', choices=['ml-1m', 'amazon-book', 'gowalla', 'yelp2018'],
                    help='dataset')
parser.add_argument('--T_groc', type=float, default=0.7, help='param temperature for GROC')
parser.add_argument('--ptb_rate', type=float, default=0.5, help='perturbation rate')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--embed_attack_method', type=str, default='Gradient', choices=['Gradient', 'rdm'],
                    help='model variant')
parser.add_argument('--path_modified_adj', type=str,
                    default=os.path.abspath(os.path.dirname(os.getcwd())) + '/data/{}/modified_adj_{}_{}.pt',
                    help='path where modified adj matrix are saved')
parser.add_argument('--modified_adj_name', type=list,
                    default=['a_02', 'a_04', 'a_06', 'a_08', 'a_1', 'a_12', 'a_14', 'a_16', 'a_18', 'a_2'],
                    help='we attack adj twice for GROC training so we will have 2 modified adj matrix. In order to distinguish them we set a flag to save them independently')
parser.add_argument('--modified_adj_name_with_rdm_ptb_a', type=list,
                    default=['a_02_w_r', 'a_04_w_r', 'a_06_w_r', 'a_08_w_r', 'a_1_w_r', 'a_12_w_r', 'a_14_w_r',
                             'a_16_w_r', 'a_18_w_r', 'a_2_w_r'],
                    help='we attack adj twice for GROC training, 1st random 2nd PGD.')
parser.add_argument('--modified_adj_name_with_rdm_ptb_b', type=list,
                    default=['a_02_w_r_b', 'a_04_w_r_b', 'a_06_w_r_b', 'a_08_w_r_b', 'a_1_w_r_b', 'a_12_w_r_b',
                             'a_14_w_r_b', 'a_16_w_r_b', 'a_18_w_r_b', 'a_2_w_r_b'],
                    help='we attack adj twice for GROC training, 1st random 2nd PGD.')
parser.add_argument('--modified_adj_name_with_masked_M_a', type=list,
                    default=['a_02_mM_a', 'a_04_mM_a', 'a_06_mM_a', 'a_08_mM_a', 'a_1_mM_a', 'a_12_mM_a', 'a_14_mM_a',
                             'a_16_mM_a', 'a_18_mM_a', 'a_2_mM_a'],
                    help='masked_M indicates masked model(embedding mask)')
parser.add_argument('--modified_adj_name_with_masked_M_b', type=list,
                    default=['a_02_mM_b', 'a_04_mM_b', 'a_06_mM_b', 'a_08_mM_b', 'a_1_mM_b', 'a_12_mM_b', 'a_14_mM_b',
                             'a_16_mM_b', 'a_18_mM_b', 'a_2_mM_b'],
                    help='masked_M indicates masked model(embedding mask)')
parser.add_argument('--mask_prob_list', type=list, default=[0.1, 0.2, 0.3, 0.4],
                    help='we attack adj twice for GROC training, 1st random 2nd PGD.')
parser.add_argument('--mask_prob_idx', type=int, default=1,
                    help='we attack adj twice for GROC training, 1st random 2nd PGD.')
parser.add_argument('--perturb_strength_list', type=list, default=[10, 5, 3.33, 2.5, 2, 1.67, 1.42, 1.25, 1.11, 1],
                    help='2 perturb strength for 2 PGD attacks')
parser.add_argument('--modified_adj_id', type=int, default=0, help='select adj matrix from modified adj matrix ids')
parser.add_argument('--masked_model_a_id', type=int, default=2, help='select adj matrix from modified adj matrix ids')
parser.add_argument('--masked_model_b_id', type=int, default=1, help='select adj matrix from modified adj matrix ids')
parser.add_argument('--path_modified_models', type=str,
                    default=os.path.abspath(os.path.dirname(os.getcwd())) + '/data/modified_model_{}.pt',
                    help='path where modified model is saved')
parser.add_argument('--modified_models_name', type=list,
                    default=['02', '04', '06', '08', '1', '12', '14', '16', '18', '2'],
                    help='list of flags for modified models')
parser.add_argument('--eps', type=list, default=[0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2],
                    help='attack restriction eps for embedding attack')
parser.add_argument('--modified_models_id', type=int, default=0,
                    help='select model matrix from modified model matrix ids')
parser.add_argument('--mask_prob_1', type=float, default=0.3, help='mask embedding of users/items of GCN')
parser.add_argument('--mask_prob_2', type=float, default=0.4, help='mask embedding of users/items of GCN')
parser.add_argument('--insert_prob_1', type=float, default=0.4, help='mask embedding of users/items of GCN')
parser.add_argument('--insert_prob_2', type=float, default=0.4, help='mask embedding of users/items of GCN')
parser.add_argument('--remove_prob_1', type=float, default=0.2, help='mask embedding of users/items of GCN')
parser.add_argument('--remove_prob_2', type=float, default=0.4, help='mask embedding of users/items of GCN')
parser.add_argument('--generate_perturb_adj', type=bool, default=True, help='mask embedding of users/items of GCN')
parser.add_argument('--test_ratio', type=float, default=0.2, help='mask embedding of users/items of GCN')
parser.add_argument('--max_train_num', type=int, default=200, help='mask embedding of users/items of GCN')
parser.add_argument('--use_embedding', type=bool, default=False, help='mask embedding of users/items of GCN')
parser.add_argument('--hop', type=int, default=1, help='mask embedding of users/items of GCN')
parser.add_argument('--no_parallel', type=bool, default=True, help='mask embedding of users/items of GCN')
parser.add_argument('--max_nodes_per_hop', type=int, default=20, help='mask embedding of users/items of GCN')
parser.add_argument('--node_percentage_list', type=list, default=[0.25, 0.5, 0.75, 1],
                    help='mask embedding of users/items of GCN')
parser.add_argument('--node_percentage_list_index', type=int, default=0, help='mask embedding of users/items of GCN')
parser.add_argument('--model_ngcf', type=bool, default=False, help='mask embedding of users/items of GCN')
parser.add_argument('--model_gccf', type=bool, default=False, help='mask embedding of users/items of GCN')
parser.add_argument('--model_gcmc', type=bool, default=False, help='mask embedding of users/items of GCN')
parser.add_argument('--model_lightgcn', type=bool, default=False, help='mask embedding of users/items of GCN')
parser.add_argument('--use_dcl', type=bool, default=False, help='mask embedding of users/items of GCN')
parser.add_argument('--k', type=float, default=0.01, help='mask embedding of users/items of GCN')
parser.add_argument('--valid_freq', type=int, default=1, help='mask embedding of users/items of GCN')
parser.add_argument('--save_to', type=str, default='tmp', help='mask embedding of users/items of GCN')
parser.add_argument('--mask_type', type=str, default='mask_normalized_aggregated_emb',
                    help='mask embedding of users/items of GCN. Candidates: mask_aggregated_emb mask_normalized_aggregated_emb'
                         'and mask_emb')
parser.add_argument('--val_batch_size', type=int, default=2048, help='BS.')
parser.add_argument('--train_baseline', type=bool, default=False, help='BS.')
parser.add_argument('--prepare_adj_data', type=bool, default=False, help='BS.')
parser.add_argument('--only_groc_for_cal', type=bool, default=False, help='Train Model only with GCL.')
parser.add_argument('--with_bpr', type=bool, default=False, help='Import baseline and train with bpr loss backwards.')
parser.add_argument('--train_groc_pipeline', type=bool, default=False, help='GROC training')
parser.add_argument('--double_loss_baseline', type=bool, default=False, help='CL for RS baseline ')
parser.add_argument('--double_loss', type=bool, default=False, help='CL for RS double loss ')
parser.add_argument('--baseline_single_loss', type=bool, default=False, help='baseline with one loss. DCL or BPR')
parser.add_argument('--train_with_bpr_perturb', type=bool, default=False, help='GROC training controller. GCL_DCL training ')
parser.add_argument('--only_user_groc', type=bool, default=False, help='GROC training anchor node only from users ')
parser.add_argument('--finetune_pretrained', type=bool, default=False, help='GROC training anchor node only from users ')
parser.add_argument('--sgl_t', type=float, default=0.5, help='GROC training anchor node only from users ')
parser.add_argument('--drop_rate_inverse', type=float, default=0.95, help='GROC training anchor node only from users ')
parser.add_argument('--with_bpr_gradient', type=bool, default=False,
                    help='GROC adj insert/remove with bpr gradient signals.')

args = parser.parse_args()
num_users = dataset.n_user
num_items = dataset.m_item

print("=================================================")
print("All parameters in args")
print(args)
print("=================================================")
today = datetime.date.today().strftime('%y%m%d')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
net = dataset.getSparseGraph()
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)
    torch.cuda.empty_cache()
gc.collect()

ori_adj_path = os.path.abspath(os.path.dirname(os.getcwd())) + '/adj/{}/ori_adj.pt'.format(args.dataset)
if not os.path.exists(ori_adj_path):
    adj = utils.to_tensor(dataset.getSparseGraph(), device=device)
else:
    adj = torch.load(ori_adj_path, map_location='cpu').to(device)
# adj matrix only contains users and items
perturbations = int(args.ptb_rate * (net.sum() // args.perturb_strength_list[args.modified_adj_id]))
# perturbations = int(args.ptb_rate * (net.sum()))

rowsum = torch.tensor(net.sum(1)).to(device)
r_inv = rowsum.pow(-1 / 2).flatten()
r_inv[torch.isinf(r_inv)] = 0.

val_diag = r_inv
idx = np.where((torch.add(r_inv != 0, r_inv == 0)).detach().cpu())[0]
indices_diag = np.vstack((idx, idx))

i_d = torch.LongTensor(indices_diag).to(device)
v_d = torch.FloatTensor(val_diag.detach().cpu()).to(device)
shape = adj.shape

d_mtr = torch.sparse_coo_tensor(i_d, v_d, torch.Size(shape)).to(device)

# load training data (ID)
users, posItems, negItems = utils.getTrainSet(dataset)
users_val, posItems_val, negItems_val = utils.getValidSet(dataset)

# comment for GPU code, only for debugging
#
# users = users[:2048]
# posItems = posItems[:2048]
# negItems = negItems[:2048]

data_len = len(users)


def load_p_adj(dataset_name, model_name):
    a2 = torch.load("/home/stud/zongyue/workarea/RobustGNNforRS/data/{}/modified_adj_{}_a_02.pt".format(dataset_name, model_name)).to(device)
    a4 = torch.load("/home/stud/zongyue/workarea/RobustGNNforRS/data/{}/modified_adj_{}_a_04.pt".format(dataset_name, model_name)).to(device)
    a6 = torch.load("/home/stud/zongyue/workarea/RobustGNNforRS/data/{}/modified_adj_{}_a_06.pt".format(dataset_name, model_name)).to(device)
    a8 = torch.load("/home/stud/zongyue/workarea/RobustGNNforRS/data/{}/modified_adj_{}_a_08.pt".format(dataset_name, model_name)).to(device)
    # a2 = torch.load("C:/tmp/modified_adj_a_02.pt", map_location=torch.device('cpu')).to(device)
    # a4 = torch.load("C:/tmp/modified_adj_a_04.pt", map_location=torch.device('cpu')).to(device)
    # a6 = torch.load("C:/tmp/modified_adj_a_06.pt", map_location=torch.device('cpu')).to(device)
    # a8 = torch.load("C:/tmp/modified_adj_a_08.pt", map_location=torch.device('cpu')).to(device)
    return [a2, a4, a6, a8]


def train_groc_pipe(args_, model_, device_, dataset_, num_users_, num_items_, adj_, Recmodel_, d_mtr_, today_,
                    bpr_gradient_, bpr_flag_, data_len_, users_, posItems_, negItems_, users_val_, posItems_val_,
                    negItems_val_, adj_list_):
    mode = 'GROC'
    adj_path_ = os.path.abspath(os.path.dirname(os.getcwd())) + '/adj/{}/{}_adj_2_hops.pt'.format(args_.dataset,
                                                                                                  model_)
    utils.insert_adj_construction_pipeline(adj_path_, model_, args_, device_, dataset_, num_users_, num_items_,
                                           adj_)
    adj_2_hops_ = torch.load(adj_path_)
    Recmodel_ = Recmodel_.to(device)
    adj_2_hops_ = adj_2_hops_.to_dense().to(device)
    if not os.path.exists(os.path.abspath(os.path.dirname(os.getcwd())) + '/models/GROC_models'):
        os.mkdir(os.path.abspath(os.path.dirname(os.getcwd())) + '/models/GROC_models')
    if not os.path.exists(
            os.path.abspath(os.path.dirname(os.getcwd())) + '/models/GROC_models/{}'.format(args_.dataset)):
        os.mkdir(os.path.abspath(os.path.dirname(os.getcwd())) + '/models/GROC_models/{}'.format(args_.dataset))
    if not os.path.exists(os.path.abspath(os.path.dirname(os.getcwd())) + '/log/GROC_logs'):
        os.mkdir(os.path.abspath(os.path.dirname(os.getcwd())) + '/log/GROC_logs')
    if not os.path.exists(
            os.path.abspath(os.path.dirname(os.getcwd())) + '/log/GROC_logs/{}'.format(args_.dataset)):
        os.mkdir(os.path.abspath(os.path.dirname(os.getcwd())) + '/log/GROC_logs/{}'.format(args_.dataset))
    if args_.use_dcl:
        dcl = 'use_dcl'
    else:
        dcl = 'no_use_dcl'
    adj_rm_1 = None
    adj_rm_2 = None

    groc_ = GROC_loss(Recmodel_, adj_, d_mtr_, adj_2_hops_, args_, adj_list_)
    if args.double_loss:
        mode = 'GCLBPR_largest_drop'
        if args.double_loss_baseline:
            baseline = None
        else:
            local_path = os.path.abspath(os.path.dirname(os.getcwd()))
            if model_ == 'NGCF':
                path_ = local_path + '/models/{}/NGCF_baseline.ckpt'.format(args.dataset)
                baseline = ngcf_ori.NGCF(device, num_users, num_items, use_dcl=False)
                baseline.load_state_dict(torch.load(path_))
                baseline = baseline.to(device)

            elif model_ == 'GCMC':
                path_ = local_path + '/models/{}/GCMC_baseline.ckpt'.format(args.dataset)
                baseline = ngcf_ori.NGCF(device, num_users, num_items, is_gcmc=True, use_dcl=False)
                baseline.load_state_dict(torch.load(path_))
                baseline = baseline.to(device)

            elif model_ == 'GCCF':
                path_ = local_path + '/models/{}/GCCF_baseline.ckpt'.format(args.dataset)
                baseline = lightgcn.LightGCN(device, num_users, num_items, is_light_gcn=False, use_dcl=False)
                baseline.load_state_dict(torch.load(path_))
                baseline = baseline.to(device)

            elif model_ == 'LightGCN':
                path_ = local_path + '/models/{}/LightGCN_baseline.ckpt'.format(args.dataset)
                baseline = lightgcn.LightGCN(device, num_users, num_items, use_dcl=False)
                baseline.load_state_dict(torch.load(path_, map_location=torch.device('cpu')))
                baseline = baseline.to(device)
            else:
                raise Exception("Baseline Model Unknown.")
        adj_rm_1 = []
        adj_rm_2 = []

    model_path_ = os.path.abspath(os.path.dirname(os.getcwd())) + \
                  '/models/GROC_models/{}/{}_{}_{}_after_{}_{}_{}_{}_{}.ckpt'.format(args_.dataset, today_,
                                                                                       model_, dcl, bpr_gradient_, mode,
                                                                                       bpr_flag_, args_.loss_weight_bpr,
                                                                                       args_.groc_batch_size)
    log_path_ = os.path.abspath(os.path.dirname(os.getcwd())) + \
                '/log/GROC_logs/{}/{}_{}_{}_after_{}_{}_{}_{}_{}.log'.format(args_.dataset, today_, model_, dcl,
                                                                               bpr_gradient_, mode, bpr_flag_,
                                                                               args_.loss_weight_bpr, args_.groc_batch_size)
    groc_.groc_train_with_bpr_sparse(data_len_, users_, posItems_, negItems_, users_val_, posItems_val_,
                                     negItems_val_, model_path_, log_path_, adj_rm_1=adj_rm_1, adj_rm_2=adj_rm_2,
                                     sparse=False)

    print("===========================")
    print("GROC training finished!")


def attack_adjs(baseline_, adj_, perturbations_, rate_, users_, posItems_, negItems_, device_, drop_only=True, largest=False):
    """
    perturbations: # of perturbated edges
    rate: drop rate
    """
    if baseline_ is not None:
        baseline_.train()
    gradient_adj = torch.zeros(adj_.size()).to(device_)
    ori_adj_sparse = utils.normalize_adj_tensor(adj_).to_sparse()  # for bpr loss

    adj_perturb = None
    if drop_only:
        if args.double_loss_baseline:
            random_tensor = 1 - rate_
            sparse_adj = adj_.to_sparse().to(device_)
            random_tensor += torch.rand(sparse_adj._nnz()).to(device_)
            dropout_mask = torch.floor(random_tensor).type(torch.bool)
            i = sparse_adj._indices()
            v = sparse_adj._values()

            i = i[:, dropout_mask]
            v = v[dropout_mask]

            out = torch.sparse.FloatTensor(i, v, adj_.shape).to(device_)
            adj_perturb = out.to(device_)
            adj_perturb = utils.normalize_adj_tensor(adj_perturb, d_mtr, sparse=True)

        else:
            users = users_.to(device_)
            posItems = posItems_.to(device_)
            negItems = negItems_.to(device_)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            baseline_.train()
            for (batch_i,
                 (batch_users,
                  batch_pos,
                  batch_neg)) in enumerate(utils.minibatch(users,
                                                           posItems,
                                                           negItems,
                                                           batch_size=2048)):
                ori_adj_sparse.requires_grad = True
                bpr_loss, _ = baseline_.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg)

                gradient_bpr = torch.autograd.grad(bpr_loss, ori_adj_sparse, retain_graph=True)[0].to_dense()
                gradient_adj = gradient_adj + gradient_bpr
            # TODO: remove who? Largest or Smallest?
            gradient_adj = gradient_adj * adj_
            v, i = torch.topk(gradient_adj.flatten(), perturbations_, largest=largest)
            ind_rm = torch.tensor(np.array(np.unravel_index(i.detach().cpu().numpy(), gradient_adj.shape)).T).reshape(2,
                                                                                                                      -1).to(
                device_)
            m = (torch.rand(perturbations_) > 0.6).to(device_)  # 0.4 * 0.5 = 0.2 drop
            ind_rm = ind_rm[:, m]
            val_rm = torch.ones(ind_rm.shape[1]).to(device_)

            out = torch.sparse.FloatTensor(ind_rm, val_rm, adj_.shape).to(device_)
            adj_perturb = (out * (1. / (1 - rate_))).to_dense().to(device_)

    return adj_perturb


if args.train_baseline:
    def train_baseline(baseline, ori_adj_tensor, degree_mtx, all_users, pos_items_pair, neg_items_sample, dataset_):
        baseline = baseline.to(device)
        baseline.fit(ori_adj_tensor, degree_mtx, all_users, pos_items_pair, neg_items_sample, dataset_, dataset)

    if args.baseline_single_loss:
        if args.model_ngcf:
            print("NGCF Baseline Model Calibration.")
            model = ngcf_ori.NGCF(device, num_users, num_items, use_dcl=args.use_dcl)
            train_baseline(model, adj, d_mtr, users, posItems, negItems, args.dataset)
        if args.model_gcmc:
            print("GCMC Baseline Model Calibration.")
            model = ngcf_ori.NGCF(device, num_users, num_items, is_gcmc=True, use_dcl=args.use_dcl)
            train_baseline(model, adj, d_mtr, users, posItems, negItems, args.dataset)
        if args.model_lightgcn:
            print("LightGCN Baseline Model Calibration.")
            model = lightgcn.LightGCN(device, num_users, num_items, use_dcl=args.use_dcl)
            train_baseline(model, adj, d_mtr, users, posItems, negItems, args.dataset)
        if args.model_gccf:
            print("LR-GCCF Baseline Model Calibration.")
            model = lightgcn.LightGCN(device, num_users, num_items, is_light_gcn=False, use_dcl=args.use_dcl)
            train_baseline(model, adj, d_mtr, users, posItems, negItems, args.dataset)

    elif args.double_loss:
        adj = adj.to_dense().to(device)
        bpr_gradient = 'baseline'
        bpr_flag = 'double_loss'
        if args.model_ngcf:
            print("NGCF Baseline Model with double Loss Calibration.")
            model = 'NGCF'
            adj_list = load_p_adj(args.dataset, model)
            Recmodel = ngcf_ori.NGCF(device, num_users, num_items, use_dcl=args.use_dcl)
            train_groc_pipe(args, model, device, dataset, num_users, num_items, adj, Recmodel, d_mtr, today,
                            bpr_gradient, bpr_flag, data_len, users, posItems, negItems, users_val, posItems_val,
                            negItems_val, adj_list)

        if args.model_lightgcn:
            print("LightGCN Baseline Model with double Loss Calibration.")
            model = 'LightGCN'
            adj_list = load_p_adj(args.dataset, model)
            Recmodel = lightgcn.LightGCN(device, num_users, num_items, is_light_gcn=False, use_dcl=args.use_dcl)
            train_groc_pipe(args, model, device, dataset, num_users, num_items, adj, Recmodel, d_mtr, today,
                            bpr_gradient, bpr_flag, data_len, users, posItems, negItems, users_val, posItems_val,
                            negItems_val, adj_list)

        if args.model_gccf:
            print("GCCF Baseline Model with double Loss Calibration.")
            model = 'GCCF'
            adj_list = load_p_adj(args.dataset, model)
            Recmodel = lightgcn.LightGCN(device, num_users, num_items, is_light_gcn=False, use_dcl=args.use_dcl)
            train_groc_pipe(args, model, device, dataset, num_users, num_items, adj, Recmodel, d_mtr, today,
                            bpr_gradient, bpr_flag, data_len, users, posItems, negItems, users_val, posItems_val,
                            negItems_val, adj_list)

        if args.model_gcmc:
            print("GCMC Baseline Model with double Loss Calibration.")
            model = 'GCMC'
            adj_list = load_p_adj(args.dataset, model)
            Recmodel = ngcf_ori.NGCF(device, num_users, num_items, is_gcmc=True, use_dcl=args.use_dcl)
            train_groc_pipe(args, model, device, dataset, num_users, num_items, adj, Recmodel, d_mtr, today,
                            bpr_gradient, bpr_flag, data_len, users, posItems, negItems, users_val, posItems_val,
                            negItems_val, adj_list)

if args.train_groc:
    if args.model_ngcf:
        adj = adj.to_dense().to(device)
        print("train model NGCF")
        print("=================================================")
        Recmodel = ngcf_ori.NGCF(device, num_users, num_items, use_dcl=args.use_dcl)
        model = 'NGCF'
        adj_list = load_p_adj(args.dataset, model)
        bpr_flag = 'with_BPR'
        bpr_gradient = 'without_bpr_gradient'
        if args.with_bpr_gradient:
            bpr_gradient = 'with_bpr_gradient'
        if not args.with_bpr:
            path = os.path.abspath(os.path.dirname(os.getcwd())) + '/models/{}/NGCF_baseline.ckpt'.format(args.dataset)
            if not os.path.exists(path):
                raise Exception("Baseline model not found. Please calibrate models first.")
            Recmodel.load_state_dict(torch.load(path))
            bpr_flag = 'without_BPR'
        train_groc_pipe(args, model, device, dataset, num_users, num_items, adj, Recmodel, d_mtr, today, bpr_gradient,
                        bpr_flag, data_len, users, posItems, negItems, users_val, posItems_val, negItems_val, adj_list)

    if args.model_lightgcn:
        adj = adj.to_dense().to(device)
        print("train model LightGCN")
        print("=================================================")
        Recmodel = lightgcn.LightGCN(device, num_users, num_items, use_dcl=args.use_dcl)
        model = 'LightGCN'
        adj_list = load_p_adj(args.dataset, model)
        bpr_flag = 'with_BPR'
        bpr_gradient = 'without_bpr_gradient'
        if args.with_bpr_gradient:
            bpr_gradient = 'with_bpr_gradient'
        if not args.with_bpr and args.finetune_pretrained:
            path = os.path.abspath(os.path.dirname(os.getcwd())) + '/models/{}/LightGCN_baseline.ckpt'.format(
                args.dataset)
            if not os.path.exists(path):
                raise Exception("Baseline model not found. Please calibrate models first.")
            Recmodel.load_state_dict(torch.load(path))

            bpr_flag = 'without_BPR'
        train_groc_pipe(args, model, device, dataset, num_users, num_items, adj, Recmodel, d_mtr, today, bpr_gradient,
                        bpr_flag, data_len, users, posItems, negItems, users_val, posItems_val, negItems_val, adj_list)

    if args.model_gcmc:
        adj = adj.to_dense().to(device)
        print("train model GCMC")
        print("=================================================")
        Recmodel = ngcf_ori.NGCF(device, num_users, num_items, is_gcmc=True, use_dcl=args.use_dcl)
        model = 'GCMC'
        adj_list = load_p_adj(args.dataset, model)
        bpr_flag = 'with_BPR'
        bpr_gradient = 'without_bpr_gradient'
        if args.with_bpr_gradient:
            bpr_gradient = 'with_bpr_gradient'
        if not args.with_bpr:
            path = os.path.abspath(os.path.dirname(os.getcwd())) + '/models/{}/GCMC_baseline.ckpt'.format(args.dataset)
            if not os.path.exists(path):
                raise Exception("Baseline model not found. Please calibrate models first.")
            Recmodel.load_state_dict(torch.load(path))

            bpr_flag = 'without_BPR'
        train_groc_pipe(args, model, device, dataset, num_users, num_items, adj, Recmodel, d_mtr, today, bpr_gradient,
                        bpr_flag, data_len, users, posItems, negItems, users_val, posItems_val, negItems_val, adj_list)
    if args.model_gccf:
        adj = adj.to_dense().to(device)
        print("train model LR-GCCF")
        print("=================================================")
        Recmodel = lightgcn.LightGCN(device, num_users, num_items, is_light_gcn=False, use_dcl=args.use_dcl)
        model = 'GCCF'
        adj_list = load_p_adj(args.dataset, model)
        bpr_flag = 'with_BPR'
        bpr_gradient = 'without_bpr_gradient'
        if args.with_bpr_gradient:
            bpr_gradient = 'with_bpr_gradient'
        if not args.with_bpr:
            path = os.path.abspath(os.path.dirname(os.getcwd())) + '/models/{}/GCCF_baseline.ckpt'.format(args.dataset)
            if not os.path.exists(path):
                raise Exception("Baseline model not found. Please calibrate models first.")
            Recmodel.load_state_dict(torch.load(path))
            # Recmodel._is_sparse = False
            bpr_flag = 'without_BPR'
        train_groc_pipe(args, model, device, dataset, num_users, num_items, adj, Recmodel, d_mtr, today, bpr_gradient,
                        bpr_flag, data_len, users, posItems, negItems, users_val, posItems_val, negItems_val, adj_list)

if args.random_perturb:
    print("train model using random perturbation")
    print("=================================================")
    modified_adj = attack_randomly(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                   args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
    try:
        print("modified adjacency is same as original adjacency: ", (modified_adj == adj).all())
    except AttributeError:
        print("adjacency is not modified. Check your perturbation and make sure 0 isn't assigned.")

    Recmodel_ = lightgcn.LightGCN(device)
    Recmodel_ = Recmodel_.to(device)
    Recmodel_.fit(adj, users, posItems, negItems)
    print("evaluate the model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel_, 1, utils.normalize_adj_tensor(modified_adj), None, 0)
    print("=================================================")

if args.pgd_attack:
    print("train model with pgd attack")
    print("=================================================")
    adj = adj.to_dense()
    # Setup Attack Model
    local_path = os.path.abspath(os.path.dirname(os.getcwd()))
    if args.model_ngcf:
        print("NGCF PGD attack.")
        Recmodel = ngcf_ori.NGCF(device, num_users, num_items, use_dcl=args.use_dcl)
        model = 'NGCF'
        path_ = local_path + '/models/{}/NGCF_baseline_no_use_dcl.ckpt'.format(args.dataset)
        Recmodel.load_state_dict(torch.load(path_))
        Recmodel = Recmodel.to(device)
        modified_adj = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                    args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device, model,
                                    args.dataset)

    if args.model_gcmc:
        print("GCMC PGD attack.")
        Recmodel = ngcf_ori.NGCF(device, num_users, num_items, is_gcmc=True, use_dcl=args.use_dcl)
        model = 'GCMC'
        path_ = local_path + '/models/{}/GCMC_baseline_no_use_dcl.ckpt'.format(args.dataset)
        Recmodel.load_state_dict(torch.load(path_))
        Recmodel = Recmodel.to(device)
        modified_adj = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                    args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device, model,
                                    args.dataset)

    # if args.model_lightgcn:
    #     print("LightGCN PGD attack.")
    #     Recmodel = lightgcn.LightGCN(device, num_users, num_items, use_dcl=args.use_dcl)
    if args.model_lightgcn:
        print("LightGCN PGD attack.")
        Recmodel = lightgcn.LightGCN(device, num_users, num_items, use_dcl=args.use_dcl)
        model = 'LightGCN'
        path_ = local_path + '/models/{}/LightGCN_baseline_no_use_dcl.ckpt'.format(args.dataset)
        Recmodel.load_state_dict(torch.load(path_))
        Recmodel = Recmodel.to(device)
        modified_adj = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                    args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device, model,
                                    args.dataset)

    if args.model_gccf:
        print("LR-GCCF PGD attack.")
        Recmodel = lightgcn.LightGCN(device, num_users, num_items, is_light_gcn=False, use_dcl=args.use_dcl)
        model = 'GCCF'
        path_ = local_path + '/models/{}/GCCF_baseline_no_use_dcl.ckpt'.format(args.dataset)
        Recmodel.load_state_dict(torch.load(path_))
        Recmodel = Recmodel.to(device)
        modified_adj = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                    args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device, model,
                                    args.dataset)


if args.embedding_attack:
    print("train model with embedding adversarial attack")
    print("=================================================")
    fit_model = lightgcn.LightGCN(device)
    modified_model = attack_embedding(fit_model, adj, args.eps[args.modified_models_id],
                                      args.path_modified_models, args.modified_models_name, args.modified_models_id,
                                      users, posItems, negItems, num_users, device)

    fit_model = fit_model.to(device)
    fit_model.fit(adj, users, posItems, negItems)

    print("evaluate the ATTACKED model with original adjacency matrix")
    Procedure.Test(dataset, fit_model, 1, utils.normalize_adj_tensor(adj), None, 0)
    print("=================================================")
