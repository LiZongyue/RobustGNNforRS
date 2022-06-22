import scipy.sparse as sp
import world
import random
import gc
import torch
from torch import nn, optim
import numpy as np
from dataloader import BasicDataset
from time import time
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import os
import lightgcn
import ngcf_ori
from datetime import datetime
from transformers import get_linear_schedule_with_warmup


def save_model(model, file_name):
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print('Saving model to', file_name)
    torch.save(model.state_dict(), file_name)
    print('Saved model to ', file_name)
    return


def build_score(device, adj_u_i, args, num_users, num_items):
    # make adj_u_i a tensor
    # calculate 3 dense hop neighbors
    print("Starting calculate 3 hops neighbours...")
    adj_after_2_hops = torch.mm(torch.mm(adj_u_i, adj_u_i.t()), adj_u_i).bool()
    adj_u_i = adj_u_i.bool()

    if device != 'cpu':
        torch.cuda.empty_cache()
    gc.collect()
    print("Neighbours calculation finished!")

    adj_insert = adj_after_2_hops ^ adj_u_i  # subtraction (XOR)

    del adj_after_2_hops, adj_u_i
    if device != 'cpu':
        torch.cuda.empty_cache()
    gc.collect()

    chunk_size = 1000
    adj_insert_list = []
    for chunk_i in range(1, int(adj_insert.shape[0] / chunk_size) + 1):
        score_i = adj_insert[(chunk_i - 1) * chunk_size:chunk_i * chunk_size, :]
        adj_insert_list.append(score_i)
    adj_insert_list.append(adj_insert[chunk_i * chunk_size:, :])

    del adj_insert
    if device != 'cpu':
        torch.cuda.empty_cache()
    gc.collect()

    # import calibrated GNN model and utilize its embeddings for topK
    local_path = os.path.abspath(os.path.dirname(os.getcwd()))
    user_embed, item_embed = None, None
    if args.baseline == 'NGCF':
        print('loading baseline Model NGCF...')
        path = local_path + '/models/NGCF_baseline.ckpt'
        baseline = ngcf_ori.NGCF(device, num_users, num_items)
        baseline.load_state_dict(torch.load(path))
        baseline = baseline.to(device)
        user_embed = baseline.embedding_dict["user_emb"].data
        item_embed = baseline.embedding_dict["item_emb"].data
    if args.baseline == 'GCMC':
        print('loading baseline Model GCMC...')
        path = local_path + '/models/GCMC_baseline.ckpt'
        baseline = ngcf_ori.NGCF(device, num_users, num_items, is_gcmc=True)
        baseline.load_state_dict(torch.load(path))
        baseline = baseline.to(device)
        user_embed = baseline.embedding_dict["user_emb"].data
        item_embed = baseline.embedding_dict["item_emb"].data
    if args.baseline == 'lightGCN':
        print('loading baseline Model lightGCN...')
        path = local_path + '/models/lightGCN_baseline.ckpt'
        baseline = lightgcn.LightGCN(device)
        baseline.load_state_dict(torch.load(path))
        baseline = baseline.to(device)
        user_embed = baseline.embedding_user.data
        item_embed = baseline.embedding_item.data
    if args.baseline == 'LR-GCCF':
        print('loading baseline Model LR-GCCF...')
        path = local_path + '/models/gccf_baseline.ckpt'
        baseline = lightgcn.LightGCN(device, is_light_gcn=False)
        baseline.load_state_dict(torch.load(path))
        baseline = baseline.to(device)
        user_embed = baseline.embedding_user.data
        item_embed = baseline.embedding_item.data

    if user_embed is None or item_embed is None:
        raise Exception('check BaseLine loading! No Embedding loaded.')

    score = user_embed @ item_embed.T

    del user_embed, item_embed, baseline
    if device != 'cpu':
        torch.cuda.empty_cache()
    gc.collect()
    chunk_size = 1000
    scores = []
    for chunk_i in range(1, int(score.shape[0] / chunk_size) + 1):
        score_i = score[(chunk_i - 1) * chunk_size:chunk_i * chunk_size, :] * adj_insert_list[0]
        scores.append(score_i)
        del adj_insert_list[0]
        if device != 'cpu':
            torch.cuda.empty_cache()
        gc.collect()

    scores.append(score[(chunk_i) * chunk_size:, :] * adj_insert_list[0])
    del adj_insert_list[-1], score
    if device != 'cpu':
        torch.cuda.empty_cache()
    gc.collect()

    score = torch.cat(scores, 0)
    return score


def build_two_hop_adj(device, adj, score, args, num_users):

    add_num_row = (torch.count_nonzero(score, 1) * args.k).int()
    add_num_row = add_num_row.detach().cpu().numpy()
    insert_ind = []

    for idx in range(len(add_num_row)):
        _, col_idx = torch.topk(score[idx], add_num_row[idx])
        if col_idx.nelement() != 0:  # check the col_idx is empty
            insert_ind.append(torch.stack([torch.tensor(idx).repeat(len(col_idx)).to(device), col_idx], 0))

    insert_ind = torch.cat(insert_ind, 1)

    ind_up_tri = torch.stack((insert_ind[0], insert_ind[1].clone() + num_users))
    ind_down_tri = ind_up_tri.clone()
    ind_down_tri[[1, 0]] = ind_down_tri.clone()
    ind = torch.cat([ind_up_tri, ind_down_tri, adj.coalesce().indices()], -1)
    adj_2_hops = torch.sparse_coo_tensor(ind, torch.ones(ind.shape[1]), adj.shape).to(device)

    return adj_2_hops


def append_log_to_file(eval_log, epoch, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('Creating new log file')
    f = open(filename, 'a+')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f.write('Log time: %s\n' % dt_string)
    f.write('Epoch %d\n' % epoch)
    for line in eval_log:
        f.write('%s\n' % line)
    f.write('\n')
    f.close()


def tensor2onehot(labels):
    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False, device='cpu'):
    if preprocess_adj:
        adj_norm = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())
    return adj.to(device), features.to(device), labels.to(device)


def to_tensor(adj, device='cuda:0'):
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)

    return adj.to(device)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


def normalize_feature(mx):
    """Row-normalize sparse matrix"""
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj_tensor(adj, m_d=None, sparse=False):
    device = torch.device("cuda" if adj.is_cuda else "cpu")
    if sparse and m_d is not None:

        index = torch.arange(0, adj.size()[0])
        i = torch.stack((index, index)).to(device)
        v = torch.ones(i.shape[1]).to(device)
        e = torch.sparse_coo_tensor(i, v, adj.size()).to(device)

        mx = adj + e

        mx = torch.sparse.mm(m_d, mx)
        mx = torch.sparse.mm(mx, m_d)

        return mx
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        # adj = to_scipy(adj)
        # mx = normalize_adj(adj)
        # return sparse_mx_to_torch_sparse_tensor(mx).to(device)

    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def to_scipy(tensor):
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def is_sparse_tensor(tensor):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class BPRLoss:
    def __init__(self,
                 recmodel: PairWiseModel,
                 config: dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def UniformSample_original(users, dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset: BasicDataset
    '''
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    '''

    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for user in range(dataset.n_users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        # posindex = np.random.randint(0, len(posForUser))
        for posindex in range(len(posForUser)):
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
            end = time()
            sample_time1 += end - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]


def UniformSample_originalTest(users, dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset: BasicDataset
    '''
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    '''

    allPos = dataset.allPostest
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for user in range(dataset.n_users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        # posindex = np.random.randint(0, len(posForUser))
        for posindex in range(len(posForUser)):
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
            end = time()
            sample_time1 += end - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]


def getTrainSet(dataset):
    allusers = list(range(dataset.n_users))
    S, sam_time = UniformSample_original(allusers, dataset)
    print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    return users, posItems, negItems


def getTestSet(dataset):
    allusers = list(range(dataset.n_users))
    S, sam_time = UniformSample_originalTest(allusers, dataset)
    print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    return users, posItems, negItems


# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def scheduler_groc(optimizer, data_len, warmup_steps, n_batch, n_epochs):
    num_training_steps = int(data_len / n_batch * n_epochs)
    scheduler_ = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    return scheduler_


def node_list_generation(args, num_users, num_items, i, adj_shape):
    if i > num_users:  # i is item
        n = int(args.node_percentage_list[args.node_percentage_list_index] * num_users)
    else:
        n = int(args.node_percentage_list[args.node_percentage_list_index] * num_items)
    return random.sample(range(num_users, adj_shape), n)


class Fake_model:
    def __init__(self):
        self.with_relu = False
        self.nclass = 1
        self.nfeat = 100
        self.hidden_sizes = 128

# ====================end Metrics=============================
# =========================================================
