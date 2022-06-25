import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import utils


class NGCF(nn.Module):
    def __init__(self, device, n_user, n_item, is_gcmc=False, sparse=True, use_dcl=True):
        super(NGCF, self).__init__()
        self.device = device
        self.lr = 0.001
        self.weight_decay = 1e-4
        self.n_layers = 2
        self.num_users = n_user
        self.num_items = n_item
        self.adj_shape = self.num_items + self.num_users
        self.latent_dim = 64
        self.f = nn.Sigmoid()
        self._is_sparse = sparse
        self.is_gcmc = is_gcmc
        self.use_dcl = use_dcl
        # self.mess_dropout = [0.1, 0.1]

        self.tau_plus = 1e-3
        self.T = 0.07

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.num_users,
                                                 self.latent_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.num_items,
                                                 self.latent_dim)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.latent_dim] + self.n_layers * [self.latent_dim]
        for k in range(self.n_layers):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_mlp_%d' % k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                                     layers[k + 1])))})
            weight_dict.update({'b_mlp_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

        return embedding_dict, weight_dict

    def fit(self, adj, d, users, posItems, negItems, users_val, posItems_val, negItems_val, dataset):
        if self._is_sparse:
            if type(adj) is not torch.Tensor:
                adj_norm = utils.normalize_adj_tensor(adj, d, sparse=True)
                adj = utils.to_tensor(adj_norm, device=self.device)
            else:
                adj_norm = utils.normalize_adj_tensor(adj, d, sparse=True)
                adj = adj_norm.to(self.device)
            # self.adj=adj
        else:
            if type(adj) is not torch.Tensor:
                adj_norm = utils.normalize_adj_tensor(adj)
                adj = utils.to_tensor(adj_norm, device=self.device)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
                adj = adj_norm.to(self.device)

        self._train_with_val(adj, users, posItems, negItems, users_val, posItems_val, negItems_val, dataset)

    def getUsersRating(self, adj, users):
        all_users, all_items = self.computer(adj)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def computer(self, adj):
        # TODO: override lightGCN here
        """
        propagate methods for lightGCN
        """

        g_droped = self.sparse_dropout(adj, 0.2, adj._nnz())

        all_emb = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embedding = [all_emb]

        for k in range(self.n_layers):
            side_embeddings = torch.sparse.mm(g_droped, all_emb)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(all_emb, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            # ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            if self.is_gcmc:
                mlp_embeddings = torch.matmul(norm_embeddings, self.weight_dict['W_mlp_%d' % k]) + self.weight_dict['b_mlp_%d' % k]
                # mlp_embeddings = torch.dropout(mlp_embeddings, 1 - self.mess_dropout[k])
                all_embedding += [mlp_embeddings]
            else:
                all_embedding += [norm_embeddings]

        all_emb = torch.cat(all_embedding, 1)
        users, items = torch.split(all_emb, [self.num_users, self.num_items])
        return users, items

    def forward(self, adj, users, items):
        # compute embedding
        all_users, all_items = self.computer(adj)
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def getEmbedding(self, adj, users, pos_items, neg_items=None, query_groc=False):
        """
        query from GROC means that we want to push adj into computational graph
        """

        all_users, all_items = self.computer(adj)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        # neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_dict['user_emb'].data[users]
        pos_emb_ego = self.embedding_dict['item_emb'].data[pos_items]
        # neg_emb_ego = self.embedding_item(neg_items)
        if neg_items is None:
            return users_emb, pos_emb, users_emb_ego, pos_emb_ego
        else:
            neg_emb = all_items[neg_items]
            return users_emb, pos_emb, users_emb_ego, pos_emb_ego, neg_emb

    def get_negative_mask_1(self, batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            # negative_mask[i,i+batch_size]=0

        # negative_mask=torch.cat((negative_mask,negative_mask),0)
        return negative_mask

    def bpr_loss(self, adj, users, poss, negative):
        '''
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(adj, users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        '''

        (users_emb, pos_emb, userEmb0, posEmb0, neg_emb) = self.getEmbedding(adj, users.long(), poss.long(), negative.long())
        # pos_emb_old=pos_emb
        if self.use_dcl:
            users_emb = nn.functional.normalize(users_emb, dim=1)
            pos_emb = nn.functional.normalize(pos_emb, dim=1)

            # neg score
            # out=torch.cat([users_emb,pos_emb],dim=0)
            # neg=torch.exp(torch.mm(out,out.t().contiguous())/self.T)
            neg = torch.exp(torch.mm(users_emb, pos_emb.t().contiguous()) / self.T)
            mask = self.get_negative_mask_1(pos_emb.size(0)).to(self.device)
            # neg=neg.masked_select(mask).view(2*pos_emb.size(0),-1)
            neg = neg.masked_select(mask).view(pos_emb.size(0), -1)

            # pos score
            pos = torch.exp(torch.sum(users_emb * pos_emb, dim=-1) / self.T)
            # pos=torch.cat([pos,pos],dim=0)

            # estimator g()
            # N=pos_emb.size(0)*2-2
            N = pos_emb.size(0) - 1
            Ng = (-self.tau_plus * N * pos + neg.sum(dim=-1)) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.T))

            loss = (-torch.log(pos / (pos + Ng))).mean()

            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                                  posEmb0.norm(2).pow(2)) / float(len(users))

        else:
            pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
            neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

            reg_loss = (torch.norm(users_emb) ** 2 + torch.norm(pos_emb) ** 2 + torch.norm(neg_emb) ** 2) / 2

        return loss, reg_loss

    def _train_with_val(self, adj, users, posItems, negItems, users_val, posItems_val, negItems_val, dataset):
        local_path = os.path.abspath(os.path.dirname(os.getcwd()))
        if not os.path.exists(local_path + '/models/{}'.format(dataset)):
            os.mkdir(local_path + '/models/{}'.format(dataset))
        if not os.path.exists(local_path + '/log/{}'.format(dataset)):
            os.mkdir(local_path + '/log/{}'.format(dataset))

        if self.is_gcmc:
            checkpoint_file_name = '{}/models/{}/GCMC_baseline.ckpt'.format(local_path, dataset)
            log_file_name = '{}/log/{}/GCMC_baseline.log'.format(local_path, dataset)
        else:
            checkpoint_file_name = '{}/models/{}/NGCF_baseline.ckpt'.format(local_path, dataset)
            log_file_name = '{}/log/{}/NGCF_baseline.log'.format(local_path, dataset)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        min_val_loss = float('Inf')
        for i in range(100):
            eval_log = []
            self.train()
            optimizer.zero_grad()
            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)
            total_batch = len(users) // 2048 + 1
            aver_loss = 0.
            for (batch_i,
                 (batch_users,
                  batch_pos,
                  batch_neg)) in enumerate(utils.minibatch(users,
                                                           posItems,
                                                           negItems,
                                                           batch_size=2048)):
                loss, reg_loss = self.bpr_loss(adj, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.weight_decay
                loss = loss + reg_loss

                loss.backward()
                optimizer.step()

                aver_loss += loss.cpu().item()
            aver_loss = aver_loss / total_batch
            if i % 10 == 0:
                print("Epoch {} BPR training Loss: {}".format(i, aver_loss))

            self.eval()
            save = False
            with torch.no_grad():
                aver_val_loss = 0.
                total_batch_val = len(users_val) // 2048 + 1
                users_val = users_val.to(self.device)
                posItems_val = posItems_val.to(self.device)
                negItems_val = negItems_val.to(self.device)
                users_val, posItems_val, negItems_val = utils.shuffle(users_val, posItems_val, negItems_val)
                for (batch_i,
                     (batch_users_val,
                      batch_pos_val,
                      batch_neg_val)) in enumerate(utils.minibatch(users_val,
                                                                   posItems_val,
                                                                   negItems_val,
                                                                   batch_size=2048)):
                    val_loss, val_reg_loss = self.bpr_loss(adj, batch_users_val, batch_pos_val, batch_neg_val)
                    val_reg_loss = val_reg_loss * self.weight_decay
                    val_loss = val_loss + val_reg_loss

                    aver_val_loss += val_loss

                aver_val_loss = aver_val_loss / total_batch_val
                eval_log.append("Valid Epoch: {}:".format(i))
                eval_log.append("average Val Loss NGCF: {}:".format(aver_val_loss))

                if aver_val_loss < min_val_loss:
                    save = True

                if save:
                    utils.save_model(self, checkpoint_file_name)
                    eval_log.append("Val loss decrease from {} to {}, save model!".format(min_val_loss, aver_val_loss))
                    utils.append_log_to_file(eval_log, i, log_file_name)
                    min_val_loss = aver_val_loss
