import torch
import os
from torch import nn, optim
import numpy as np
from register import dataset
import utils


class LightGCN(nn.Module):
    def __init__(self, device=None, sparse=True, is_light_gcn=True, use_dcl=True):
        super(LightGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.lr = 0.001
        self.weight_decay = 1e-4
        self.n_layers = 3
        self.num_users = dataset.n_user
        self.num_items = dataset.m_item
        self.adj_shape = self.num_users + self.num_items
        self.latent_dim = 64
        self.f = nn.Sigmoid()
        self._is_sparse = sparse
        self.is_lightgcn = is_light_gcn
        self.use_dcl = use_dcl
        self.adj = nn.Parameter(torch.sparse_coo_tensor(size=(self.adj_shape, self.adj_shape)))

        self.tau_plus = 1e-3
        self.T = 0.07

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.is_lightgcn:
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        else:
            nn.init.normal_(self.embedding_user.weight, std=0.01)
            nn.init.normal_(self.embedding_item.weight, std=0.01)

    def fit(self, adj, d, users, posItems, negItems, users_val, posItems_val, negItems_val):
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

        self._train_with_val(adj, users, posItems, negItems, users_val, posItems_val, negItems_val)

    def getUsersRating(self, adj, users):
        all_users, all_items = self.computer(adj)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def computer(self, adj, delta_u=None, delta_i=None):
        """
        propagate methods for lightGCN
        """
        if delta_i is None and delta_u is None:
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
        else:
            users_emb = self.embedding_user.weight + delta_u
            items_emb = self.embedding_item.weight + delta_i

        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]

        g_droped = adj

        for layer in range(self.n_layers):
            if self._is_sparse:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            else:
                all_emb = torch.mm(g_droped, all_emb)
            embs.append(all_emb)
        if self.is_lightgcn:
            embs = torch.stack(embs, dim=1)
        # print(embs.size())
            light_out = torch.mean(embs, dim=1)
        else:
            # gccf
            light_out = torch.cat(embs, dim=-1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
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
        self.adj = nn.Parameter(adj)
        all_users, all_items = self.computer(self.adj)

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

        else:
            # pos score
            pos_score = torch.exp(torch.sum(users_emb * pos_emb, dim=-1))
            # neg score
            neg_score = torch.exp(torch.sum(users_emb * neg_emb, dim=-1))
            loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2)) / float(len(users))

        return loss, reg_loss

    def _train_with_val(self, adj, users, posItems, negItems, users_val, posItems_val, negItems_val):
        local_path = os.path.abspath(os.path.dirname(os.getcwd()))
        if not os.path.exists(local_path + '/models'):
            os.mkdir(local_path + '/models')
        if not os.path.exists(local_path + '/log'):
            os.mkdir(local_path + '/log')

        if self.is_lightgcn:
            checkpoint_file_name = '{}/models/LightGCN_baseline.ckpt'.format(local_path)
            log_file_name = '{}/log/LightGCN_baseline.log'.format(local_path)
        else:
            checkpoint_file_name = '{}/models/GCCF_baseline.ckpt'.format(local_path)
            log_file_name = '{}/log/GCCF_baseline.log'.format(local_path)
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
                    eval_log.append("Val loss decrease from {} to {}, save model!".format(aver_val_loss, min_val_loss))
                    utils.append_log_to_file(eval_log, i, log_file_name)
                    min_val_loss = aver_val_loss
