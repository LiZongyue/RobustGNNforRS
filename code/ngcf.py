import torch
from torch import nn, optim
import numpy as np
from register import dataset
import utils
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, device=None, sparse=True):
        super(NGCF, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.lr = 0.001
        self.weight_decay = 1e-4
        self.n_layers = 2
        self.num_users = dataset.n_user
        self.num_items = dataset.m_item
        self.latent_dim = 64
        self.f = nn.Sigmoid()
        self._is_sparse = sparse

        self.tau_plus = 1e-3
        self.T = 0.07

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        weight_dict = nn.ParameterDict()

        for k in range(self.n_layers):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(self.latent_dim,
                                                                                    self.latent_dim)))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, self.latent_dim)))})

            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(self.latent_dim,
                                                                                    self.latent_dim)))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, self.latent_dim)))})

        return weight_dict

    def fit(self, adj, users, posItems, negItems, x):
        if self._is_sparse:
            if type(adj) is not torch.Tensor:
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
                adj = utils.to_tensor(adj_norm, device=self.device)
            else:
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
                adj = adj_norm.to(self.device)
            # self.adj=adj
        else:
            if type(adj) is not torch.Tensor:
                adj_norm = utils.normalize_adj_tensor(adj)
                adj = utils.to_tensor(adj_norm, device=self.device)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
                adj = adj_norm.to(self.device)

        self._train_without_val(adj, users, posItems, negItems, x)

    def getUsersRating(self, adj, users):
        all_users, all_items = self.computer(adj)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def computer(self, adj, x=None):
        # TODO: override lightGCN here
        """
        propagate methods for lightGCN
        """

        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight

        # all_emb = torch.cat([users_emb, items_emb])

        # embs = [all_emb]  # list for append

        g_droped = adj

        users, items = None, None

        if x is not None:
            self.init_weight()
            all_emb = x

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

                all_emb += [norm_embeddings]

            all_emb = torch.cat(all_emb, 1)
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

    def getEmbedding(self, adj, users, pos_items, x):
        """
        query from GROC means that we want to push adj into computational graph
        """

        all_users, all_items = self.computer(adj, x)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        # neg_emb = all_items[neg_items]
        users_emb_ego = x(users)
        pos_emb_ego = x(pos_items)
        # neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, users_emb_ego, pos_emb_ego

    def get_negative_mask_1(self, batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            # negative_mask[i,i+batch_size]=0

        # negative_mask=torch.cat((negative_mask,negative_mask),0)
        return negative_mask

    def bpr_loss(self, adj, users, poss, neg, x):
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

        (users_emb, pos_emb, userEmb0, posEmb0) = self.getEmbedding(adj, users.long(), poss.long(), x)
        # pos_emb_old=pos_emb
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

        # loss=(-torch.log(pos/(pos+Ng))).mean()#+self.lambda_g*(users_emb-pos_emb).norm(2).pow(2)
        loss = (-torch.log(pos / (pos + Ng))).mean()

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2)) / float(len(users))

        return loss, reg_loss

    def _train_without_val(self, adj, users, posItems, negItems, x):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(100):
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
                loss, reg_loss = self.bpr_loss(adj, batch_users, batch_pos, batch_neg, x)
                reg_loss = reg_loss * self.weight_decay
                loss = loss + reg_loss

                loss.backward()
                optimizer.step()

                aver_loss += loss.cpu().item()
            aver_loss = aver_loss / total_batch
            if i % 10 == 0:
                print(aver_loss)

        self.eval()

        # users = users.to(self.device)
        # posItems = posItems.to(self.device)
        # negItems = negItems.to(self.device)
        # users, posItems, negItems = utils.shuffle(users, posItems, negItems)
        output = self.forward(adj, users, posItems)
        self.output = output