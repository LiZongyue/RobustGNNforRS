import torch
import numpy as np
from tqdm import tqdm
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from base_attack import BaseAttack
import utils


class PGDAttack(BaseAttack):
    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True,
                 attack_features=False, device=None, model_name=None, dataset=None):
        super(PGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'
        self.surrogate._is_sparse = False
        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        self.model_name = model_name
        self.dataset = dataset
        if attack_structure:
            assert nnodes is not None, "Please give nnodes="
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes * (nnodes - 1) / 2)))
            self.adj_changes.data.fill_(0)

        self.complementary = None

    def attack(self, ori_adj, perturbations, users, posItems, negItems, num_users, path, ids, flag):
        victim_model = self.surrogate

        # self.sparse_features=sp.issparse(ori_features)
        # print(sp.issparse(ori_adj))
        ori_adj = utils.to_tensor(ori_adj.cpu(), device=self.device)

        victim_model.eval()
        epochs = 200
        users = users.to(self.device)
        posItems = posItems.to(self.device)
        negItems = negItems.to(self.device)

        degree_sequence_start = ori_adj.sum(0)
        current_degree_sequence = degree_sequence_start.clone()

        d_min = 2

        S_d_start = torch.sum(torch.log(degree_sequence_start[degree_sequence_start >= d_min]))
        current_S_d = torch.sum(torch.log(current_degree_sequence[current_degree_sequence >= d_min]))
        n_start = torch.sum(degree_sequence_start >= d_min)
        current_n = torch.sum(current_degree_sequence >= d_min)
        alpha_start = self.compute_alpha(n_start, S_d_start, d_min)

        log_likelihood_orig = self.compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

        for t in tqdm(range(epochs)):
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)
            modified_adj = self.get_modified_adj(ori_adj, num_users)

            deltas = 2 * (1 - modified_adj[0]) - 1
            d_edges_old = current_degree_sequence
            d_edges_new = current_degree_sequence + deltas[:, None]
            new_S_d, new_n = self.update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
            new_alphas = self.compute_alpha(new_n, new_S_d, d_min)
            new_alphas = new_alphas.cpu().detach()
            new_ll = self.compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
            alphas_combined = self.compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
            alphas_combined = alphas_combined.cpu().detach()
            new_ll_combined = self.compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
            new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)

            # Do not consider edges that, if added/removed, would lead to a violation of the
            # likelihood ration Chi_square cutoff value.
            if self.filter_chisquare(new_ratios, cutoff=0.004):
                adj_norm = utils.normalize_adj_tensor(modified_adj)

                for (batch_i,
                     (batch_users,
                      batch_pos,
                      batch_neg)) in enumerate(utils.minibatch(users,
                                                               posItems,
                                                               negItems,
                                                               batch_size=2048)):

                    loss, reg_loss = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg)

                    adj_grad = torch.autograd.grad(loss, self.adj_changes, retain_graph=True)[0]

                    lr = 200 / np.sqrt(t + 1)
                    self.adj_changes.data.add_(lr * adj_grad)

                    self.projection(perturbations)

        self.random_sample(ori_adj, perturbations, users, posItems, negItems, num_users)
        self.modified_adj = self.get_modified_adj(ori_adj, num_users).detach()

        torch.save(self.modified_adj, path.format(self.dataset, self.model_name, ids[flag]))

    def random_sample(self, ori_adj, perturbations, users, posItems, negItems, num_users):
        K = 5
        best_loss = -1000
        victim_model = self.surrogate
        with torch.no_grad():
            s = self.adj_changes.cpu().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                if sampled.sum() > perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))

                users = users.to(self.device)
                posItems = posItems.to(self.device)
                negItems = negItems.to(self.device)
                users, posItems, negItems = utils.shuffle(users, posItems, negItems)

                modified_adj = self.get_modified_adj(ori_adj, num_users)
                adj_norm = utils.normalize_adj_tensor(modified_adj)

                loss_total = 0.

                for (batch_i,
                     (batch_users,
                      batch_pos,
                      batch_neg)) in enumerate(utils.minibatch(users,
                                                               posItems,
                                                               negItems,
                                                               batch_size=2048)):
                    loss, reg_loss = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg)

                    # print(loss)
                    loss_total += loss.cpu().item()

                if best_loss < loss_total:
                    best_loss = loss_total
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def projection(self, perturbations):
        if torch.clamp(self.adj_changes, 0, 1).sum() > perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj, num_users):
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
            # self.complementary=(1-torch.eye(self.nnodes).to(self.device)-ori_adj)-ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes - 1, col=self.nnodes - 1, offset=0).to(self.device)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj
        # modified_adj=m+ori_adj
        modified_adj[:num_users, :num_users] = 0
        modified_adj[num_users:, num_users:] = 0

        return modified_adj

    def bisection(self, a, b, perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - perturbations

        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            if (func(miu) == 0.0):
                break
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        return miu

    @staticmethod
    def compute_alpha(n, S_d, d_min):
        """
        Approximate the alpha of a power law distribution.

        """

        return n / (S_d - n * np.log(d_min - 0.5)) + 1

    @staticmethod
    def compute_log_likelihood(n, alpha, S_d, d_min):
        """
        Compute log likelihood of the powerlaw fit.

        """

        return n * np.log(alpha.detach().cpu()) + n.detach().cpu() * alpha.detach().cpu() * np.log(d_min) + (alpha.detach().cpu() + 1) * S_d.detach().cpu()


    def update_Sx(self, S_old, n_old, d_old, d_new, d_min):
        """
        Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
        a single edge.
        """
        d_new = d_new[0]
        old_in_range = d_old >= d_min
        new_in_range = d_new >= d_min

        d_old_in_range = d_old * old_in_range
        d_new_in_range = d_new * new_in_range

        new_S_d = S_old - torch.log(torch.maximum(d_old_in_range, torch.tensor([1]).to(self.device))).sum() + torch.log(torch.maximum(d_new_in_range, torch.tensor([1]).to(self.device))).sum()
        new_n = n_old - torch.sum(old_in_range) + torch.sum(new_in_range)

        return new_S_d, new_n

    @staticmethod
    def filter_chisquare(ll_ratios, cutoff):
        return cutoff < ll_ratios


class EmbeddingAttack(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cuda:0'):
        super(EmbeddingAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        if attack_structure:
            self.delta_U = Parameter(torch.FloatTensor(self.surrogate.num_users, self.surrogate.latent_dim))
            self.delta_I = Parameter(torch.FloatTensor(self.surrogate.num_items, self.surrogate.latent_dim))

            self.delta_U.data.fill_(0)
            self.delta_I.data.fill_(0)

    def attack(self, ori_adj, eps, users, posItems, negItems, num_users, path, ids, flag):
        victim_model = self.surrogate

        adj_norm = utils.normalize_adj_tensor(utils.to_tensor(ori_adj.cpu(), device=self.device))

        U_delta_adv = torch.zeros(self.surrogate.num_users, self.surrogate.latent_dim).to(self.device)
        I_delta_adv = torch.zeros(self.surrogate.num_items, self.surrogate.latent_dim).to(self.device)

        victim_model.eval()

        epochs = 100
        for t in tqdm(range(epochs)):
            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            for (batch_i,
                 (batch_users,
                  batch_pos,
                  batch_neg)) in enumerate(utils.minibatch(users,
                                                           posItems,
                                                           negItems,
                                                           batch_size=2048)):
                loss, _ = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg, self.delta_U, self.delta_I)

                U_delta_adv += torch.autograd.grad(loss, self.delta_U, retain_graph=True)[0]
                I_delta_adv += torch.autograd.grad(loss, self.delta_I, retain_graph=True)[0]

        U_delta_adv = eps * nn.functional.normalize(U_delta_adv, dim=0)
        I_delta_adv = eps * nn.functional.normalize(I_delta_adv, dim=0)

        victim_model.embedding_user.weight.data.copy_(victim_model.embedding_user.weight + U_delta_adv)
        victim_model.embedding_item.weight.data.copy_(victim_model.embedding_item.weight + I_delta_adv)

        torch.save(victim_model.state_dict(), path.format(ids[flag]))
