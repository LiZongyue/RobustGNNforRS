from datetime import datetime
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from utils import scheduler_groc
from utils_attack import attack_model
import utils
import torch.nn.functional as F
from IntegratedGradient import IntegratedGradients
from GraphContrastiveLoss import ori_gcl_computing


class GROC_loss(nn.Module):
    def __init__(self, ori_model, ori_adj, d_mtr, adj_with_2_hops, args, pgd_model=None):
        super(GROC_loss, self).__init__()
        self.ori_adj = ori_adj
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ori_model = ori_model
        self.d_mtr = d_mtr
        self.adj_with_2_hops = adj_with_2_hops
        self.args = args
        self.num_users = self.ori_model.num_users
        self.num_items = self.ori_model.num_items
        self.pgd_model = pgd_model
        if self.args.use_IntegratedGradient:
            self.integrated_gradient = IntegratedGradients(self.ori_model, self.args, self.device, sparse=True)

    def get_embed_groc(self, trn_model, modified_adj, users, items):

        adj_norm = utils.normalize_adj_tensor(modified_adj, sparse=True)
        modified_adj = adj_norm.to(self.device)

        del adj_norm
        gc.collect()  # garbage collection of passed-in tensor

        (users_emb, item_emb, _, _) = trn_model.getEmbedding(modified_adj, users.long(), items.long(), query_groc=True)

        users_emb = nn.functional.normalize(users_emb, dim=1)
        item_emb = nn.functional.normalize(item_emb, dim=1)

        return torch.cat([users_emb, item_emb])

    def groc_loss_vec(self, trn_model, modified_adj_a, modified_adj_b, users, items):
        batch_emb_a = self.get_embed_groc(trn_model, modified_adj_a, users, items)
        batch_emb_b = self.get_embed_groc(trn_model, modified_adj_b, users, items)

        contrastive_similarity = torch.exp(torch.sum(batch_emb_a * batch_emb_b, dim=-1) / self.args.T_groc)

        # contrastive_similarity size： [batch_size,]
        self_neg_similarity_matrix = torch.matmul(batch_emb_a, batch_emb_a.t().contiguous())  # tau_1(v) * tau_2(v)
        contrastive_neg_similarity_matrix = torch.matmul(batch_emb_a, batch_emb_b.t().contiguous())
        # tau_1(v) * tau2(neg), neg includes v itself, will be masked below
        # self_neg_contrastive_similarity_matrix size： [batch_size, batch_size]

        # mask diagonal
        mask = torch.eye(batch_emb_b.size(0), batch_emb_b.size(0)).bool().to(self.device)
        # tensor mask with diagonal all True others all False
        self_neg_similarity_matrix.masked_fill_(mask, 0)
        contrastive_neg_similarity_matrix.masked_fill_(mask, 0)
        # concatenates tau_1(v) * tau_2(v) with 0-diagonal and tau_1(v) * tau2(neg) with 0-diagonal in row
        # we mask 2 diagonal out because we don't want to get the similarity of an embedding with itself
        neg_contrastive_similarity_matrix = \
            torch.cat([self_neg_similarity_matrix, contrastive_neg_similarity_matrix], -1)
        # sum the matrix up by row
        neg_contrastive_similarity = torch.sum(torch.exp(neg_contrastive_similarity_matrix) / self.args.T_groc, 1)

        loss_vec = -torch.log(contrastive_similarity / (contrastive_similarity + neg_contrastive_similarity))

        return loss_vec

    def groc_loss(self, trn_model, modified_adj_a, modified_adj_b, users, items):
        loss_vec_a = self.groc_loss_vec(trn_model, modified_adj_a, modified_adj_b, users, items)
        loss_vec_b = self.groc_loss_vec(trn_model, modified_adj_b, modified_adj_a, users, items)

        return torch.sum(torch.add(loss_vec_a, loss_vec_b)) / (2 * loss_vec_a.size(0))

    def get_modified_adj_for_insert(self, batch_nodes, adj_with_2_hops):
        """
        reset flag is a flag that indicate the adj will insert edges(flag==False, do sum) or set the adj back to original adj
        """
        # use one-hot embedding matrix to index 2 adj matrix(1. adj with 2 hops, 2. original adj) and subtract the
        # result to see, where to insert new edges (For one batch)
        i = torch.stack((batch_nodes, batch_nodes))
        v = torch.ones(i.shape[1]).to(self.device)
        batch_nodes_in_matrix = torch.sparse_coo_tensor(i, v, adj_with_2_hops.shape).to(self.device)

        where_to_insert = (torch.sparse.mm(batch_nodes_in_matrix, adj_with_2_hops) -
                           torch.sparse.mm(batch_nodes_in_matrix, self.ori_adj)).to(self.device)

        num_insert = torch.sparse.sum(where_to_insert)

        where_to_insert = where_to_insert + where_to_insert.t()

        adj_with_insert = self.ori_adj + where_to_insert / num_insert

        return adj_with_insert, num_insert

    def get_modified_adj_with_insert_and_remove_by_gradient(self, remove_prob, insert_prob, batch_users_unique,
                                                            edge_gradient, adj_with_insert, num_insert):
        i = torch.stack((batch_users_unique, batch_users_unique))
        v = torch.ones(i.shape[1]).to(self.device)
        batch_nodes_in_matrix = torch.sparse_coo_tensor(i, v, self.ori_adj.shape).to(self.device)

        ori_adj_ind = self.ori_adj.coalesce().indices()
        k_remove = int(remove_prob * torch.sparse.sum(torch.sparse.mm(batch_nodes_in_matrix, self.ori_adj)))
        # k_insert = int(insert_prob * len(batch_users_unique) * (len(batch_users_unique) - 1) / 2)
        k_insert = int(insert_prob * num_insert)

        # filter added weighted edges, use element-wise multiplication (.mul() for sparse tensor)

        edge_gradient_matrix = torch.sparse.mm(batch_nodes_in_matrix, edge_gradient).mul(self.ori_adj)

        # only remove edges that are related to the current batch
        # according to gradient value, find out edges indices that have min. gradients
        edge_gradient_batch = edge_gradient_matrix.coalesce().values()
        _, ind_rm = torch.topk(edge_gradient_batch, k_remove, largest=False)

        # mask generation
        mask_rm = torch.ones(ori_adj_ind.shape[1]).bool().to(self.device)
        mask_rm[ind_rm] = False

        edge_gradient_ir = torch.sparse.mm(batch_nodes_in_matrix, edge_gradient).mul(adj_with_insert - self.ori_adj)
        _, indices_ir = torch.topk(edge_gradient_ir.coalesce().values(), k_insert)

        ind_rm_ir = edge_gradient_ir.coalesce().indices()[:, indices_ir]
        ind_rm_ir = torch.cat((self.ori_adj.coalesce().indices()[:, mask_rm], ind_rm_ir), -1)
        val_rm_ir = torch.ones(ind_rm_ir.shape[1]).to(self.device)

        adj_insert_remove = torch.sparse_coo_tensor(ind_rm_ir, val_rm_ir, self.ori_adj.shape).to(self.device)

        return adj_insert_remove

    def attack_adjs(self, adj_a, adj_b, perturbations, users, posItems, negItems):
        modified_adj_a = attack_model(self.ori_model, adj_a, perturbations, self.args.path_modified_adj,
                                      self.args.modified_adj_name_with_rdm_ptb_a, self.args.modified_adj_id,
                                      users, posItems, negItems, self.ori_model.num_users, self.device)

        modified_adj_b = attack_model(self.ori_model, adj_b, perturbations, self.args.path_modified_adj,
                                      self.args.modified_adj_name_with_rdm_ptb_b, self.args.modified_adj_id,
                                      users, posItems, negItems, self.ori_model.num_users, self.device)

        try:
            print("modified adjacency matrix are not same:", (modified_adj_a == modified_adj_b).all())
        except AttributeError:
            print("2 modified adjacency matrix are same. Check your perturbation value")

        return modified_adj_a, modified_adj_b

    def ori_gcl_computing(self, trn_model, gra1, gra2, users, poss, query_groc=None):
        (user_emb, _, _, _) = trn_model.getEmbedding(self.ori_adj, users.long(), poss.long())

        (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(), query_groc=query_groc)

        users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1)
        (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra2, users.long(), poss.long(), query_groc=query_groc)

        users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1)
        users_dot_12 = torch.bmm(users_emb_perturb_1.unsqueeze(1), users_emb_perturb_2.unsqueeze(2)).squeeze(2)

        users_dot_12 /= self.args.T_groc
        fenzi_12 = torch.exp(users_dot_12).sum(1)

        neg_emb_users_12 = users_emb_perturb_2.unsqueeze(0).repeat(user_emb.size(0), 1, 1)
        neg_dot_12 = torch.bmm(neg_emb_users_12, users_emb_perturb_1.unsqueeze(2)).squeeze(2)
        neg_dot_12 /= self.args.T_groc
        neg_dot_12 = torch.exp(neg_dot_12).sum(1)

        mask_11 = self.get_negative_mask_perturb(users_emb_perturb_1.size(0)).to(self.device)
        neg_dot_11 = torch.exp(torch.mm(users_emb_perturb_1, users_emb_perturb_1.t()) / self.args.T_groc)
        neg_dot_11 = neg_dot_11.masked_select(mask_11).view(users_emb_perturb_1.size(0), -1).sum(1)
        loss_perturb_11 = (-torch.log(fenzi_12 / (neg_dot_11 + neg_dot_12))).mean()

        users_dot_21 = torch.bmm(users_emb_perturb_2.unsqueeze(1), users_emb_perturb_1.unsqueeze(2)).squeeze(2)
        users_dot_21 /= self.args.T_groc
        fenzi_21 = torch.exp(users_dot_21).sum(1)

        neg_emb_users_21 = users_emb_perturb_1.unsqueeze(0).repeat(user_emb.size(0), 1, 1)
        neg_dot_21 = torch.bmm(neg_emb_users_21, users_emb_perturb_2.unsqueeze(2)).squeeze(2)
        neg_dot_21 /= self.args.T_groc
        neg_dot_21 = torch.exp(neg_dot_21).sum(1)

        mask_22 = self.get_negative_mask_perturb(users_emb_perturb_2.size(0)).to(self.device)
        neg_dot_22 = torch.exp(torch.mm(users_emb_perturb_2, users_emb_perturb_2.t()) / self.args.T_groc)
        neg_dot_22 = neg_dot_22.masked_select(mask_22).view(users_emb_perturb_2.size(0), -1).sum(1)
        loss_perturb_22 = (-torch.log(fenzi_21 / (neg_dot_22 + neg_dot_21))).mean()

        loss_perturb = loss_perturb_11 + loss_perturb_22

        return loss_perturb

    @staticmethod
    def get_negative_mask_perturb(batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0

        return negative_mask

    def groc_train(self):
        self.ori_model.train()
        embedding_param = []
        adj_param = []
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.ori_model.lr, weight_decay=self.ori_model.weight_decay)

        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, self.num_users + self.num_items, self.args.warmup_steps,
                                       self.args.groc_batch_size,
                                       self.args.groc_epochs)

        all_node_index = torch.arange(0, self.num_users + self.num_items, 1).to(self.device)
        all_node_index = utils.shuffle(all_node_index)

        total_batch = len(all_node_index) // self.args.groc_batch_size + 1

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()
            aver_loss = 0.
            for (batch_i, (batch_all_node)) in \
                    enumerate(utils.minibatch(all_node_index, batch_size=self.args.groc_batch_size)):
                user_filter = (batch_all_node < self.num_users).to(self.device)
                batch_users = torch.masked_select(batch_all_node, user_filter).to(self.device)
                batch_items = torch.sub(torch.masked_select(batch_all_node, ~user_filter), self.num_users).to(
                    self.device)
                adj_with_insert = self.get_modified_adj_for_insert(batch_all_node)  # 2 views are same

                loss_for_grad = self.groc_loss(self.ori_model, adj_with_insert, adj_with_insert, batch_users,
                                               batch_items)

                # remove index of diagonal

                edge_gradient = torch.autograd.grad(loss_for_grad, self.ori_model.adj, retain_graph=True)[0]

                adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_1,
                                                                                               self.args.remove_prob_1,
                                                                                               batch_all_node,
                                                                                               edge_gradient,
                                                                                               adj_with_insert)
                adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_2,
                                                                                               self.args.remove_prob_2,
                                                                                               batch_all_node,
                                                                                               edge_gradient,
                                                                                               adj_with_insert)

                loss = self.groc_loss(self.ori_model, adj_insert_remove_1, adj_insert_remove_2, batch_users,
                                      batch_items)
                loss.backward()
                optimizer.step()
                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()

            aver_loss = aver_loss / total_batch

            if i % 10 == 0:
                print("GROC Loss: ", aver_loss)

    def bpr_with_dcl(self, data_len_, modified_adj_a, modified_adj_b, users, posItems, negItems):
        self.ori_model.train()
        optimizer = optim.Adam(self.ori_model.parameters(), lr=self.ori_model.lr,
                               weight_decay=self.ori_model.weight_decay)
        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.batch_size,
                                       self.args.groc_epochs)

        total_batch = len(users) // self.args.batch_size + 1

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()

            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss = 0.
            aver_bpr_loss = 0.
            aver_dcl_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=10)):
                self.ori_adj = utils.normalize_adj_tensor(self.ori_adj, sparse=True)
                modified_adj_a = utils.normalize_adj_tensor(modified_adj_a, sparse=True)
                modified_adj_b = utils.normalize_adj_tensor(modified_adj_b, sparse=True)

                gc.collect()

                bpr_loss, reg_loss = self.ori_model.bpr_loss(self.ori_adj, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.ori_model.weight_decay
                dcl_loss = self.groc_loss(self.ori_model, modified_adj_a, modified_adj_b, batch_users, batch_pos)
                loss = self.args.loss_weight_bpr * bpr_loss + reg_loss + (1 - self.args.loss_weight_bpr) * dcl_loss

                loss.backward()
                optimizer.step()
                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()
                aver_bpr_loss += bpr_loss.cpu().item()
                aver_dcl_loss += dcl_loss.cpu().item()

            aver_loss = aver_loss / total_batch
            aver_bpr_loss = aver_bpr_loss / total_batch
            aver_dcl_loss = aver_dcl_loss / total_batch

            if i % 10 == 0:
                print("GROC Loss: ", aver_loss)
                print("BPR Loss: ", aver_bpr_loss)
                print("DCL Loss: ", aver_dcl_loss)

    def groc_train_with_bpr(self, data_len_, users, posItems, negItems, perturbations):
        self.ori_model.train()
        embedding_param = []
        adj_param = []
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.ori_model.lr, weight_decay=self.ori_model.weight_decay)

        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.groc_batch_size,
                                       self.args.groc_epochs)

        total_batch = len(users) // self.args.batch_size + 1
        ori_adj_sparse = utils.normalize_adj_tensor(self.ori_adj, self.d_mtr, sparse=True).to(self.device)  # for bpr loss

        tril_adj_index = torch.tril_indices(row=len(self.ori_adj) - 1, col=len(self.ori_adj) - 1, offset=0)
        tril_adj_index = tril_adj_index.to(self.device)
        tril_adj_index_0 = tril_adj_index[0]
        tril_adj_index_1 = tril_adj_index[1]

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()

            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss = 0.
            aver_bpr_loss = 0.
            aver_groc_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.batch_size)):
                # batch_items = utils.shuffle(torch.cat((batch_pos, batch_neg))).to(self.device)

                batch_users_unique = batch_users.unique()  # only select 10 anchor nodes for adj_edge insertion

                if not self.args.use_groc_pgd:
                    # perturb adj inside training. Insert value (1 / num_inserted) to ori_adj. Where to insert, check GROC
                    adj_with_insert = self.get_modified_adj_for_insert(batch_users_unique,
                                                                       self.adj_with_2_hops)  # 2 views are same

                    # batch_users_groc = batch_all_node[batch_all_node < self.num_users]
                    # batch_items = batch_all_node[batch_all_node >= self.num_users] - self.num_users

                    # Normalize perturbed adj (with insertion)
                    adj_for_loss_gradient = utils.normalize_adj_tensor(adj_with_insert.to_sparse(), self.d_mtr,
                                                                       sparse=True)

                    if not self.args.use_IntegratedGradient:
                        loss_for_grad = ori_gcl_computing(self.ori_adj, self.ori_model, adj_for_loss_gradient,
                                                          adj_for_loss_gradient, batch_users, batch_pos, self.args,
                                                          self.device, True, self.args.mask_prob_1,
                                                          self.args.mask_prob_2, query_groc=True)

                        edge_gradient = torch.autograd.grad(loss_for_grad, self.ori_model.adj, retain_graph=True)[0]

                    else:
                        edge_gradient = self.integrated_gradient.get_integrated_gradient(adj_for_loss_gradient,
                                                                                         self.ori_model, self.ori_adj,
                                                                                         batch_users, batch_pos)
                    del adj_for_loss_gradient
                    gc.collect()

                    adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_1,
                                                                                                   self.args.remove_prob_1,
                                                                                                   batch_users_unique,
                                                                                                   edge_gradient,
                                                                                                   adj_with_insert,
                                                                                                   tril_adj_index_0,
                                                                                                   tril_adj_index_1)

                    adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_2,
                                                                                                   self.args.remove_prob_2,
                                                                                                   batch_users_unique,
                                                                                                   edge_gradient,
                                                                                                   adj_with_insert,
                                                                                                   tril_adj_index_0,
                                                                                                   tril_adj_index_1)

                    del adj_with_insert

                    adj_norm_1 = utils.normalize_adj_tensor(adj_insert_remove_1.to_sparse(), self.d_mtr, sparse=True)
                    adj_norm_2 = utils.normalize_adj_tensor(adj_insert_remove_2.to_sparse(), self.d_mtr, sparse=True)

                    del adj_insert_remove_1
                    del adj_insert_remove_2

                else:
                    adj_pgd_1 = self.pgd_model.attack_per_batch(self.ori_adj, perturbations, batch_users,
                                                                batch_pos, batch_neg, self.num_users)
                    adj_pgd_2 = self.pgd_model.attack_per_batch(self.ori_adj, perturbations, batch_users,
                                                                batch_pos, batch_neg, self.num_users)

                    adj_norm_1 = utils.normalize_adj_tensor(adj_pgd_1.to_sparse(), self.d_mtr, sparse=True)
                    adj_norm_2 = utils.normalize_adj_tensor(adj_pgd_2.to_sparse(), self.d_mtr, sparse=True)

                    gc.collect()

                groc_loss = ori_gcl_computing(self.ori_adj, self.ori_model, adj_norm_1, adj_norm_2, batch_users,
                                              batch_pos, self.args, self.device, mask_1=self.args.mask_prob_1,
                                              mask_2=self.args.mask_prob_2)

                del adj_norm_1
                del adj_norm_2

                bpr_loss, reg_loss = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.ori_model.weight_decay

                loss = self.args.loss_weight_bpr * (bpr_loss + reg_loss) + (1 - self.args.loss_weight_bpr) * groc_loss

                loss.backward()

                optimizer.step()

                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()
                aver_bpr_loss += bpr_loss.cpu().item()
                aver_groc_loss += groc_loss.cpu().item()

            aver_loss = aver_loss / total_batch
            aver_bpr_loss = aver_bpr_loss / total_batch
            aver_dcl_loss = aver_groc_loss / total_batch

            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("=======================")

            print("Epoch: {}:".format(i))
            print("GROC Loss: ", aver_loss)
            print("BPR Loss: ", aver_bpr_loss)
            print("DCL Loss: ", aver_dcl_loss)
            print("=========================")

    def ori_gcl_train_with_bpr(self, gra1, gra2, data_len_, users, posItems, negItems):
        self.ori_adj = utils.normalize_adj_tensor(self.ori_adj, sparse=True)
        gra1 = utils.normalize_adj_tensor(gra1, sparse=True)
        gra2 = utils.normalize_adj_tensor(gra2, sparse=True)

        gc.collect()

        self.ori_model.train()
        embedding_param = []
        adj_param = []
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.ori_model.lr, weight_decay=self.ori_model.weight_decay)

        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.groc_batch_size,
                                       self.args.groc_epochs)

        total_batch = len(users) // self.args.batch_size + 1

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()
            # data
            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss = 0.
            aver_bpr_loss = 0.
            aver_groc_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.batch_size)):
                tic = time.time()
                # graph contrastive loss for 2 views of graph
                gcl = self.ori_gcl_computing(self.ori_model, gra1, gra2, batch_users, batch_pos)
                toc = time.time()

                print("time for gcl calculation:", toc - tic)

                bpr_loss, reg_loss = self.ori_model.bpr_loss(self.ori_adj, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.ori_model.weight_decay

                loss = self.args.loss_weight_bpr * bpr_loss + reg_loss + (1 - self.args.loss_weight_bpr) * gcl
                loss.backward()
                optimizer.step()

                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()
                aver_bpr_loss += bpr_loss.cpu().item()
                aver_groc_loss += gcl.cpu().item()

            aver_loss = aver_loss / total_batch
            aver_bpr_loss = aver_bpr_loss / total_batch
            aver_dcl_loss = aver_groc_loss / total_batch

            print("Epoch: {}".format(i))
            print("GROC Loss: ", aver_loss)
            print("BPR Loss: ", aver_bpr_loss)
            print("DCL Loss: ", aver_dcl_loss)

    def groc_train_with_bpr_cat(self, data_len_, users, posItems, negItems):
        self.ori_model.train()
        embedding_param = []
        adj_param = []
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.ori_model.lr, weight_decay=self.ori_model.weight_decay)

        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.groc_batch_size,
                                       self.args.groc_epochs)

        total_batch = len(users) // self.args.batch_size + 1
        ori_adj_sparse = utils.normalize_adj_tensor(self.ori_adj).to_sparse().to(self.device)  # for bpr loss

        adj_with_2_hops = self.contruct_adj_after_n_hops()  # dense

        tril_adj_index = torch.tril_indices(row=len(self.ori_adj) - 1, col=len(self.ori_adj) - 1, offset=0)
        tril_adj_index = tril_adj_index.to(self.device)
        tril_adj_index_0 = tril_adj_index[0]
        tril_adj_index_1 = tril_adj_index[1]

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()

            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss = 0.
            aver_bpr_loss = 0.
            aver_groc_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.batch_size)):

                batch_users_unique = batch_users.unique()  # only select 10 anchor nodes for adj_edge insertion

                adj_with_insert = self.get_modified_adj_for_insert(batch_users_unique,
                                                                   adj_with_2_hops)  # 2 views are same

                mask_1 = (torch.FloatTensor(self.ori_model.latent_dim).uniform_() < self.args.mask_prob_1) \
                    .to(self.device)
                mask_2 = (torch.FloatTensor(self.ori_model.latent_dim).uniform_() < self.args.mask_prob_2) \
                    .to(self.device)

                # batch_users_groc = batch_all_node[batch_all_node < self.num_users]
                # batch_items = batch_all_node[batch_all_node >= self.num_users] - self.num_users

                adj_for_loss_gradient = utils.normalize_adj_tensor(adj_with_insert.to_sparse(), self.d_mtr, sparse=True)

                if not self.args.use_IntegratedGradient:
                    gcl_grad = ori_gcl_computing(self.ori_adj, self.ori_model, adj_for_loss_gradient,
                                                 adj_for_loss_gradient, batch_users, batch_pos, self.args, self.device,
                                                 True, mask_1, mask_2, query_groc=True)
                    bpr_loss_grad, reg_loss_grad = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos,
                                                                           batch_neg)
                    reg_loss_grad = reg_loss_grad * self.ori_model.weight_decay

                    loss_for_grad = self.args.loss_weight_bpr * bpr_loss_grad + reg_loss_grad + \
                                    (1 - self.args.loss_weight_bpr) * gcl_grad

                    edge_gradient = torch.autograd.grad(loss_for_grad, self.ori_model.adj, retain_graph=True)[0]

                else:
                    edge_gradient = self.integrated_gradient.get_integrated_gradient(adj_for_loss_gradient,
                                                                                     self.ori_model, self.ori_adj,
                                                                                     batch_users, batch_pos,
                                                                                     mask_1, mask_2)
                del adj_for_loss_gradient
                gc.collect()

                adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_1,
                                                                                               self.args.remove_prob_1,
                                                                                               batch_users_unique,
                                                                                               edge_gradient,
                                                                                               adj_with_insert,
                                                                                               tril_adj_index_0,
                                                                                               tril_adj_index_1)

                adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_2,
                                                                                               self.args.remove_prob_2,
                                                                                               batch_users_unique,
                                                                                               edge_gradient,
                                                                                               adj_with_insert,
                                                                                               tril_adj_index_0,
                                                                                               tril_adj_index_1)

                del adj_with_insert

                adj_norm_1 = utils.normalize_adj_tensor(adj_insert_remove_1.to_sparse(), self.d_mtr, sparse=True)
                adj_norm_2 = utils.normalize_adj_tensor(adj_insert_remove_2.to_sparse(), self.d_mtr, sparse=True)

                groc_loss = ori_gcl_computing(self.ori_adj, self.ori_model, adj_norm_1, adj_norm_2, batch_users,
                                              batch_pos, self.args, self.device, mask_1=mask_1, mask_2=mask_2)

                del adj_insert_remove_1
                del adj_insert_remove_2
                del adj_norm_1
                del adj_norm_2

                bpr_loss, reg_loss = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.ori_model.weight_decay

                loss = self.args.loss_weight_bpr * bpr_loss + reg_loss + (1 - self.args.loss_weight_bpr) * groc_loss

                loss.backward()

                optimizer.step()

                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()
                aver_bpr_loss += bpr_loss.cpu().item()
                aver_groc_loss += groc_loss.cpu().item()

            aver_loss = aver_loss / total_batch
            aver_bpr_loss = aver_bpr_loss / total_batch
            aver_dcl_loss = aver_groc_loss / total_batch

            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("=======================")

            print("Epoch: {}:".format(i))
            print("GROC Loss: ", aver_loss)
            print("BPR Loss: ", aver_bpr_loss)
            print("DCL Loss: ", aver_dcl_loss)
            print("=========================")

    def optimizer_init(self, adj_param, embedding_param):
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.ori_model.lr, weight_decay=self.ori_model.weight_decay)

        return optimizer

    def groc_train_with_bpr_sparse(self, data_len_, users, posItems, negItems, users_val, posItems_val, negItems_val):
        checkpoint_file_name = 'Robustness-1/models/{model_file}.ckpt'.format(model_file=self.args.save_to)
        log_file_name = 'Robustness-1/log/{model_file}.log'.format(model_file=self.args.save_to)

        self.ori_model.train()
        embedding_param = []
        adj_param = []
        scheduler = None

        optimizer = self.optimizer_init(adj_param, embedding_param)

        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.groc_batch_size,
                                       self.args.groc_epochs)

        total_batch = len(users) // self.args.batch_size + 1
        total_val_batch = len(users_val) // self.args.val_batch_size + 1
        ori_adj_sparse = utils.normalize_adj_tensor(self.ori_adj, self.d_mtr, sparse=True).to(self.device)  # for bpr loss

        for i in range(self.args.groc_epochs):
            eval_log = []
            optimizer.zero_grad()
            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss, aver_bpr_loss, aver_groc_loss = 0., 0., 0.
            val_aver_loss, val_aver_bpr_loss, val_aver_groc_loss = 0., 0., 0.
            val_max_loss, val_max_bpr_loss, val_max_groc_loss = float('Inf'), float('Inf'), float('Inf')

            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.batch_size)):
                loss, bpr_loss, groc_loss = \
                    self.groc_train_with_bpr_one_batch(batch_users, batch_pos, batch_neg, ori_adj_sparse, optimizer, scheduler)

                aver_loss += loss.cpu().item()
                aver_bpr_loss += bpr_loss.cpu().item()
                aver_groc_loss += groc_loss.cpu().item()

            aver_loss = aver_loss / total_batch
            aver_bpr_loss = aver_bpr_loss / total_batch
            aver_groc_loss = aver_groc_loss / total_batch
            del loss, bpr_loss, groc_loss
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            gc.collect()
            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("=======================")

            print("Epoch: {}:".format(i))
            print("Train GROC Loss: ", aver_loss)
            print("Train BPR Loss: ", aver_bpr_loss)
            print("Train DCL Loss: ", aver_groc_loss)
            print("=========================")

            if (i + 1) % self.args.valid_freq == 0:
                print('Starting validation')
                eval_log.append("Valid Epoch: {}:".format(i))

                users_val = users_val.to(self.device)
                posItems_val = posItems_val.to(self.device)
                negItems_val = negItems_val.to(self.device)
                users_val, posItems_val, negItems_val = utils.shuffle(users_val, posItems_val, negItems_val)
                for (batch_i, (batch_users, batch_pos, batch_neg)) \
                        in enumerate(utils.minibatch(users_val, posItems_val, negItems_val, batch_size=self.args.val_batch_size)):
                    val_loss, val_bpr_loss, val_dcl_loss = \
                        self.groc_val_with_bpr_one_batch(batch_users, batch_pos, batch_neg, ori_adj_sparse)
                    val_aver_loss += val_loss.cpu().item()
                    val_aver_bpr_loss += val_bpr_loss.cpu().item()
                    val_aver_groc_loss += val_dcl_loss.cpu().item()

                val_aver_loss = val_aver_loss / total_val_batch
                val_aver_bpr_loss = val_aver_bpr_loss / total_val_batch
                val_aver_groc_loss = val_aver_groc_loss / total_val_batch
                save = False

                if val_max_groc_loss > val_aver_groc_loss and val_max_loss > val_aver_loss and val_max_bpr_loss > val_aver_bpr_loss:
                    save = True
                    eval_log.append(f'Valid loss score decreased from {val_max_loss} to {val_aver_loss}')
                    eval_log.append(f'Valid bpr loss score decreased from {val_max_bpr_loss} to {val_aver_bpr_loss}')
                    eval_log.append(f'Valid loss score decreased from {val_max_groc_loss} to {val_aver_groc_loss}')

                    val_max_groc_loss = val_aver_groc_loss
                    val_max_loss = val_aver_loss
                    val_max_bpr_loss = val_aver_bpr_loss

                if save:
                    utils.save_model(self.ori_model, checkpoint_file_name)

                now = datetime.now()

                current_time = now.strftime("%H:%M:%S")
                eval_log.append("Current Time = {}".format(current_time))
                eval_log.append("=======================")
                eval_log.append("Valid GROC Loss: {}".format(val_aver_loss))
                eval_log.append("Valid BPR Loss: {}".format(val_aver_bpr_loss))
                eval_log.append("Valid DCL Loss: {}".format(val_aver_groc_loss))
                eval_log.append("=========================")

                utils.append_log_to_file(eval_log, i, log_file_name)

    def groc_val_with_bpr(self, users, posItems, negItems, i):
        self.ori_model.eval()
        total_batch = len(users) // self.args.batch_size + 1
        ori_adj_sparse = utils.normalize_adj_tensor(self.ori_adj, self.d_mtr, sparse=True).to(self.device)  # bpr loss

        users = users.to(self.device)
        posItems = posItems.to(self.device)
        negItems = negItems.to(self.device)
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)

        aver_loss = 0.
        aver_bpr_loss = 0.
        aver_groc_loss = 0.
        for (batch_i, (batch_users, batch_pos, batch_neg)) \
                in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.val_batch_size)):
            loss, bpr_loss, groc_loss = self.groc_val_with_bpr_one_batch(batch_users, batch_pos, batch_neg, ori_adj_sparse)

            aver_loss += loss.cpu().item()
            aver_bpr_loss += bpr_loss.cpu().item()
            aver_groc_loss += groc_loss.cpu().item()

        aver_loss = aver_loss / total_batch
        aver_bpr_loss = aver_bpr_loss / total_batch
        aver_dcl_loss = aver_groc_loss / total_batch

        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        print("=======================")

        print("Epoch: {}:".format(i))
        print("VAL GROC Loss: ", aver_loss)
        print("VAL BPR Loss: ", aver_bpr_loss)
        print("VAL DCL Loss: ", aver_dcl_loss)
        print("=========================")

        return aver_loss, aver_bpr_loss, aver_dcl_loss

    def groc_train_with_bpr_one_batch(self, batch_users, batch_pos, batch_neg, ori_adj_sparse, optimizer, scheduler):
        loss, bpr_loss, groc_loss = self.forward_pass_groc_with_bpr(batch_users, batch_pos, batch_neg, ori_adj_sparse)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        return loss, bpr_loss, groc_loss

    def groc_val_with_bpr_one_batch(self, batch_users, batch_pos, batch_neg, ori_adj_sparse):
        loss, bpr_loss, groc_loss = self.forward_pass_groc_with_bpr(batch_users, batch_pos, batch_neg, ori_adj_sparse)

        return loss, bpr_loss, groc_loss

    def forward_pass_groc_with_bpr(self, batch_users, batch_pos, batch_neg, ori_adj_sparse):
        batch_users_unique = batch_users.unique()  # only select 10 anchor nodes for adj_edge insertion

        # perturb adj inside training. Insert value (1 / num_inserted) to ori_adj. Where to insert, check GROC

        adj_with_insert, num_insert = self.get_modified_adj_for_insert(batch_users_unique,
                                                           self.adj_with_2_hops)  # 2 views are same

        # Normalize perturbed adj (with insertion)
        adj_for_loss_gradient = utils.normalize_adj_tensor(adj_with_insert, self.d_mtr, sparse=True)
        adj_for_loss_gradient.requires_grad = True
        loss_for_grad = ori_gcl_computing(self.ori_adj, self.ori_model, adj_for_loss_gradient,
                                          adj_for_loss_gradient, batch_users, batch_pos, self.args,
                                          self.device, True, self.args.mask_prob_1,
                                          self.args.mask_prob_2, query_groc=True)

        edge_gradient = torch.autograd.grad(loss_for_grad, adj_for_loss_gradient, retain_graph=True)[0]

        adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_1,
                                                                                       self.args.remove_prob_1,
                                                                                       batch_users_unique,
                                                                                       edge_gradient,
                                                                                       adj_with_insert, num_insert)

        adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_2,
                                                                                       self.args.remove_prob_2,
                                                                                       batch_users_unique,
                                                                                       edge_gradient,
                                                                                       adj_with_insert, num_insert)

        adj_norm_1 = utils.normalize_adj_tensor(adj_insert_remove_1, self.d_mtr, sparse=True)
        adj_norm_2 = utils.normalize_adj_tensor(adj_insert_remove_2, self.d_mtr, sparse=True)

        groc_loss = ori_gcl_computing(self.ori_adj, self.ori_model, adj_norm_1, adj_norm_2, batch_users,
                                      batch_pos, self.args, self.device, mask_1=self.args.mask_prob_1,
                                      mask_2=self.args.mask_prob_2)

        bpr_loss, reg_loss = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg)
        reg_loss = reg_loss * self.ori_model.weight_decay
        loss = self.args.loss_weight_bpr * (bpr_loss + reg_loss) + (1 - self.args.loss_weight_bpr) * groc_loss

        return loss, bpr_loss, groc_loss

    def fit(self):
        pass
