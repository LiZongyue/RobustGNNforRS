from datetime import datetime
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from utils import scheduler_groc
import numpy as np
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

    def get_modified_adj_for_insert(self, batch_nodes, adj_with_2_hops, sparse=True):
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
        if sparse:
            num_insert = torch.sparse.sum(where_to_insert)
        else:
            num_insert = torch.sum(where_to_insert)

        # where_to_insert = where_to_insert + where_to_insert.t()

        adj_with_insert = self.ori_adj + where_to_insert / num_insert

        return adj_with_insert, num_insert

    def get_modified_adj_with_insert_and_remove_by_gradient_sparse(self, remove_prob, insert_prob, batch_users_unique,
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
        edge_gradient_batch = edge_gradient_matrix.values()
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

    def get_modified_adj_with_insert_and_remove_by_gradient(self, insert_prob, remove_prob, batch_users_unique,
                                                            edge_gradient, adj_with_insert, tril_adj_index_0,
                                                            tril_adj_index_1, num_insert):
        i = torch.stack((batch_users_unique, batch_users_unique))
        v = torch.ones(i.shape[1]).to(self.device)
        batch_nodes_in_matrix = torch.sparse_coo_tensor(i, v, self.ori_adj.shape).to(self.device)

        adj_insert_remove = self.ori_adj.clone().to(self.device)

        k_remove = int(remove_prob * self.ori_adj[batch_users_unique].sum())
        k_insert = int(insert_prob * num_insert)

        edge_gradient = edge_gradient.to_dense().to(self.device)
        edge_gradient_remove = self.ori_adj * torch.sparse.mm(batch_nodes_in_matrix, edge_gradient)
        _, i = torch.topk(edge_gradient_remove.flatten(), k_remove, largest=False)
        indices_rm = torch.tensor(np.array(np.unravel_index(i.detach().cpu().numpy(), self.ori_adj.shape)).T).reshape(2, -1).to(self.device)

        adj_insert_remove[indices_rm] = 0.

        edge_gradient_insert = (1 - self.ori_adj) * torch.sparse.mm(batch_nodes_in_matrix, edge_gradient)
        _, i = torch.topk(edge_gradient_insert.flatten(), k_insert)
        indices_ir = torch.tensor(np.array(np.unravel_index(i.detach().cpu().numpy(), self.ori_adj.shape)).T).reshape(2, -1).to(self.device)

        adj_insert_remove[indices_ir] = 1.

        # edge_gradient_remove = \
        #     (self.ori_adj * torch.sparse.mm(batch_nodes_in_matrix, edge_gradient))[tril_adj_index_1, tril_adj_index_0]
        #
        # _, indices_rm = torch.topk(edge_gradient_remove, k_remove, largest=False)
        #
        # low_tril_matrix = adj_insert_remove[tril_adj_index_0, tril_adj_index_1]
        # up_tril_matrix = adj_insert_remove[tril_adj_index_1, tril_adj_index_0]
        # low_tril_matrix[indices_rm] = 0.
        # up_tril_matrix[indices_rm] = 0.
        #
        # # k_insert = int(insert_prob * len(batch_users_unique) * (len(batch_users_unique) - 1) / 2)
        # edge_gradient_insert = (edge_gradient *
        #                         (adj_with_insert - self.ori_adj))[tril_adj_index_0, tril_adj_index_1]
        # _, indices_ir = torch.topk(edge_gradient_insert, k_insert)
        # low_tril_matrix[indices_ir] = 1.
        # up_tril_matrix[indices_ir] = 1.
        #
        # adj_insert_remove[tril_adj_index_0, tril_adj_index_1] = low_tril_matrix
        # adj_insert_remove[tril_adj_index_1, tril_adj_index_0] = up_tril_matrix
        #
        # del low_tril_matrix
        # del up_tril_matrix
        #
        # del edge_gradient
        # del edge_gradient_insert
        # del edge_gradient_remove
        #
        # gc.collect()
        #
        return adj_insert_remove

    @staticmethod
    def get_negative_mask_perturb(batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0

        return negative_mask

    def optimizer_init(self, adj_param, embedding_param):
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.args.lr, weight_decay=self.ori_model.weight_decay)

        return optimizer

    def groc_train_with_bpr_sparse(self, data_len_, users, posItems, negItems, users_val, posItems_val, negItems_val, checkpoint_file_name, log_file_name, adj_rm_1=None, adj_rm_2=None, sparse=True):
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
        ori_adj_sparse = utils.normalize_adj_tensor(self.ori_adj).to_sparse()  # for bpr loss
        val_max_loss, val_max_bpr_loss, val_max_groc_loss = float('Inf'), float('Inf'), float('Inf')
        tril_adj_index = torch.tril_indices(row=len(self.ori_adj), col=len(self.ori_adj), offset=0)
        tril_adj_index = tril_adj_index.to(self.device)
        tril_adj_index_0 = tril_adj_index[0]
        tril_adj_index_1 = tril_adj_index[1]
        for i in range(self.args.groc_epochs):
            eval_log = []
            optimizer.zero_grad()
            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss, aver_bpr_loss, aver_groc_loss = 0., 0., 0.
            val_aver_loss, val_aver_bpr_loss, val_aver_groc_loss = 0., 0., 0.

            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.batch_size)):

                if self.args.train_groc_pipeline:  # GROC training, Towards Robust GNN
                    if self.args.with_bpr:
                        loss, bpr_loss, groc_loss = \
                            self.groc_train_with_bpr_one_batch(batch_users, batch_pos, batch_neg, ori_adj_sparse, optimizer, scheduler, tril_adj_index_0, tril_adj_index_1, sparse)
                    else:
                        loss, bpr_loss, groc_loss = \
                            self.groc_train_without_bpr_one_batch(batch_users, batch_pos, batch_neg, ori_adj_sparse, optimizer, scheduler, tril_adj_index_0, tril_adj_index_1, sparse)
                elif self.args.train_with_bpr_perturb:  # perturb adj with bpr gradient, train as CL for RS
                    loss, bpr_loss, groc_loss = \
                        self.clrs_train_with_gradient_perturb_one_batch(adj_rm_1, adj_rm_2, batch_users, batch_pos, batch_neg, ori_adj_sparse, optimizer, scheduler, sparse, False)
                else:
                    raise Exception("No Training process is running.")
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
                    if self.args.train_groc_pipeline:
                        val_loss, val_bpr_loss, val_dcl_loss = \
                            self.groc_val_with_bpr_one_batch(batch_users, batch_pos, batch_neg, ori_adj_sparse, tril_adj_index_0, tril_adj_index_1, sparse)
                    elif self.args.train_with_bpr_perturb:  # perturb adj with bpr gradient, train as CL for RS
                        val_loss, val_bpr_loss, val_dcl_loss = \
                            self.clrs_train_with_gradient_perturb_one_batch(adj_rm_1, adj_rm_2, batch_users, batch_pos,
                                                                            batch_neg, ori_adj_sparse, optimizer, scheduler,
                                                                            sparse, True)
                    else:
                        raise Exception("No validation process is running.")

                    val_aver_loss += val_loss.cpu().item()
                    val_aver_bpr_loss += val_bpr_loss.cpu().item()
                    val_aver_groc_loss += val_dcl_loss.cpu().item()

                val_aver_loss = val_aver_loss / total_val_batch
                val_aver_bpr_loss = val_aver_bpr_loss / total_val_batch
                val_aver_groc_loss = val_aver_groc_loss / total_val_batch
                save = False

                if val_max_loss > val_aver_loss:
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
                eval_log.append("Valid total Loss: {}".format(val_aver_loss))
                eval_log.append("Valid BPR Loss: {}".format(val_aver_bpr_loss))
                eval_log.append("Valid GROC Loss: {}".format(val_aver_groc_loss))
                eval_log.append("=========================")

                utils.append_log_to_file(eval_log, i, log_file_name)

    def groc_train_with_bpr_one_batch(self, batch_users, batch_pos, batch_neg, ori_adj_sparse, optimizer, scheduler, tril_adj_index_0, tril_adj_index_1, sparse):
        loss, bpr_loss, groc_loss = self.forward_pass_groc_with_bpr(batch_users, batch_pos, batch_neg, ori_adj_sparse, tril_adj_index_0, tril_adj_index_1, sparse)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        return loss, bpr_loss, groc_loss

    def groc_train_without_bpr_one_batch(self, batch_users, batch_pos, batch_neg, ori_adj_sparse, optimizer, scheduler, tril_adj_index_0, tril_adj_index_1, sparse):
        loss, bpr_loss, groc_loss = self.forward_pass_groc_with_bpr(batch_users, batch_pos, batch_neg, ori_adj_sparse, tril_adj_index_0, tril_adj_index_1, sparse)
        groc_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        return loss, bpr_loss, groc_loss

    def clrs_train_with_gradient_perturb_one_batch(self, adj_rm_1, adj_rm_2, batch_users, batch_pos, batch_neg, ori_adj_sparse, optimizer, scheduler, sparse, val):
        loss, bpr_loss, groc_loss = self.forward_pass_clrs_with_gradient_perturb(adj_rm_1, adj_rm_2, batch_users, batch_pos, batch_neg, ori_adj_sparse, sparse, val)
        if not val:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        return loss, bpr_loss, groc_loss

    def groc_val_with_bpr_one_batch(self, batch_users, batch_pos, batch_neg, ori_adj_sparse, tril_adj_index_0, tril_adj_index_1, sparse):
        loss, bpr_loss, groc_loss = self.forward_pass_groc_with_bpr(batch_users, batch_pos, batch_neg, ori_adj_sparse, tril_adj_index_0, tril_adj_index_1, sparse, val=True)

        return loss, bpr_loss, groc_loss

    def forward_pass_clrs_with_gradient_perturb(self, adj_rm_1, adj_rm_2, batch_users, batch_pos, batch_neg, ori_adj_sparse, sparse, val):
        self.ori_model.requires_grad_(True)

        # Normalize perturbed adj (with insertion)
        if sparse:
            adj_rm_norm_1 = utils.normalize_adj_tensor(adj_rm_1, self.d_mtr, sparse=True)
            adj_rm_norm_2 = utils.normalize_adj_tensor(adj_rm_2, self.d_mtr, sparse=True)
        else:
            adj_rm_norm_1 = utils.normalize_adj_tensor(adj_rm_1.to_sparse(), self.d_mtr, sparse=True)
            adj_rm_norm_2 = utils.normalize_adj_tensor(adj_rm_2.to_sparse(), self.d_mtr, sparse=True)

        if val:
            self.ori_model.requires_grad_(False)
        model_name = self.ori_model.__class__.__name__
        groc_loss = ori_gcl_computing(self.ori_model, adj_rm_norm_1, adj_rm_norm_2, batch_users,
                                      batch_pos, self.args, self.device, mask_1=self.args.mask_prob_1,
                                      mask_2=self.args.mask_prob_2, model_name=model_name)
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        gc.collect()
        if model_name in ['NGCF', 'GCMC']:
            bpr_loss, reg_loss = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg,
                                                         adj_drop_out=True)
        else:
            bpr_loss, reg_loss = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg)
        reg_loss = reg_loss * self.ori_model.weight_decay
        loss = self.args.loss_weight_bpr * (bpr_loss + reg_loss) + (1 - self.args.loss_weight_bpr) * groc_loss

        return loss, bpr_loss, groc_loss

    def forward_pass_groc_with_bpr(self, batch_users, batch_pos, batch_neg, ori_adj_sparse, tril_adj_index_0, tril_adj_index_1, sparse, val=False):
        self.ori_model.requires_grad_(True)
        idx_num = self.args.groc_batch_size
        if self.args.only_user_groc:
            batch_users_unique = batch_users.unique()
            # batch_users_unique = self.data_selection_groc(batch_users_unique, idx_num)
        else:
            batch_users_unique = torch.cat((batch_users.unique(), batch_pos.unique()), 0)
            batch_users_unique = self.data_selection_groc(batch_users_unique, idx_num)

        # perturb adj inside training. Insert value (1 / num_inserted) to ori_adj. Where to insert, check GROC

        adj_with_insert, num_insert = self.get_modified_adj_for_insert(batch_users_unique, self.adj_with_2_hops, False)

        # Normalize perturbed adj (with insertion)
        if sparse:
            adj_for_loss_gradient = utils.normalize_adj_tensor(adj_with_insert, self.d_mtr, sparse=True)
        else:
            adj_for_loss_gradient = utils.normalize_adj_tensor(adj_with_insert.to_sparse(), self.d_mtr, sparse=True)
        adj_for_loss_gradient.requires_grad = True
        # self.ori_adj.requires_grad=True
        model_name = self.ori_model.__class__.__name__
        loss_for_grad = ori_gcl_computing(self.ori_model, adj_for_loss_gradient,
                                          adj_for_loss_gradient, batch_users, batch_pos, self.args,
                                          self.device, True, self.args.mask_prob_1,
                                          self.args.mask_prob_2, model_name=model_name)
        if self.args.with_bpr_gradient:
            if model_name in ['NGCF', 'GCMC']:
                bpr_loss, reg_loss = self.ori_model.bpr_loss(adj_for_loss_gradient, batch_users, batch_pos, batch_neg, adj_drop_out=True)
            else:
                bpr_loss, _ = self.ori_model.bpr_loss(adj_for_loss_gradient, batch_users, batch_pos, batch_neg)
            loss = self.args.loss_weight_bpr * bpr_loss + (1 - self.args.loss_weight_bpr) * loss_for_grad

            edge_gradient = torch.autograd.grad(loss, adj_for_loss_gradient, retain_graph=True)[0]
        else:
            edge_gradient = torch.autograd.grad(loss_for_grad, adj_for_loss_gradient, retain_graph=True)[0]

        del adj_for_loss_gradient, loss_for_grad
        gc.collect()
        if sparse:
            adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient_sparse(self.args.insert_prob_1,
                                                                                                  self.args.remove_prob_1,
                                                                                                  batch_users_unique,
                                                                                                  edge_gradient,
                                                                                                  adj_with_insert,
                                                                                                  num_insert)

            adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient_sparse(self.args.insert_prob_2,
                                                                                                  self.args.remove_prob_2,
                                                                                                  batch_users_unique,
                                                                                                  edge_gradient,
                                                                                                  adj_with_insert,
                                                                                                  num_insert)
        else:
            adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_1,
                                                                                           self.args.remove_prob_1,
                                                                                           batch_users_unique,
                                                                                           edge_gradient,
                                                                                           adj_with_insert,
                                                                                           tril_adj_index_0,
                                                                                           tril_adj_index_1,
                                                                                           num_insert)
            adj_insert_remove_1 = adj_insert_remove_1.to_sparse()

            adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_2,
                                                                                           self.args.remove_prob_2,
                                                                                           batch_users_unique,
                                                                                           edge_gradient,
                                                                                           adj_with_insert,
                                                                                           tril_adj_index_0,
                                                                                           tril_adj_index_1,
                                                                                           num_insert)
            adj_insert_remove_2 = adj_insert_remove_2.to_sparse()

        del adj_with_insert, edge_gradient

        adj_norm_1 = utils.normalize_adj_tensor(adj_insert_remove_1, self.d_mtr, sparse=True)
        adj_norm_2 = utils.normalize_adj_tensor(adj_insert_remove_2, self.d_mtr, sparse=True)

        del adj_insert_remove_1
        del adj_insert_remove_2

        gc.collect()
        if val:
            self.ori_model.requires_grad_(False)
        groc_loss = ori_gcl_computing(self.ori_model, adj_norm_1, adj_norm_2, batch_users,
                                      batch_pos, self.args, self.device, mask_1=self.args.mask_prob_1,
                                      mask_2=self.args.mask_prob_2, model_name=model_name)

        del adj_norm_1
        del adj_norm_2
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        gc.collect()
        if model_name in ['NGCF', 'GCMC']:
            bpr_loss, reg_loss = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg, adj_drop_out=True)
        else:
            bpr_loss, reg_loss = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg)
        reg_loss = reg_loss * self.ori_model.weight_decay
        loss = self.args.loss_weight_bpr * (bpr_loss + reg_loss) + (1 - self.args.loss_weight_bpr) * groc_loss

        return loss, bpr_loss, groc_loss

    def data_selection_groc(self, batch_users_unique, idx_num):
        if self.args.groc_batch_size < len(batch_users_unique):
            idx_num = len(batch_users_unique)
        selection_id = torch.randint(0, len(batch_users_unique), (idx_num,)).to(self.device)
        batch_users_unique = batch_users_unique[selection_id]  # only select idx_num anchor nodes for adj_edge insertion3

        return batch_users_unique

    def fit(self):
        pass
