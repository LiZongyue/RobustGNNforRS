import torch
import torch.nn as nn
import torch.nn.functional as F


def embed_mask(trn_model, mask_type, users_emb_perturb_1, users_emb_perturb_2, mask_p_1, mask_p_2):
    mask_1 = (torch.FloatTensor(trn_model.latent_dim).uniform_() < mask_p_1).to(trn_model.device)
    mask_2 = (torch.FloatTensor(trn_model.latent_dim).uniform_() < mask_p_2).to(trn_model.device)
    if mask_type == 'mask_normalized_aggregated_emb':
        users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1).masked_fill_(mask_1, 0.)
        users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1).masked_fill_(mask_2, 0.)
    elif mask_type == 'mask_aggregated_emb':
        users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1.masked_fill_(mask_1, 0.), dim=1)
        users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2.masked_fill_(mask_2, 0.), dim=1)
    elif mask_type == 'mask_aggregated_emb_node':
        users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1)
        users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1)
        mask_1 = users_emb_perturb_1.new_empty((users_emb_perturb_1.shape[0], 1)).bernoulli_(1 - mask_p_1).expand_as(users_emb_perturb_1).to(trn_model.device)
        users_emb_perturb_1 = mask_1 * users_emb_perturb_1
        mask_2 = users_emb_perturb_2.new_empty((users_emb_perturb_2.shape[0], 1)).bernoulli_(1 - mask_p_2).expand_as(users_emb_perturb_2).to(trn_model.device)
        users_emb_perturb_2 = mask_2 * users_emb_perturb_2
    elif mask_type == 'mask_emb':
        users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1)
        users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1)
    else:
        raise Exception('mask_type parameter initilized incorrectly.')
    return users_emb_perturb_1, users_emb_perturb_2


def ori_gcl_computing(trn_model, gra1, gra2, users, poss, args, device, gcl_for_gradient=False,
                      mask_p_1=None, mask_p_2=None, model_name=None):
    """
    mask_1: prob of mask_1
    """

    if gcl_for_gradient:  # first GCL computing for edge gradient
        if model_name in ['NGCF', 'GCMC']:
            if args.mask_type == 'mask_emb':
                (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(),
                                                                        adj_drop_out=False, mask_prob=mask_p_1)
                (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(),
                                                                        adj_drop_out=False, mask_prob=mask_p_2)
            else:
                (users_emb_perturb, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(), adj_drop_out=False)
                users_emb_perturb_1, users_emb_perturb_2 = embed_mask(trn_model, args.mask_type, users_emb_perturb, users_emb_perturb,
                                                                      mask_p_1, mask_p_2)
        else:
            if args.mask_type == 'mask_emb':
                (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(), mask_prob=mask_p_1)
                (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(), mask_prob=mask_p_2)
            else:
                (users_emb_perturb, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long())
                users_emb_perturb_1, users_emb_perturb_2 = embed_mask(trn_model, args.mask_type, users_emb_perturb, users_emb_perturb,
                                                                      mask_p_1, mask_p_2)
    else:  # generate gcl loss for 2 views of perturbated adjs and utilized for backward optimization
        if model_name in ['NGCF', 'GCMC']:
            if args.mask_type == 'mask_emb':
                (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(), adj_drop_out=False, mask_prob=mask_p_1)
                (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra2, users.long(), poss.long(), adj_drop_out=False, mask_prob=mask_p_2)
            else:
                (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(),
                                                                        adj_drop_out=False)
                (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra2, users.long(), poss.long(),
                                                                        adj_drop_out=False)
                users_emb_perturb_1, users_emb_perturb_2 = embed_mask(trn_model, args.mask_type, users_emb_perturb_1,
                                                                      users_emb_perturb_2, mask_p_1, mask_p_2)
        else:
            if args.mask_type == 'mask_emb':
                (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(), mask_prob=mask_p_1)
                (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra2, users.long(), poss.long(), mask_prob=mask_p_2)
            else:
                (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long())
                (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra2, users.long(), poss.long())
                users_emb_perturb_1, users_emb_perturb_2 = embed_mask(trn_model, args.mask_type, users_emb_perturb_1,
                                                                      users_emb_perturb_2, mask_p_1, mask_p_2)
        # if mask_1 is not None:
        #     mask_1 = (torch.FloatTensor(users_emb_perturb_1.shape[1]).uniform_() < mask_1).to(trn_model.device)
        #     users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1).masked_fill_(mask_1, 0.)
        # else:
        #     users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1)
        #
        #
        # if mask_2 is not None:
        #     mask_2 = (torch.FloatTensor(users_emb_perturb_2.shape[1]).uniform_() < mask_2).to(trn_model.device)
        #     users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1).masked_fill_(mask_2, 0.)
        # else:
        #     users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1)
    users_dot_12 = torch.bmm(users_emb_perturb_1.unsqueeze(1), users_emb_perturb_2.unsqueeze(2)).squeeze(2)
    users_dot_12 /= args.T_groc
    fenzi_12 = torch.exp(users_dot_12).sum(1)

    neg_emb_users_12 = users_emb_perturb_2.unsqueeze(0).repeat(len(poss), 1, 1)
    neg_dot_12 = torch.bmm(neg_emb_users_12, users_emb_perturb_1.unsqueeze(2)).squeeze(2)
    neg_dot_12 /= args.T_groc
    neg_dot_12 = torch.exp(neg_dot_12).sum(1)

    mask_11 = get_negative_mask_perturb(users_emb_perturb_1.size(0)).to(device)
    neg_dot_11 = torch.exp(torch.mm(users_emb_perturb_1, users_emb_perturb_1.t()) / args.T_groc)
    neg_dot_11 = neg_dot_11.masked_select(mask_11).view(users_emb_perturb_1.size(0), -1).sum(1)
    loss_perturb_11 = (-torch.log(fenzi_12 / (neg_dot_11 + neg_dot_12))).mean()

    users_dot_21 = torch.bmm(users_emb_perturb_2.unsqueeze(1), users_emb_perturb_1.unsqueeze(2)).squeeze(2)
    users_dot_21 /= args.T_groc
    fenzi_21 = torch.exp(users_dot_21).sum(1)

    neg_emb_users_21 = users_emb_perturb_1.unsqueeze(0).repeat(len(poss), 1, 1)
    neg_dot_21 = torch.bmm(neg_emb_users_21, users_emb_perturb_2.unsqueeze(2)).squeeze(2)
    neg_dot_21 /= args.T_groc
    neg_dot_21 = torch.exp(neg_dot_21).sum(1)

    mask_22 = get_negative_mask_perturb(users_emb_perturb_2.size(0)).to(device)
    neg_dot_22 = torch.exp(torch.mm(users_emb_perturb_2, users_emb_perturb_2.t()) / args.T_groc)
    neg_dot_22 = neg_dot_22.masked_select(mask_22).view(users_emb_perturb_2.size(0), -1).sum(1)
    loss_perturb_22 = (-torch.log(fenzi_21 / (neg_dot_22 + neg_dot_21))).mean()

    loss_perturb = loss_perturb_11 + loss_perturb_22

    return loss_perturb


def get_negative_mask_perturb(batch_size):
    negative_mask = torch.ones(batch_size, batch_size).bool()
    for i in range(batch_size):
        negative_mask[i, i] = 0

    return negative_mask
