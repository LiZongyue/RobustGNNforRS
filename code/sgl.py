import torch
import torch.nn as nn
import torch.nn.functional as F


def sgl_computing(trn_model, gra1, gra2, users, item, args, model_name=None):
    """
    mask_1: prob of mask_1
    """
    if model_name in ['NGCF', 'GCMC']:
        (users_emb_perturb_1, items_emb_perturb_1, _, _) = trn_model.getEmbedding(gra1, users.long(), item.long(),
                                                                                  adj_drop_out=False, sgl=True)
        (users_emb_perturb_2, items_emb_perturb_2, _, _) = trn_model.getEmbedding(gra2, users.long(), item.long(),
                                                                                  adj_drop_out=False, sgl=True)
        all_users_1, all_items_1 = trn_model._forward_gcn(gra1)
        all_users_2, all_items_2 = trn_model._forward_gcn(gra2)

        pos_ratings_user = inner_product(users_emb_perturb_1, users_emb_perturb_2)  # [batch_size]
        pos_ratings_item = inner_product(items_emb_perturb_1, items_emb_perturb_2)  # [batch_size]

        tot_ratings_user = torch.matmul(users_emb_perturb_1,
                                        torch.transpose(all_users_1, 0, 1))  # [batch_size, num_users]
        tot_ratings_item = torch.matmul(items_emb_perturb_1,
                                        torch.transpose(all_items_2, 0, 1))  # [batch_size, num_items]

    else:
        (users_emb_perturb_1, items_emb_perturb_1, _, _) = trn_model.getEmbedding(gra1, users.long(), item.long(), sgl=True)
        (users_emb_perturb_2, items_emb_perturb_2, _, _) = trn_model.getEmbedding(gra2, users.long(), item.long(), sgl=True)

        all_users_1, all_items_1 = trn_model._forward_gcn(gra1)
        all_users_2, all_items_2 = trn_model._forward_gcn(gra2)

        users_norm_1 = nn.functional.normalize(users_emb_perturb_1, dim=1)
        users_norm_2 = nn.functional.normalize(users_emb_perturb_2, dim=1)
        items_norm_1 = nn.functional.normalize(items_emb_perturb_1, dim=1)
        items_norm_2 = nn.functional.normalize(items_emb_perturb_2, dim=1)

        pos_ratings_user = inner_product(users_norm_1, users_norm_2)  # [batch_size]
        pos_ratings_item = inner_product(items_norm_1, items_norm_2)  # [batch_size]
        tot_ratings_user = torch.matmul(users_norm_1,
                                        torch.transpose(all_users_1, 0, 1))  # [batch_size, num_users]
        tot_ratings_item = torch.matmul(items_norm_1,
                                        torch.transpose(all_items_2, 0, 1))  # [batch_size, num_items]

    ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]  # [batch_size, num_users]
    ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]  # [batch_size, num_users]

    clogits_user = torch.logsumexp(ssl_logits_user / args.sgl_t, dim=1)
    clogits_item = torch.logsumexp(ssl_logits_item / args.sgl_t, dim=1)
    infonce_loss = torch.sum(clogits_user + clogits_item)

    return infonce_loss


def inner_product(a, b):
    return torch.sum(a*b, dim=-1)
