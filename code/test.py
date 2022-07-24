import torch
import lightgcn
from register import dataset
import utils
import Procedure
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='', help='file name of model')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a2 = torch.load("/home/stud/zongyue/workarea/RobustGNNforRS/data/perturb-adj/modified_adj_a_02.pt").to(device)
a4 = torch.load("/home/stud/zongyue/workarea/RobustGNNforRS/data/perturb-adj/modified_adj_a_04.pt").to(device)
a6 = torch.load("/home/stud/zongyue/workarea/RobustGNNforRS/data/perturb-adj/modified_adj_a_06.pt").to(device)
a8 = torch.load("/home/stud/zongyue/workarea/RobustGNNforRS/data/perturb-adj/modified_adj_a_08.pt").to(device)

adj_list = [a2, a4, a6, a8]
num_users = dataset.n_user
num_items = dataset.m_item
path_model_base = "/home/stud/zongyue/workarea/RobustGNNforRS/models/GROC_models/ml-1m/" + args.file

groc_lgn = lightgcn.LightGCN(device, num_users, num_items, sparse=True, is_light_gcn=True, use_dcl=False, train_groc=False)
groc_lgn.load_state_dict(torch.load(path_model_base)).to(device)

for adj in adj_list:
    Procedure.Test(dataset, groc_lgn, 100, utils.normalize_adj_tensor(adj), None, 0)
