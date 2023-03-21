import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D
import math
from tensorboardX import SummaryWriter
import time
from torch.nn.parameter import Parameter

def loss_function(preds, labels, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder


    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 / n_nodes * th.mean(th.sum(
    #     1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        th.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        # input = input.to(th.double)
        # self.weight  =self.weight.to(th.double)
        bs ,n_agents, n_fea= input.size(0),input.size(1),input.size(2)
        input = input.reshape(-1,n_fea)
        # inputs = th.transpose(inputs,0,1)#
        # inputs = inputs.reshape(self.n_agents,-1)

        support = th.mm(input, self.weight)
        support = support.reshape(bs,n_agents,-1)
        support = th.transpose(support,0,1)
        support = support.reshape(n_agents,-1)

        output = th.spmm(adj, support)
        output = self.act(output)
        output =output.reshape(n_agents,bs,-1)
        output =th.transpose(output,0,1)
        output = output.reshape(bs*n_agents,-1)
        return output

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout,  act=th.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, n_agents):
        n_fea = z.size(1)
        z  = z.reshape(-1,n_agents,n_fea)
        z = th.mean(z,dim = 0,keepdim =False)
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(th.mm(z, z.t()))
        return adj


class LatentCEDisRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentCEDisRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 0

        self.embed_fc_input_size = input_shape
        NN_HIDDEN_SIZE = args.NN_HIDDEN_SIZE
        activation_func=nn.LeakyReLU()

        self.embed_net = nn.Sequential(nn.Linear(self.embed_fc_input_size, NN_HIDDEN_SIZE),
                                       nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       activation_func,
                                       nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))

        """config network params LJT//2022/05/10"""
        #for gcn
        self.gc = GraphConvolution(input_shape, args.latent_dim * 2, dropout= 0.0, act=F.relu)
        self.dc = InnerProductDecoder(dropout = 0.0, act = lambda x: x)

        self.inference_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + input_shape, NN_HIDDEN_SIZE),
                                           nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                           activation_func,
                                           nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))

        self.latent = th.rand(args.n_agents, args.latent_dim * 2)  # (n,mu+var)
        self.latent_infer = th.rand(args.n_agents, args.latent_dim * 2)  # (n,mu+var)

        self.latent_net = nn.Sequential(nn.Linear(args.latent_dim, NN_HIDDEN_SIZE),
                                        nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                        activation_func)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc2_w_nn = nn.Linear(NN_HIDDEN_SIZE, args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_nn = nn.Linear(NN_HIDDEN_SIZE, args.n_actions)

        # Dis Net
        self.dis_net = nn.Sequential(nn.Linear(args.latent_dim * 2, NN_HIDDEN_SIZE ),
                                     nn.BatchNorm1d(NN_HIDDEN_SIZE ),
                                     activation_func,
                                     nn.Linear(NN_HIDDEN_SIZE , 1))

        self.mi = th.rand(args.n_agents*args.n_agents)
        self.dissimilarity = th.rand(args.n_agents*args.n_agents)

        if args.dis_sigmoid:
            print('>>> sigmoid')
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_sigmoid
        else:
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_step

    def init_latent(self, bs):
        self.bs = bs
        loss = 0

        if self.args.runner == "episode":
            self.writer = SummaryWriter(
                "results/tb_logs/test_latent-" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        self.trajectory = []

        var_mean=self.latent[:self.n_agents, self.args.latent_dim:].detach().mean()

        #mask = 1 - th.eye(self.n_agents).byte()
        #mi=self.mi.view(self.n_agents,self.n_agents).masked_select(mask)
        #di=self.dissimilarity.view(self.n_agents,self.n_agents).masked_select(mask)
        mi = self.mi
        di = self.dissimilarity
        indicator=[var_mean,mi.max(),mi.min(),mi.mean(),mi.std(),di.max(),di.min(),di.mean(),di.std()]
        return indicator, self.latent[:self.n_agents, :].detach(), self.latent_infer[:self.n_agents, :].detach()

    def forward(self, inputs,adj, hidden_state, t=0, batch=None, test_mode=None, t_glob=0, train_mode=False):
        inputs = inputs.reshape(-1, self.n_agents,self.input_shape)

        h_in = hidden_state.reshape(-1, self.hidden_dim)

        # embed_fc_input = inputs[:, - self.embed_fc_input_size:]  # own features(unit_type_bits+shield_bits_ally)+id
        # self.latent = self.embed_net(embed_fc_input)

        #GCN embeding
        adj_norm = adj["adj_norm"]
        self.latent = self.gc(inputs, adj_norm) #(bs,n,dim)

        # loss_gcn = loss_function(preds=self.dc(self.latent,self.n_agents), labels=adj["adj_label"],
        #                      norm=adj["norm"], pos_weight=adj["pos_weight"])

        inputs = inputs.reshape(-1,self.input_shape)

        self.latent[:, -self.latent_dim:] = th.clamp(th.exp(self.latent[:, -self.latent_dim:]), min=self.args.var_floor)  # var
        #self.latent[:, -self.latent_dim:] = th.full_like(self.latent[:, -self.latent_dim:],1.0)

        latent_embed = self.latent.reshape(self.bs * self.n_agents, self.latent_dim * 2)

        #latent = latent_embed[:, :self.latent_dim]
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        latent = gaussian_embed.rsample()

        c_dis_loss = th.tensor(0.0).to(self.args.device)
        ce_loss = th.tensor(0.0).to(self.args.device)
        loss = th.tensor(0.0).to(self.args.device)

        if train_mode and (not self.args.roma_raw):
            #gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
            #latent = gaussian_embed.rsample()

            self.latent_infer = self.inference_net(th.cat([h_in.detach(), inputs], dim=1))
            self.latent_infer[:, -self.latent_dim:] = th.clamp(th.exp(self.latent_infer[:, -self.latent_dim:]),min=self.args.var_floor)
            #self.latent_infer[:, -self.latent_dim:] = th.full_like(self.latent_infer[:, -self.latent_dim:],1.0)
            gaussian_infer = D.Normal(self.latent_infer[:, :self.latent_dim], (self.latent_infer[:, self.latent_dim:]) ** (1 / 2))
            latent_infer = gaussian_infer.rsample()

            loss = gaussian_embed.entropy().sum(dim=-1).mean() * self.args.h_loss_weight + kl_divergence(gaussian_embed, gaussian_infer).sum(dim=-1).mean() * self.args.kl_loss_weight   # CE = H + KL
            loss = th.clamp(loss, max=2e3)
            # loss = loss / (self.bs * self.n_agents)
            ce_loss = th.log(1 + th.exp(loss))

            # Dis Loss
            cur_dis_loss_weight = self.dis_loss_weight_schedule(t_glob)
            if cur_dis_loss_weight > 0:
                dis_loss = 0
                dissimilarity_cat = None
                mi_cat = None
                latent_dis = latent.clone().view(self.bs, self.n_agents, -1)
                latent_move = latent.clone().view(self.bs, self.n_agents, -1)
                for agent_i in range(self.n_agents):
                    latent_move = th.cat(
                        [latent_move[:, -1, :].unsqueeze(1), latent_move[:, :-1, :]], dim=1)
                    latent_dis_pair = th.cat([latent_dis[:, :, :self.latent_dim],
                                              latent_move[:, :, :self.latent_dim],
                                            # (latent_dis[:, :, :self.latent_dim]-latent_move[:, :, :self.latent_dim])**2
                                              ], dim=2)
                    mi = th.clamp(gaussian_embed.log_prob(latent_move.view(self.bs * self.n_agents, -1))+13.9, min=-13.9).sum(dim=1,keepdim=True) / self.latent_dim

                    dissimilarity = th.abs(self.dis_net(latent_dis_pair.view(-1, 2 * self.latent_dim)))

                    if dissimilarity_cat is None:
                        dissimilarity_cat = dissimilarity.view(self.bs, -1).clone()
                    else:
                        dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(self.bs, -1)], dim=1)
                    if mi_cat is None:
                        mi_cat = mi.view(self.bs, -1).clone()
                    else:
                        mi_cat = th.cat([mi_cat,mi.view(self.bs,-1)],dim=1)

                    #dis_loss -= th.clamp(mi / 100 + dissimilarity, max=0.18).sum() / self.bs / self.n_agents

                mi_min=mi_cat.min(dim=1,keepdim=True)[0]
                mi_max=mi_cat.max(dim=1,keepdim=True)[0]
                di_min = dissimilarity_cat.min(dim=1, keepdim=True)[0]
                di_max = dissimilarity_cat.max(dim=1, keepdim=True)[0]

                mi_cat=(mi_cat-mi_min)/(mi_max-mi_min+ 1e-12 )
                dissimilarity_cat=(dissimilarity_cat-di_min)/(di_max-di_min+ 1e-12 )

                dis_loss = - th.clamp(mi_cat+dissimilarity_cat, max=1.0).sum()/self.bs/self.n_agents
                #dis_loss = ((mi_cat + dissimilarity_cat - 1.0 )**2).sum() / self.bs / self.n_agents
                dis_norm = th.norm(dissimilarity_cat, p=1, dim=1).sum() / self.bs / self.n_agents

                #c_dis_loss = (dis_loss + dis_norm) / self.n_agents * cur_dis_loss_weight
                c_dis_loss = (dis_norm + self.args.soft_constraint_weight * dis_loss) / self.n_agents * cur_dis_loss_weight
                loss = ce_loss +  c_dis_loss

                self.mi = mi_cat[0]
                self.dissimilarity = dissimilarity_cat[0]
            else:
                c_dis_loss = th.zeros_like(loss)
                loss = ce_loss


        # Role -> FC2 Params
        latent = self.latent_net(latent)

        fc2_w = self.fc2_w_nn(latent)
        fc2_b = self.fc2_b_nn(latent)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))

        x = F.relu(self.fc1(inputs))  # (bs*n,(obs+act+id)) at time t
        h = self.rnn(x, h_in)
        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        q = th.bmm(h, fc2_w) + fc2_b

        h = h.reshape(-1, self.args.rnn_hidden_dim)

        if self.args.runner == "episode":
            self.writer.add_embedding(self.latent.reshape(-1, self.latent_dim * 2), list(range(self.args.n_agents)),
                                      global_step=t, tag="latent-cur")
            self.writer.add_embedding(self.latent_infer.reshape(-1, self.latent_dim * 2),
                                      list(range(self.args.n_agents)),
                                      global_step=t, tag="latent-hist")

        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), loss, c_dis_loss, ce_loss

    def dis_loss_weight_schedule_step(self, t_glob):
        if t_glob > self.args.dis_time:
            return self.args.dis_loss_weight
        else:
            return 0

    def dis_loss_weight_schedule_sigmoid(self, t_glob):
        return self.args.dis_loss_weight / (1 + math.exp((1e7 - t_glob) / 2e6))
