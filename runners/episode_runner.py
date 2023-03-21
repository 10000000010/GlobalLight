from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from anon_env import AnonEnv
import os
import time
from keras.utils import np_utils,to_categorical
import scipy.sparse as sp
import torch as th

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

class EpisodeRunner:

    def __init__(self, args,
     dic_path, dic_exp_conf, dic_traffic_env_conf, cnt_gen=1,cnt_round=0,episode_limit=0):
        self.args = args
        # self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        ##########
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        if self.dic_exp_conf["PRETRAIN"]:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
        else:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))

        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        self.env = AnonEnv( path_to_log = self.path_to_log,
                              path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf = self.dic_traffic_env_conf)

        self.episode_limit = episode_limit # 360
        ##############
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac,episode_limit,device):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, episode_limit + 1,
                                 preprocess=preprocess, device=device)
        self.mac = mac
        self.device =device

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        state  = self.env.reset()
        self.t = 0
        return state

    def adjacency_index2matrix(self,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """
        #[batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index,axis=-1)
        l = to_categorical(adjacency_index_new,num_classes=self.args.n_agents)
        return l

    def get_state_obs(self,state):
        observations = []
        # nf_al = 12+4
        adj = []
        for j in range(self.args.n_agents):
            agent_obs = []
            #own_feats
            own_fea = []
            lane_num_vehicle = state[j]["lane_num_vehicle"]
            own_fea.extend(lane_num_vehicle)
            action_id = state[j]["cur_phase"][0]
            action_onehot = [0] * 4
            action_onehot[action_id - 1] = 1
            own_fea.extend(action_onehot)

            # #other_f
            # other_f = []
            # for i in range(self.args.n_agents):
            #     if i != j:
            #         fea = []
            #         lane_num_vehicle = state[i]["lane_num_vehicle"]
            #         own_fea.extend(lane_num_vehicle)
            #         action_id = state[i]["cur_phase"][0]
            #         action_onehot = [0] * 4
            #         action_onehot[action_id - 1] = 1
            #         fea.extend(action_onehot)
            #
            #         other_f.extend(fea)
            # agent_obs.extend(other_f)
            agent_obs.extend(own_fea)

            agent_obs = np.array(agent_obs,dtype=np.float32)

            observations.append(agent_obs)

            adj.append(state[j]['adjacency_matrix'])

        states = np.concatenate(observations, axis= 0).astype(
                np.float32
            )

        # observations = np.reshape(observations,[self.args.n_agents,-1])
        total_adjs = self.adjacency_index2matrix(np.array(adj))
        # self.adjs = th.from_numpy(np.sum(total_adjs,axis=1)).to(th.float32).to(device=self.device)
        A =  np.sum(total_adjs,axis=1)
        adj = sp.csr_matrix(A)

        # Some preprocessing
        adj_norm = preprocess_graph(adj)

        adj_label = adj + sp.eye(adj.shape[0])

        adj_label = th.FloatTensor(adj_label.toarray())
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        self.adjs ={"adj":adj, "adj_label":adj_label.to(device = "cuda"),"pos_weight":pos_weight,
                    "norm":norm,"adj_norm":adj_norm.to(device= "cuda")}

        pre_transition_data = {
            "state": [states],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [observations],
        }
        return pre_transition_data

    def run(self, test_mode=False):
        self.reset()
        terminated = False
        episode_return = 0
        num_vechile_sum_list = []

        self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.mac == "separate_mac":
            self.mac.init_latent(batch_size=self.batch_size)

        while not terminated:
            state,_ = self.env.get_state()

            pre_transition_data =self.get_state_obs(state)
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1


            actions = self.mac.select_actions(self.batch, self.adjs ,t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            next_state, reward, terminated, _ = self.env.step(actions[0])

            reward = sum(reward)
            episode_return += reward

            num_vechile_sum_list.append(reward/(-0.25))

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

            if self.t == self.episode_limit:
                terminated = True

            # if test_mode == True and self.t  == 1:
            #     indicator, emb, emb_vae = self.mac.init_latent(batch_size=self.batch_size)
            #     emb = emb.cpu().numpy()
            #     np.save('./emb_{}'.format(self.t), emb)
            #
            # if test_mode == True and self.t % 100 == 0:
            #     indicator, emb, emb_vae = self.mac.init_latent(batch_size=self.batch_size)
            #     emb = emb.cpu().numpy()
            #     np.save('./emb_{}'.format(self.t), emb)

            if test_mode == True:
                indicator, emb, emb_vae = self.mac.init_latent(batch_size=self.batch_size)
                emb = emb.cpu().numpy()
                np.save('./emb/emb_{}'.format(self.t), emb)

        num_stop_vechile_sum = np.mean(num_vechile_sum_list)

        print("sum_num_stop_vechiles= {}".format(num_stop_vechile_sum))

        last_state, _ = self.env.get_state()
        last_data = self.get_state_obs(last_state)
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, self.adjs,t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                print("epsilon: {}".format(self.mac.action_selector.epsilon) + 'time:{}'.format(self.t_env))
            self.log_train_stats_t = self.t_env


        log_start_time = time.time()
        print("start logging")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time

        self.env.end_sumo()


        return self.batch

    def _log(self, returns,prefix):
        print(prefix + "return_mean: {}".format(np.mean(returns) ) + "    time:{}".format(self.t_env))
        print(prefix + "return_std: {}".format(np.std(returns)) +"   time:{}".format(self.t_env))
        returns.clear()

    # def _log(self, returns, stats, prefix):
    #     self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
    #     self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
    #     returns.clear()
    #
    #     for k, v in stats.items():
    #         if k != "n_episodes":
    #             self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
    #     stats.clear()
