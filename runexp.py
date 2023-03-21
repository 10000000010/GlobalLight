import datetime
import time
import os
from os.path import dirname, abspath
import config
import copy
from pipeline import Pipeline
import argparse
from runners.episode_runner import EpisodeRunner
import numpy as np
import collections
from copy import deepcopy
import json
from anon_env import AnonEnv

# from sacred import Experiment, SETTINGS
# from sacred.observers import FileStorageObserver
# from sacred.utils import apply_backspaces_and_linefeeds
import sys
from types import SimpleNamespace as SN
import yaml

import torch as th
from utils.logging import get_logger
from utils.timehelper import time_left, time_str

from components.transforms import OneHot
from components.episode_buffer import ReplayBuffer
from controllers.separate_controller import SeparateMAC

from latent_q_learner import LatentQLearner

from construct_sample import ConstructSample

import matplotlib.pyplot as plt

logger = get_logger()

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

logger.info("Saving to FileStorageObserver in results/sacred.")


def parse_args():
    parser = argparse.ArgumentParser()
    # The file folder to create/log in
    parser.add_argument("--memo", type=str, default='0715_afternoon_Roma_10_10_bi')  # 1_3,2_2,3_3,4_4
    parser.add_argument("--env", type=int, default=1)  # env=1 means you will run CityFlow
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--road_net", type=str, default='10_10')  # '1_2') # which road net you are going to run
    parser.add_argument("--volume", type=str, default='300')  # '300'
    parser.add_argument("--suffix", type=str, default="0.3_bi")  # 0.3

    global hangzhou_archive
    hangzhou_archive = False
    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY = 5
    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE = 5
    global NUM_ROUNDS
    NUM_ROUNDS = 100
    global EARLY_STOP
    EARLY_STOP = False
    global NEIGHBOR
    # TAKE CARE
    NEIGHBOR = False
    global SAVEREPLAY  # if you want to relay your simulation, set it to be True
    SAVEREPLAY = True
    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE
    ADJACENCY_BY_CONNECTION_OR_GEO = False

    # modify:TOP_K_ADJACENCY in line 154
    global PRETRAIN
    PRETRAIN = False
    parser.add_argument("--mod", type=str, default='CoLight')  # SimpleDQN,SimpleDQNOne,GCN,CoLight,Lit
    parser.add_argument("--cnt", type=int, default=3600)  # 3600
    parser.add_argument("--gen", type=int, default=1)  # 4

    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--onemodel", type=bool, default=False)

    parser.add_argument("--visible_gpu", type=str, default="-1")
    global ANON_PHASE_REPRE
    tt = parser.parse_args()
    if 'CoLight_Signal' in tt.mod:
        # 12dim
        ANON_PHASE_REPRE = {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
        }
    else:
        # 8dim
        ANON_PHASE_REPRE = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        }

    print('agent_name:%s', tt.mod)
    print('ANON_PHASE_REPRE:', ANON_PHASE_REPRE)

    return parser.parse_args()

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf, # experiment config
                   dic_agent_conf=dic_agent_conf, # RL agent config
                   dic_traffic_env_conf=dic_traffic_env_conf, # the simulation configuration
                   dic_path=dic_path # where should I save the logs?
                   )
    global multi_process
    multi_process =False #
    # ppl.run(multi_process=multi_process)

    print("pipeline_wrapper end")
    return

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def main(memo, env, road_net, gui, volume, suffix, mod, cnt, gen, r_all, workers, onemodel):

    # main(args.memo, args.env, args.road_net, args.gui, args.volume, args.ratio, args.mod, args.cnt, args.gen)
    #Jinan_3_4
    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:',num_intersections)

    ENVIRONMENT = ["sumo", "anon"][env]


    traffic_file_list=["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, suffix)]
    traffic_file_list = [i+ ".json" for i in traffic_file_list ]


    process_list = []
    n_workers = workers     #len(traffic_file_list)
    multi_process = False

    global PRETRAIN
    global NUM_ROUNDS
    global EARLY_STOP
    for traffic_file in traffic_file_list:
        dic_exp_conf_extra = {

            "RUN_COUNTS": cnt, #3600
            "MODEL_NAME": mod,
            "TRAFFIC_FILE": [traffic_file], # here: change to multi_traffic

            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "NUM_ROUNDS": NUM_ROUNDS, #100
            "NUM_GENERATORS": gen, #4

            "MODEL_POOL": False,
            "NUM_BEST_MODEL": 3,

            "PRETRAIN": PRETRAIN,#
            "PRETRAIN_MODEL_NAME":mod,
            "PRETRAIN_NUM_ROUNDS": 0,
            "PRETRAIN_NUM_GENERATORS": 15,

            "AGGREGATE": False,
            "DEBUG": False,
            "EARLY_STOP": EARLY_STOP,
        }

        dic_agent_conf_extra = {
            "EPOCHS": 100,
            "SAMPLE_SIZE": 1000,
            "MAX_MEMORY_LEN": 10000,
            "UPDATE_Q_BAR_EVERY_C_ROUND": False,
            "UPDATE_Q_BAR_FREQ": 5,
            # network

            "N_LAYER": 2,
            "TRAFFIC_FILE": traffic_file,
        }

        global TOP_K_ADJACENCY
        global TOP_K_ADJACENCY_LANE
        global NEIGHBOR
        global SAVEREPLAY
        global ADJACENCY_BY_CONNECTION_OR_GEO
        global ANON_PHASE_REPRE
        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,
            "ONE_MODEL": onemodel,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            "IF_GUI": gui,
            "DEBUG": False,
            "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
            "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,
            "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": True,

            "NEIGHBOR": NEIGHBOR,
            "MODEL_NAME": mod,

            "SAVEREPLAY": SAVEREPLAY,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "VOLUME": volume,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "phase_expansion": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            },

            "phase_expansion_4_lane": {
                1: [1, 1, 0, 0],
                2: [0, 0, 1, 1],
            },


            "LIST_STATE_FEATURE": [
                "cur_phase",
                # "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",
                # "coming_vehicle",
                # "leaving_vehicle",
                # "pressure"

                # "adjacency_matrix",
                # "lane_queue_length",
                # "connectivity",

                # adjacency_matrix_lane
            ],

                "DIC_FEATURE_DIM": dict(
                    D_LANE_QUEUE_LENGTH=(4,),
                    D_LANE_NUM_VEHICLE=(4,),

                    D_COMING_VEHICLE = (12,),
                    D_LEAVING_VEHICLE = (12,),

                    D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                    D_CUR_PHASE=(1,),
                    D_NEXT_PHASE=(1,),
                    D_TIME_THIS_PHASE=(1,),
                    D_TERMINAL=(1,),
                    D_LANE_SUM_WAITING_TIME=(4,),
                    D_VEHICLE_POSITION_IMG=(4, 60,),
                    D_VEHICLE_SPEED_IMG=(4, 60,),
                    D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                    D_PRESSURE=(1,),

                    D_ADJACENCY_MATRIX=(2,),

                    D_ADJACENCY_MATRIX_LANE=(6,),

                ),

            "DIC_REWARD_INFO": {
                "flickering": 0,#-5,#
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,#-1,#
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0  # -0.25
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                },

                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                #     2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                #     3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                #     4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
                "anon":ANON_PHASE_REPRE,
                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                #     2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                #     3: [1, 0ONE_MODEL1, 0, 0, 0, 0, 0],# 'WLEL',
                #     4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
            }
        }

        ## ==================== multi_phase ====================
        global hangzhou_archive

        template = "template_lsr"

        if mod in ['CoLight','GCN','SimpleDQNOne']:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = 1
            dic_traffic_env_conf_extra['ONE_MODEL'] = False
            if "adjacency_matrix" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                "adjacency_matrix_lane" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                mod not in ['SimpleDQNOne']:
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix")
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix_lane")

                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                    (dic_traffic_env_conf_extra['TOP_K_ADJACENCY'],)

                if dic_traffic_env_conf_extra['USE_LANE_ADJACENCY']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX_LANE'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY_LANE'],)

        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)

            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

        print(traffic_file)
        prefix_intersections = str(road_net)
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),

            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            "PATH_TO_ERROR": os.path.join("errors", memo)
        }

        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
        deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(mod.upper())),
                                      dic_agent_conf_extra)
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)

        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)


        params = deepcopy(sys.argv)
        # Get the defaults from default.yaml
        with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"),
                  "r") as f:  # src/config/default.yaml
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)

        # Load algorithm and env base configs
        params.append('--config=qmix_smac_latent')
        params.append('--env-config=sc2')

        env_config = _get_config(params, "--env-config", "envs")
        alg_config = _get_config(params, "--config", "algs")
        config_dict = recursive_dict_update(config_dict, env_config)
        config_dict = recursive_dict_update(config_dict, alg_config)

        np.random.seed(610)
        th.manual_seed(610)
        config_dict['env_args']['seed'] = 610
        args = SN(**config_dict)
        # configure tensorboard logger
        unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        args.unique_token = unique_token
        if args.use_tensorboard:
            tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
            tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
            args.latent_role_direc = os.path.join(tb_exp_direc, "{}").format('latent_role')
            # logger.setup_tb(tb_exp_direc)
            # Import here so it doesn't have to be installed if you don't use it
            from tensorboard_logger import configure, log_value
            configure(tb_exp_direc)
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(tb_exp_direc + "-latent")

        pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                         dic_agent_conf=deploy_dic_agent_conf,
                         dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                         dic_path=deploy_dic_path)

        run(args = args,dic_exp_conf=deploy_dic_exp_conf,
                         dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                         dic_path=deploy_dic_path,writer = writer)

    return memo

def run(args,dic_exp_conf,dic_traffic_env_conf,dic_path,writer):

    args.batch_size_run = 1
    args.use_cuda = True
    args.device = 'cuda'
    episode_limit = 360
    args.target_update_interval = 5

    runner = EpisodeRunner(args =args,dic_exp_conf=dic_exp_conf,
                         dic_traffic_env_conf=dic_traffic_env_conf,
                         dic_path=dic_path,episode_limit =episode_limit)

    # Set up schemes and groups here
    # env_info = runner.get_env_info()

    args.n_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
    args.n_actions = len(dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']]) #4
    n_feature = args.n_actions + 12
    # obs_shape = args.n_agents * n_feature #(others_f + own_f)
    obs_shape = n_feature
    # state_shape =  args.n_agents * obs_shape #(num_agents*(others_f + own_f))
    state_shape = args.n_agents * obs_shape
    args.state_shape = state_shape

    # obs_shape = args.n_agents * 12
    # state_shape = args.n_agents * 12 + args.n_agents * args.n_actions
    scheme = {
        "state": {"vshape": (state_shape,)},
        "obs": {"vshape": obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (4,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    groups = {
        "agents": args.n_agents
    }

    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    args.buffer_size = 1000 #original =5000
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, episode_limit + 1,
                          preprocess=preprocess,
                          device="cuda" ) #args.buffer_size = 2000

    # Setup multiagent controller here
    mac = SeparateMAC(buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac,episode_limit =episode_limit,device=args.device)

    # Learner
    learner = LatentQLearner(mac, buffer.scheme, logger, args,writer)

    if args.use_cuda:
        learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    args.t_max = 150* episode_limit
    # while runner.t_env <= args.t_max: #10050000
    dis_loss_list = []
    iteration = 0
    iteration_list = []
    for cnt_round in range(150):
        cnt_gen =1
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                   "round_" + str(cnt_round), "generator_" + str(cnt_gen))

        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)

        env  = AnonEnv( path_to_log = path_to_log,
                              path_to_work_directory = dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf = dic_traffic_env_conf)

        runner.env =env

        print("round %d starts" % cnt_round)
        round_start_time = time.time()

        print("==============  generator =============")
        generator_start_time = time.time()

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        generator_end_time = time.time()
        generator_total_time = generator_end_time - generator_start_time

        args.batch_size = 2
        if buffer.can_sample(args.batch_size):

            print("==============  make samples =============")
            # make samples and determine which samples are good
            making_samples_start_time = time.time()

            train_round = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)

            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=dic_traffic_env_conf)
            cs.make_reward_for_system()

            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time

            print("==============  update network =============")
            update_rounds =10
            update_network_start_time = time.time()
            for update_round in range(update_rounds):
                episode_sample = buffer.sample(args.batch_size)
                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)
                dis_loss = learner.train(episode_sample, runner.adjs,runner.t_env, episode)

                dis_loss_list.append(dis_loss)
                iteration +=1
                iteration_list.append(iteration)
                plt.figure()
                try:
                    train_disloss_lines.remove(train_disloss_lines[0])  # 移除上一步曲线
                except Exception:
                    pass

                train_disloss_lines = plt.plot(iteration_list, dis_loss_list, 'b', lw=1)
                plt.title("disimilarity loss during training")
                plt.xlabel("iteration")
                plt.ylabel("disloss")
                plt.savefig(fname='./dis_loss.png', format="png")

            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time

            print("==============  save model =============")
            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                # "results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                print("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

            print("==============  test evaluation =============")
            test_evaluation_start_time = time.time()
            # Execute test runs once in a while
            args.test_nepisode = 1
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)

            eval(runner,dic_path["PATH_TO_MODEL"],
                 cnt_round,dic_exp_conf["RUN_COUNTS"],dic_traffic_env_conf)

            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                print("t_env: {} / {}".format(runner.t_env, args.t_max))
                print("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()
                last_test_T = runner.t_env

        episode += args.batch_size_run
        print("Finished episode = {}".format(episode))
        # if (runner.t_env - last_log_T) >= args.log_interval:
        #     logger.log_stat("episode", episode, runner.t_env)
        #     logger.print_recent_stats()
        #     last_log_T = runner.t_env

    print("Finished Training")

def eval(runner,model_dir, cnt_round, run_cnt, _dic_traffic_env_conf):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d"%cnt_round
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)
    if os.path.exists(os.path.join(records_dir, "sumo_env.conf")):
        with open(os.path.join(records_dir, "sumo_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    elif os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)

    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)

    env = AnonEnv(path_to_log=path_to_log,  path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
                                                           dic_traffic_env_conf=dic_traffic_env_conf)
    runner.env = env
    runner.run(test_mode = True)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args.memo, args.env, args.road_net, args.gui, args.volume,
         args.suffix, args.mod, args.cnt, args.gen, args.all, args.workers,
         args.onemodel)
