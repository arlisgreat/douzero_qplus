#utils.py
#工具模块
import os 
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch 
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env
from douzero.env.env import _cards2array

#扑克牌映射关系 没有joker
Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

#0-4映射到长度为4的数组
NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

#日志管理
shandle = logging.StreamHandler()#创建日志处理器，用于将日志输出到控制台
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))#定义了日志的输出格式
log = logging.getLogger('doudzero')#创建日志管理器，名字是doudzero
log.propagate = False#禁止日志传递到父记录器
log.addHandler(shandle)#将日志处理器添加到日志记录器
log.setLevel(logging.INFO)#设置级别 
#INFO 普通信息
#DEBUG 调试信息
#WARNING 警告信息
#ERROR 错误信息
#CRITICAL 严重错误信息

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
#缓冲区管理，字典 键是设备名称，值是张量列表
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

#显式调用Env
def create_env(flags):
    return Env(flags.objective)

#强化学习从共享缓冲区提取训练数据
def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    #free_queue存储空闲的buffer索引
    #full_queue存储满的buffer索引
    #lock是多进程锁
    #flags配置参数
    #buffers是共享内存的buffer
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    #取数据
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    #使用过的放回缓冲区
    #返回从缓冲区提取的batch数据
    return batch

#每个决策创建一个独立的优化器
#自适应学习率优化算法RMSprop
def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers

# #创建缓冲区，储存强化学习数据
# #地主319 农民430
# #缓冲区的数据规格：
# #episode_return: 每个episode的回报 float32
# #obs_x_no_action: 不包含动作的观察数据 (T,x_dim) uint8
# #obs_action: 动作数据 (T,54) uint8
# #obs_z: 观察数据 (T,5,162) uint8
# def create_buffers(flags, device_iterator):
#     """
#     We create buffers for different positions as well as
#     for different devices (i.e., GPU). That is, each device
#     will have three buffers for the three positions.
#     """
#     T = flags.unroll_length
#     positions = ['landlord', 'landlord_up', 'landlord_down']
#     buffers = {}
#     for device in device_iterator:
#         buffers[device] = {}
#         for position in positions:
#             x_dim = 319 if position == 'landlord' else 430
#             specs = dict(
#                 done=dict(size=(T,), dtype=torch.bool),
#                 episode_return=dict(size=(T,), dtype=torch.float32),
#                 target=dict(size=(T,), dtype=torch.float32),
#                 obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
#                 obs_action=dict(size=(T, 54), dtype=torch.int8),
#                 obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
#             )
#             _buffers: Buffers = {key: [] for key in specs}
#             for _ in range(flags.num_buffers):
#                 for key in _buffers:
#                     if not device == "cpu":
#                         _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
#                     else:
#                         _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
#                     _buffers[key].append(_buffer)
#             buffers[device][position] = _buffers
#     return buffers#嵌套字典，第一层是设备名称，第二层是位置名称，第三层是缓冲区名称



def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 319 if position == 'landlord' else 430
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
            )
            
            # 为农民角色添加协作奖励缓冲区
            if position in ['landlord_up', 'landlord_down']:
                specs['farmer_cooperation'] = dict(size=(T,), dtype=torch.float32)
                
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers



# #与环境交互的总函数
# def act(i, device, free_queue, full_queue, model, buffers, flags):
#     """
#     This function will run forever until we stop it. It will generate
#     data from the environment and send the data to buffer. It uses
#     a free queue and full queue to syncup with the main process.
#     """
#     positions = ['landlord', 'landlord_up', 'landlord_down']
#     try:
#         T = flags.unroll_length
#         #时间步
#         log.info('Device %s Actor %i started.', str(device), i)

#         env = create_env(flags)
#         env = Environment(env, device)
#         #创建封装环境

#         done_buf = {p: [] for p in positions}
#         episode_return_buf = {p: [] for p in positions}
#         target_buf = {p: [] for p in positions}
#         obs_x_no_action_buf = {p: [] for p in positions}
#         obs_action_buf = {p: [] for p in positions}
#         obs_z_buf = {p: [] for p in positions}
#         size = {p: 0 for p in positions}
#         #初始化缓冲区

#         position, obs, env_output = env.initial()
#         #初始化环境

#         while True:
#             while True:
#                 obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
#                 obs_z_buf[position].append(env_output['obs_z'])
#                 with torch.no_grad():
#                     agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
#                 _action_idx = int(agent_output['action'].cpu().detach().numpy())
#                 action = obs['legal_actions'][_action_idx]
#                 obs_action_buf[position].append(_cards2tensor(action))
#                 size[position] += 1
#                 position, obs, env_output = env.step(action)
#                 if env_output['done']:
#                     for p in positions:
#                         diff = size[p] - len(target_buf[p])
#                         if diff > 0:
#                             done_buf[p].extend([False for _ in range(diff-1)])
#                             done_buf[p].append(True)

#                             episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
#                             episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
#                             episode_return_buf[p].append(episode_return)
#                             target_buf[p].extend([episode_return for _ in range(diff)])
#                     break

#             for p in positions:
#                 while size[p] > T: 
#                     index = free_queue[p].get()
#                     if index is None:
#                         break
#                     for t in range(T):
#                         buffers[p]['done'][index][t, ...] = done_buf[p][t]
#                         buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
#                         buffers[p]['target'][index][t, ...] = target_buf[p][t]
#                         buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
#                         buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
#                         buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
#                     full_queue[p].put(index)
#                     done_buf[p] = done_buf[p][T:]
#                     episode_return_buf[p] = episode_return_buf[p][T:]
#                     target_buf[p] = target_buf[p][T:]
#                     obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
#                     obs_action_buf[p] = obs_action_buf[p][T:]
#                     obs_z_buf[p] = obs_z_buf[p][T:]
#                     size[p] -= T

#     except KeyboardInterrupt:
#         pass  
#     except Exception as e:
#         log.error('Exception in worker process %i', i)
#         traceback.print_exc()
#         print()
#         raise e


def act(i, device, free_queue, full_queue, model, buffers, flags):
    positions = ['landlord', 'landlord_up', 'landlord_down']
    farmer_positions = ['landlord_up', 'landlord_down']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        
        # 添加农民协作奖励缓冲区
        farmer_cooperation_buf = {p: [] for p in farmer_positions}
        
        # 添加任务跟踪状态
        task_state = {
            'consecutive_plays': {p: 0 for p in farmer_positions},  # 连续出牌计数
            'last_player': None,              # 上一个出牌的玩家
            'hand_cards': {p: [] for p in positions},  # 记录各玩家手牌
            'last_action': None,              # 上一个执行的动作
            'bomb_played': False,             # 是否出过炸弹
            'rocket_played': False,           # 是否出过火箭
            'landlord_pass_count': 0,         # 地主不出次数
            'rounds_history': []              # 回合历史记录
        }
        
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()
        
        # 初始化手牌信息
        for p in positions:
            if hasattr(env.env, 'info_sets') and p in env.env.info_sets:
                task_state['hand_cards'][p] = env.env.info_sets[p].player_hand_cards.copy()

        while True:
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                
                # 计算协作奖励（仅农民角色）
                cooperation_reward = 0.0
                
                if position in farmer_positions:
                    # 更新连续出牌计数
                    if task_state['last_player'] in farmer_positions and task_state['last_player'] != position:
                        # 另一个农民刚刚出牌，现在轮到当前农民
                        if action != []:  # 不是"不出"
                            task_state['consecutive_plays'][position] += 1
                            # 连续出牌奖励
                            cooperation_reward += 0.05 * task_state['consecutive_plays'][position]
                    else:
                        task_state['consecutive_plays'][position] = 1 if action != [] else 0
                    
                    # 任务1: 接队友的牌
                    other_farmer = 'landlord_up' if position == 'landlord_down' else 'landlord_down'
                    if task_state['last_player'] == other_farmer and action != []:
                        cooperation_reward += 0.1
                    
                    # 任务2: 压制地主的大牌
                    if task_state['last_player'] == 'landlord' and action != []:
                        # 判断是否是大牌（简化判断，实际上可以根据牌型进行更精确的判断）
                        if len(action) > 0 and max(action) >= 11:  # A及以上视为大牌
                            cooperation_reward += 0.15
                            # 如果是炸弹或火箭，给予更高奖励
                            if len(action) == 4 and len(set(action)) == 1:  # 炸弹
                                cooperation_reward += 0.2
                                task_state['bomb_played'] = True
                            elif set(action) == {14, 15}:  # 火箭（双王）
                                cooperation_reward += 0.3
                                task_state['rocket_played'] = True
                    
                    # 任务3: 如果地主不出牌，农民配合出牌
                    if task_state['landlord_pass_count'] > 0 and action != []:
                        cooperation_reward += 0.1 * min(task_state['landlord_pass_count'], 3)
                    
                    # 任务4: 根据手牌情况进行战略配合
                    if hasattr(env.env, 'info_sets'):
                        landlord_cards = len(task_state['hand_cards'].get('landlord', []))
                        other_farmer_cards = len(task_state['hand_cards'].get(other_farmer, []))
                        my_cards = len(task_state['hand_cards'].get(position, []))
                        
                        # 如果地主牌少，另一个农民牌多，当前农民应该出牌
                        if landlord_cards < 5 and other_farmer_cards > 5 and action != []:
                            cooperation_reward += 0.2
                        
                        # 如果当前农民牌少，另一个农民牌多，对方应该主动出牌
                        if my_cards < other_farmer_cards and task_state['last_player'] == other_farmer and action != []:
                            cooperation_reward += 0.15
                
                # 更新任务状态
                if action != []:  # 如果不是"不出"
                    # 更新手牌
                    if hasattr(env.env, 'info_sets') and position in env.env.info_sets:
                        task_state['hand_cards'][position] = [
                            card for card in task_state['hand_cards'][position] if card not in action
                        ]
                    
                    task_state['last_player'] = position
                    task_state['last_action'] = action
                    task_state['landlord_pass_count'] = 0
                else:
                    # 如果是地主不出
                    if position == 'landlord':
                        task_state['landlord_pass_count'] += 1
                
                # 保存农民协作奖励
                if position in farmer_positions:
                    farmer_cooperation_buf[position].append(cooperation_reward)
                
                size[position] += 1
                position, obs, env_output = env.step(action)
                
                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)

                            episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
                            
                            # 农民赢得游戏时额外奖励
                            if p in farmer_positions and episode_return > 0:
                                # 胜利奖励，根据使用的合作策略增加
                                win_bonus = 0.3
                                if task_state['bomb_played']:
                                    win_bonus += 0.2
                                if task_state['rocket_played']:
                                    win_bonus += 0.3
                                
                                # 填充协作奖励缓冲区，最后一步给予胜利奖励
                                if len(farmer_cooperation_buf[p]) > 0:
                                    farmer_cooperation_buf[p][-1] += win_bonus
                            
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    
                    # 重置任务状态
                    task_state = {
                        'consecutive_plays': {p: 0 for p in farmer_positions},
                        'last_player': None,
                        'hand_cards': {p: [] for p in positions},
                        'last_action': None,
                        'bomb_played': False,
                        'rocket_played': False,
                        'landlord_pass_count': 0,
                        'rounds_history': []
                    }
                    break

            for p in positions:
                while size[p] > T: 
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                        
                        # 添加农民协作奖励
                        if p in farmer_positions and t < len(farmer_cooperation_buf[p]):
                            if 'farmer_cooperation' in buffers[p]:
                                buffers[p]['farmer_cooperation'][index][t, ...] = farmer_cooperation_buf[p][t]
                    
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    
                    # 更新农民协作奖励缓冲区
                    if p in farmer_positions:
                        farmer_cooperation_buf[p] = farmer_cooperation_buf[p][T:] if len(farmer_cooperation_buf[p]) > T else []
                    
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e



def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix
