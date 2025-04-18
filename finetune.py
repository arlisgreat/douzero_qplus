import os
import argparse
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

# 假设这些是从douzero导入的模块
from douzero.env.env import Env
from douzero.dmc.utils import _cards2tensor
from douzero.dmc.file_writer import FileWriter
from douzero.dmc.models import Model

import logging

# 创建日志记录器
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('finetune')

# 创建平均回报缓冲区
mean_episode_return_buf = {p: deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}


def compute_loss(logits, targets):
    """计算损失函数：均方误差MSE"""
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def create_buffers(flags, device_iterator):
    """
    创建缓冲区，为农民角色添加协作奖励缓冲区
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

            _buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if device != "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


def create_optimizers(flags, learner_model):
    """
    创建优化器，为了微调，使用较小的学习率
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.Adam(
            learner_model.get_model(position).parameters(),
            lr=flags.learning_rate * 0.1,  # 微调时降低学习率
            betas=(0, 0.99),
            eps=flags.epsilon,
        )
        optimizers[position] = optimizer
    return optimizers


def create_env(flags):
    """创建游戏环境"""
    from douzero.env.env import Env
    return Env(flags.objective)

def get_batch(free_queue, full_queue, buffers, flags, lock):
    with lock:
        indices = [int(full_queue.get()) for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    收集经验并计算农民协作奖励（修改后版本，适配新版 Env 类及正确转换 numpy 数组为 torch.Tensor）
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    farmer_positions = ['landlord_up', 'landlord_down']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        # 创建环境
        env = create_env(flags)

        # 初始化各类缓冲区
        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        farmer_cooperation_buf = {p: [] for p in farmer_positions}

        # 初始化任务状态
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

        size = {p: 0 for p in positions}

        # 使用 reset() 得到初始观测，新版本 env 返回观测字典
        obs = env.reset()
        position = obs['position']

        # 若有可能，初始化手牌信息
        for p in positions:
            if hasattr(env, 'infoset') and env.infoset is not None and hasattr(env.infoset, 'player_hand_cards'):
                task_state['hand_cards'][p] = env.infoset.player_hand_cards.copy()

        while True:
            while True:
                # 保存缓冲区数据，直接使用观测中的 numpy 数组
                obs_x_no_action_buf[position].append(obs['x_no_action'])
                obs_z_buf[position].append(obs['z'])

                # 将观测中的 x_batch 和 z_batch 转换为 torch.Tensor
                device_obj = torch.device('cuda:' + str(device)) if device != "cpu" else torch.device('cpu')
                obs_z_batch = torch.from_numpy(obs['z_batch']).to(device_obj)
                obs_x_batch = torch.from_numpy(obs['x_batch']).to(device_obj)

                # 模型前向传播（注意输入已转换为 tensor）
                with torch.no_grad():
                    agent_output = model.forward(position, obs_z_batch, obs_x_batch, flags=flags)

                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))

                # 计算协作奖励（逻辑保持不变）
                cooperation_reward = 0.0
                if position in farmer_positions:
                    if task_state['last_player'] in farmer_positions and task_state['last_player'] != position:
                        if action != []:
                            task_state['consecutive_plays'][position] += 1
                            cooperation_reward += 0.05 * task_state['consecutive_plays'][position]
                    else:
                        task_state['consecutive_plays'][position] = 1 if action != [] else 0

                    other_farmer = 'landlord_up' if position == 'landlord_down' else 'landlord_down'
                    if task_state['last_player'] == other_farmer and action != []:
                        cooperation_reward += 0.1

                    if task_state['last_player'] == 'landlord' and action != []:
                        if len(action) > 0 and max(action) >= 11:
                            cooperation_reward += 0.15
                            if len(action) == 4 and len(set(action)) == 1:
                                cooperation_reward += 0.2
                                task_state['bomb_played'] = True
                            elif set(action) == {14, 15}:
                                cooperation_reward += 0.3
                                task_state['rocket_played'] = True

                    if task_state['landlord_pass_count'] > 0 and action != []:
                        cooperation_reward += 0.1 * min(task_state['landlord_pass_count'], 3)

                    if hasattr(env, 'infoset'):
                        landlord_cards = len(task_state['hand_cards'].get('landlord', []))
                        other_farmer_cards = len(task_state['hand_cards'].get(other_farmer, []))
                        my_cards = len(task_state['hand_cards'].get(position, []))
                        if landlord_cards < 5 and other_farmer_cards > 5 and action != []:
                            cooperation_reward += 0.2
                        if my_cards < other_farmer_cards and task_state['last_player'] == other_farmer and action != []:
                            cooperation_reward += 0.15

                if action != []:
                    if hasattr(env, 'infoset') and env.infoset is not None and hasattr(env.infoset, 'player_hand_cards'):
                        task_state['hand_cards'][position] = [
                            card for card in task_state['hand_cards'][position] if card not in action
                        ]
                    task_state['last_player'] = position
                    task_state['last_action'] = action
                    task_state['landlord_pass_count'] = 0
                else:
                    if position == 'landlord':
                        task_state['landlord_pass_count'] += 1

                if position in farmer_positions:
                    farmer_cooperation_buf[position].append(cooperation_reward)

                size[position] += 1

                # 使用新版 step 接口：返回 (obs, reward, done, info)
                obs, reward, done, info = env.step(action)

                if done:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False] * (diff - 1))
                            done_buf[p].append(True)
                            # 地主与农民获得的回报设置不同
                            episode_return = reward if p == 'landlord' else -reward
                            if p in farmer_positions and episode_return > 0:
                                win_bonus = 0.3
                                if task_state['bomb_played']:
                                    win_bonus += 0.2
                                if task_state['rocket_played']:
                                    win_bonus += 0.3
                                if len(farmer_cooperation_buf[p]) > 0:
                                    farmer_cooperation_buf[p][-1] += win_bonus
                            episode_return_buf[p].extend([0.0] * (diff - 1))
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return] * diff)
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
                else:
                    # 更新当前角色
                    position = obs['position']

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
                    if p in farmer_positions:
                        farmer_cooperation_buf[p] = farmer_cooperation_buf[p][T:] if len(farmer_cooperation_buf[p]) > T else []
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        import traceback
        traceback.print_exc()
        print()
        raise e

def learn(position, actor_models, model, batch, optimizer, flags, lock):
    """
    执行学习步骤，添加农民协作奖励
    """
    if flags.gpu_devices != "cpu":
        device = torch.device('cuda:' + str(flags.gpu_devices))
    else:
        device = torch.device('cpu')

    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)

    # 处理农民协作奖励
    farmer_cooperation = None
    if position in ['landlord_up', 'landlord_down'] and 'farmer_cooperation' in batch:
        farmer_cooperation = torch.flatten(batch['farmer_cooperation'].to(device), 0, 1)
        cooperation_weight = flags.cooperation_weight
        if farmer_cooperation is not None:
            target = target + cooperation_weight * farmer_cooperation

    # 平均回报存储到缓冲区
    episode_returns = batch['episode_return'][batch['done']]
    if len(episode_returns) > 0:
        mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)

        stats = {
            'mean_episode_return_' + position: torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_' + position: loss.item(),
        }

        if position in ['landlord_up', 'landlord_down'] and farmer_cooperation is not None:
            stats['cooperation_reward_' + position] = torch.mean(farmer_cooperation).item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats


def batch_and_learn(i, device, position, local_lock, position_lock, free_queue, full_queue, actor_models, learner_model, buffers, flags):
    """批处理和学习的线程目标函数"""
    while True:
        batch = get_batch(free_queue[position], full_queue[position], buffers[device][position], flags, local_lock)
        learn(position, actor_models, learner_model.get_model(position), batch,
              learner_model.get_optimizer(position), flags, position_lock)


def finetune(flags):
    """
    主微调函数，加载预训练模型并进行微调
    """
    if not flags.actor_device_cpu or flags.gpu_devices != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `--actor_device_cpu --training_device cpu`")

    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar'))
    )

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # 初始化模型
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # 加载预训练模型权重
    if flags.landlord:
        weights = torch.load(flags.landlord, map_location='cpu')
        models[device_iterator[0]].get_model('landlord').load_state_dict(weights)
        log.info(f"Loaded landlord model from {flags.landlord}")

    if flags.landlord_up:
        weights = torch.load(flags.landlord_up, map_location='cpu')
        models[device_iterator[0]].get_model('landlord_up').load_state_dict(weights)
        log.info(f"Loaded landlord_up model from {flags.landlord_up}")

    if flags.landlord_down:
        weights = torch.load(flags.landlord_down, map_location='cpu')
        models[device_iterator[0]].get_model('landlord_down').load_state_dict(weights)
        log.info(f"Loaded landlord_down model from {flags.landlord_down}")

    # 复制模型到所有设备
    for device in device_iterator:
        if device != device_iterator[0]:
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                models[device].get_model(position).load_state_dict(
                    models[device_iterator[0]].get_model(position).state_dict()
                )

    # 初始化缓冲区
    buffers = create_buffers(flags, device_iterator)

    # 初始化队列
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}

    for device in device_iterator:
        _free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        _full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # 学习器模型
    learner_model = Model(device=flags.gpu_devices)

    # 从预训练模型复制权重到学习器模型
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        learner_model.get_model(position).load_state_dict(
            models[device_iterator[0]].get_model(position).state_dict()
        )

    # 创建优化器并设置到学习器模型
    optimizers = create_optimizers(flags, learner_model)
    learner_model.optimizers = optimizers
    learner_model.get_optimizer = lambda position: learner_model.optimizers[position]

    # 统计键
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'cooperation_reward_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
        'cooperation_reward_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}

    # 如果存在检查点，则加载
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:" + str(flags.gpu_devices) if flags.gpu_devices != "cpu" else "cpu")
        )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # 启动演员进程
    for device in device_iterator:
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags)
            )
            actor.start()
            actor_processes.append(actor)

    # 填充空闲队列
    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['landlord'].put(m)
            free_queue[device]['landlord_up'].put(m)
            free_queue[device]['landlord_down'].put(m)

    # 启动学习线程
    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}
    position_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                thread = threading.Thread(
                    target=batch_and_learn,
                    name=f'batch-and-learn-{position}-{i}',
                    args=(i, device, position, locks[device][position], position_locks[position],
                          free_queue[device], full_queue[device], models,
                          learner_model, buffers[device], flags)
                )
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        """保存检查点"""
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position + '_weights_' + str(frames) + '.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    # 主微调循环
    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in position_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     position_frames['landlord'],
                     position_frames['landlord_up'],
                     position_frames['landlord_down'],
                     fps,
                     fps_avg,
                     position_fps['landlord'],
                     position_fps['landlord_up'],
                     position_fps['landlord_down'],
                     pprint.pformat(stats))

            frames += int(fps * 5)
            for p in position_frames:
                position_frames[p] += int(position_fps[p] * 5)

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Finetuning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()

    # 关闭演员进程
    for actor in actor_processes:
        actor.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DouZero: Farmer Cooperation Finetuning')
    parser.add_argument('--xpid', default='douzero-farmer-coop',
                        help='Experiment id (default: douzero-farmer-coop)')
    parser.add_argument('--savedir', default='checkpoints',
                        help='Root dir where experiment data will be saved')
    parser.add_argument('--num_actors', default=5, type=int, metavar='N',
                        help='Number of actors (default: 5)')
    parser.add_argument('--gpu_devices', default='1', type=str,
                        help='Comma separated list of available GPU devices (default: 0)')
    parser.add_argument('--actor_device_cpu', action='store_true',
                        help='Use CPU as actor device')
    parser.add_argument('--num_actor_devices', default=1, type=int,
                        help='The number of devices used for actor')
    parser.add_argument('--num_buffers', default=50, type=int,
                        help='Number of shared-memory buffers (default: 50)')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='Batch size for training (default: 32)')
    parser.add_argument('--unroll_length', default=100, type=int, metavar='N',
                        help='Sequence length for each update (default: 100)')
    parser.add_argument('--total_frames', default=100000000, type=int, metavar='N',
                        help='Total training frames (default: 5000000)')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learning rate for fine-tuning (default: 1e-4)')
    parser.add_argument('--epsilon', default=1e-3, type=float,
                        help='Optimizer epsilon (default: 1e-3)')
    parser.add_argument('--max_grad_norm', default=40, type=float,
                        help='Maximum gradient norm (default: 40)')
    parser.add_argument('--cooperation_weight', default=0.01, type=float,
                        help='Weight for the cooperation reward (default: 0.01)')
    parser.add_argument('--objective', default='default-objective', type=str,
                        help='Objective for environment (default: default-objective)')
    parser.add_argument('--load_model', action='store_true',
                        help='Load model from checkpoint')
    parser.add_argument('--landlord', default='baselines/douzero_ADP/landlord.ckpt', type=str,
                        help='Path to pretrained landlord model')
    parser.add_argument('--landlord_up', default='baselines/douzero_ADP/landlord_up.ckpt', type=str,
                        help='Path to pretrained landlord_up model')
    parser.add_argument('--landlord_down', default='baselines/douzero_ADP/landlord_down.ckpt', type=str,
                        help='Path to pretrained landlord_down model')
    parser.add_argument('--save_interval', default=30, type=int,
                        help='Checkpoint save interval in minutes (default: 10)')
    parser.add_argument('--num_threads', default=4, type=int,
                        help='Number of learner threads (default: 2)')
    parser.add_argument('--disable_checkpoint', action='store_true',
                        help='Disable checkpoint saving')
    parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')

    flags = parser.parse_args()

    finetune(flags)