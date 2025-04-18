import multiprocessing as mp
import pickle
import numpy as np

from douzero.env.game import GameEnv

def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players

def mp_simulate(card_play_data_list, card_play_model_path_dict, q):
    players = load_card_play_models(card_play_model_path_dict)

    env = GameEnv(players)
    
    # 添加协作行为跟踪
    cooperation_stats = {
        'consecutive_plays': 0,           # 农民连续出牌次数
        'counter_big_cards': 0,           # 农民压制地主大牌次数
        'successful_coordination': 0,     # 成功配合次数
        'bomb_rocket_played': 0,          # 出炸弹或火箭次数
        'total_games': 0,                 # 总局数
    }

    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        
        # 初始化单局游戏的协作跟踪
        game_cooperation = {
            'last_player': None,
            'consecutive_count': 0,
            'big_card_counter': 0,
            'coordination_events': 0,
            'bomb_rocket_count': 0,
            'hand_cards': {}
        }
        
        # 如果环境支持，记录初始手牌
        if hasattr(env, 'info_sets'):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                if position in env.info_sets:
                    game_cooperation['hand_cards'][position] = env.info_sets[position].player_hand_cards.copy()
        
        while not env.game_over:
            # 记录上一个出牌者
            last_position = env.acting_player_position
            last_action = env.get_last_move()
            
            # 执行动作
            env.step()
            
            # 当前出牌者
            current_position = env.acting_player_position
            current_action = env.get_last_move()
            
            # 分析协作行为
            if current_position in ['landlord_up', 'landlord_down'] and last_position in ['landlord_up', 'landlord_down']:
                # 检测连续出牌
                if current_action != [] and current_position != last_position:
                    game_cooperation['consecutive_count'] += 1
                    cooperation_stats['consecutive_plays'] += 1
            
            # 检测农民压制地主大牌
            if last_position == 'landlord' and current_position in ['landlord_up', 'landlord_down'] and current_action != []:
                # 简单判断是否是大牌
                if max(current_action) >= 11:  # A及以上视为大牌
                    game_cooperation['big_card_counter'] += 1
                    cooperation_stats['counter_big_cards'] += 1
                
                # 检测炸弹和火箭
                if len(current_action) == 4 and len(set(current_action)) == 1:  # 炸弹
                    game_cooperation['bomb_rocket_count'] += 1
                    cooperation_stats['bomb_rocket_played'] += 1
                elif set(current_action) == {14, 15}:  # 火箭（双王）
                    game_cooperation['bomb_rocket_count'] += 1
                    cooperation_stats['bomb_rocket_played'] += 1
            
            # 检测成功配合（简化版）
            if current_position in ['landlord_up', 'landlord_down'] and current_action != []:
                other_farmer = 'landlord_up' if current_position == 'landlord_down' else 'landlord_down'
                if hasattr(env, 'info_sets'):
                    landlord_cards_count = len(env.info_sets.get('landlord', {}).player_hand_cards or [])
                    if landlord_cards_count < 5:  # 地主牌数少时的积极出牌
                        game_cooperation['coordination_events'] += 1
                        cooperation_stats['successful_coordination'] += 1
                
                # 如果前一个是队友，现在接牌
                if last_position == other_farmer and last_action != [] and current_action != []:
                    game_cooperation['coordination_events'] += 1
                    cooperation_stats['successful_coordination'] += 1
            
            # 更新状态
            game_cooperation['last_player'] = current_position

        cooperation_stats['total_games'] += 1
        env.reset()

    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer'],
           cooperation_stats
         ))

def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers):
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0
    
    # 添加协作统计聚合
    total_cooperation_stats = {
        'consecutive_plays': 0,
        'counter_big_cards': 0,
        'successful_coordination': 0,
        'bomb_rocket_played': 0,
        'total_games': 0,
    }

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
                target=mp_simulate,
                args=(card_paly_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]
        
        # 聚合协作统计
        worker_coop_stats = result[4]
        for key in total_cooperation_stats:
            total_cooperation_stats[key] += worker_coop_stats[key]

    num_total_wins = num_landlord_wins + num_farmer_wins
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins))
    
    # 打印农民协作统计
    print('\nFarmer Cooperation Statistics:')
    if total_cooperation_stats['total_games'] > 0:
        avg_consecutive = total_cooperation_stats['consecutive_plays'] / total_cooperation_stats['total_games']
        avg_counter_big = total_cooperation_stats['counter_big_cards'] / total_cooperation_stats['total_games']
        avg_coordination = total_cooperation_stats['successful_coordination'] / total_cooperation_stats['total_games']
        avg_bomb_rocket = total_cooperation_stats['bomb_rocket_played'] / total_cooperation_stats['total_games']
        
        print('Average consecutive plays per game: {:.2f}'.format(avg_consecutive))
        print('Average counter big cards per game: {:.2f}'.format(avg_counter_big))
        print('Average successful coordination per game: {:.2f}'.format(avg_coordination))
        print('Average bomb/rocket plays per game: {:.2f}'.format(avg_bomb_rocket))
        
        # 农民胜率与协作行为的关联分析
        farmer_wp = num_farmer_wins / num_total_wins
        print('\nFarmer win probability: {:.2f}'.format(farmer_wp))
        print('Coordination level: {:.2f}'.format((avg_consecutive + avg_counter_big + avg_coordination) / 3))
