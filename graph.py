import csv
import matplotlib.pyplot as plt
import numpy as np

def read_csv(file_path):
    """读取 CSV 文件并返回数据列表"""
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def plot_graph(x, y, xlabel, ylabel, title, output_file):
    """生成并保存图表"""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_file, dpi=300)
    plt.close()

def generate_plots():
    # 读取 CSV 文件
    file_path = 'evaluate_results_2.csv'
    data = read_csv(file_path)
    
    # 提取 frames 和各个指标
    frames = [float(row['frames_value']) for row in data]
    wp_landlord = [float(row['wp_landlord']) if row['wp_landlord'] else None for row in data]
    wp_farmers = [float(row['wp_farmers']) if row['wp_farmers'] else None for row in data]
    adp_landlord = [float(row['adp_landlord']) if row['adp_landlord'] else None for row in data]
    adp_farmers = [float(row['adp_farmers']) if row['adp_farmers'] else None for row in data]
    consecutive_plays = [float(row['Average consecutive plays per game']) if row['Average consecutive plays per game'] else None for row in data]
    counter_big_cards = [float(row['Average counter big cards per game']) if row['Average counter big cards per game'] else None for row in data]
    successful_coordination = [float(row['Average successful coordination per game']) if row['Average successful coordination per game'] else None for row in data]
    bomb_rocket_plays = [float(row['Average bomb/rocket plays per game']) if row['Average bomb/rocket plays per game'] else None for row in data]

    # 创建各个图表并保存
    plot_graph(frames, wp_landlord, 'Frames', 'WP Landlord', 'WP Landlord vs Frames', 'wp_landlord_vs_frames_2.png')
    plot_graph(frames, wp_farmers, 'Frames', 'WP Farmers', 'WP Farmers vs Frames', 'wp_farmers_vs_frames_2.png')
    plot_graph(frames, adp_landlord, 'Frames', 'ADP Landlord', 'ADP Landlord vs Frames', 'adp_landlord_vs_frames_2.png')
    plot_graph(frames, adp_farmers, 'Frames', 'ADP Farmers', 'ADP Farmers vs Frames', 'adp_farmers_vs_frames_2.png')
    plot_graph(frames, consecutive_plays, 'Frames', 'Average Consecutive Plays', 'Average Consecutive Plays vs Frames', 'consecutive_plays_vs_frames_2.png')
    plot_graph(frames, counter_big_cards, 'Frames', 'Average Counter Big Cards', 'Average Counter Big Cards vs Frames', 'counter_big_cards_vs_frames_2.png')
    plot_graph(frames, successful_coordination, 'Frames', 'Average Successful Coordination', 'Average Successful Coordination vs Frames', 'successful_coordination_vs_frames_2.png')
    plot_graph(frames, bomb_rocket_plays, 'Frames', 'Average Bomb/Rocket Plays', 'Average Bomb/Rocket Plays vs Frames', 'bomb_rocket_plays_vs_frames_2.png')

    print("图表已保存.")

if __name__ == "__main__":
    generate_plots()