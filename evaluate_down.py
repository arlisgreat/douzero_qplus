#!/usr/bin/env python3
import os
import csv
import subprocess
import logging
import time
import re

def setup_logging():
    # 配置 logging 输出到 debug.log，级别为 DEBUG
    logging.basicConfig(filename="debug02.log",
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.debug("日志初始化成功")

def extract_number(filename):
    try:
        return int(filename.split('_')[-1].replace('.ckpt', ''))
    except Exception as e:
        logging.error(f"提取数字失败 {filename}: {e}")
        return 0

def parse_output(output):
    """
    解析 evaluate.py 的标准输出，提取需要的数据并返回字典
    """
    # 解析 WP results
    wp_match = re.search(r'WP results:\nlandlord\s*:\s*Farmers\s*-\s*([0-9\.\-]+)\s*:\s*([0-9\.\-]+)', output)
    wp_landlord = wp_match.group(1) if wp_match else None
    wp_farmers = wp_match.group(2) if wp_match else None

    # 解析 ADP results
    adp_match = re.search(r'ADP results:\nlandlord\s*:\s*Farmers\s*-\s*([0-9\.\-]+)\s*:\s*([0-9\.\-]+)', output)
    adp_landlord = adp_match.group(1) if adp_match else None
    adp_farmers = adp_match.group(2) if adp_match else None

    # 解析 Farmer Cooperation Statistics
    coop_stats = {}
    coop_match = re.findall(r'Average (.*?):\s*([0-9\.\-]+)', output)
    for stat, value in coop_match:
        # 修正字段名以匹配 CSV 中的字段名
        coop_stats[f"Average {stat}"] = value

    # 返回一个字典，包含解析出来的数据
    return {
        'wp_landlord': wp_landlord,
        'wp_farmers': wp_farmers,
        'adp_landlord': adp_landlord,
        'adp_farmers': adp_farmers,
        **coop_stats
    }

def main():
    setup_logging()
    
    # ckpt 文件所在文件夹，使用相对路径
    ckpt_folder = "./douzero_checkpoints/douzero_0/"
    
    # 获取所有以 "landlord_down" 开头、以 ".ckpt" 结尾的文件
    ckpt_files = [f for f in os.listdir(ckpt_folder)
                  if f.startswith("landlord_down") and f.endswith(".ckpt")]
    
    if not ckpt_files:
        logging.debug("没有找到以 'landlord_down' 开头的 .ckpt 文件")
        print("没有找到目标 ckpt 文件")
        return

    # 按文件名中数字部分排序
    ckpt_files = sorted(ckpt_files, key=extract_number)
    logging.debug(f"找到 {len(ckpt_files)} 个 ckpt 文件")
    
    # 保存测试结果的 CSV 文件
    results_csv = "evaluate_results_0.csv"
    # CSV 头部信息
    csv_fieldnames = [
        "ckpt_file", "frames_value", "command", 
        "wp_landlord", "wp_farmers", "adp_landlord", "adp_farmers", 
        "Average consecutive plays per game", 
        "Average counter big cards per game", 
        "Average successful coordination per game", 
        "Average bomb/rocket plays per game"
    ]
    
    # 检查文件是否存在且是否有表头
    file_exists = os.path.exists(results_csv)
    write_header = not file_exists

    # 循环处理每个 ckpt 文件
    for ckpt_file in ckpt_files:
        full_ckpt_path = os.path.join(ckpt_folder, ckpt_file)
        # 提取 frames_value，假设文件名格式为 landlord_down_weights_数字.ckpt
        try:
            frames_value = ckpt_file.split('_')[-1].replace('.ckpt', '')
        except Exception as e:
            logging.error(f"无法提取 frames_value from {ckpt_file}: {e}")
            frames_value = ""
        
        command = (
            f"python3 evaluate.py "
            f"--landlord baselines/sl/landlord.ckpt "
            f"--landlord_up baselines/sl/landlord_up.ckpt "
            f"--landlord_down {full_ckpt_path}"
        )
        logging.debug(f"开始测试：{full_ckpt_path}")
        logging.debug(f"运行命令：{command}")
        print(f"正在测试：{full_ckpt_path}")
        
        try:
            # 使用 subprocess.Popen 来运行命令，并确保等待进程完成
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()  # 等待命令完成并获取输出
            
            # 获取返回码
            returncode = process.returncode
            stdout = stdout.decode("utf-8").strip()
            stderr = stderr.decode("utf-8").strip()
            
            logging.debug(f"完成 {ckpt_file}，返回码：{returncode}")
            if stdout:
                logging.debug(f"stdout: {stdout}")
            if stderr:
                logging.error(f"stderr: {stderr}")
            
            # 解析标准输出
            result_data = parse_output(stdout)
            result_data['ckpt_file'] = ckpt_file
            result_data['frames_value'] = frames_value
            result_data['command'] = command
            
        except Exception as e:
            logging.error(f"调用 {full_ckpt_path} 时出错: {e}")
            result_data = {
                'ckpt_file': ckpt_file,
                'frames_value': frames_value,
                'command': command,
                'wp_landlord': None,
                'wp_farmers': None,
                'adp_landlord': None,
                'adp_farmers': None,
                'Average consecutive plays per game': None,
                'Average counter big cards per game': None,
                'Average successful coordination per game': None,
                'Average bomb/rocket plays per game': None
            }
        
        # 每次测试后立即写入 CSV 文件
        try:
            with open(results_csv, "a", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
                
                if write_header:  # 如果文件不存在或没有表头，则写入表头
                    writer.writeheader()
                
                writer.writerow(result_data)  # 写入当前测试的结果

            logging.debug(f"当前测试结果已保存到 {results_csv}")
            print(f"当前测试结果保存到 {results_csv}")
        
        except Exception as e:
            logging.error(f"写入 CSV 文件失败: {e}")
            print(f"写入 CSV 文件失败: {e}")
        
        # 可选：每次调用后稍作等待，便于观察和资源释放
        time.sleep(5)

if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# import os
# import csv
# import subprocess
# import logging
# import time
# import re

# def setup_logging():
#     # 配置 logging 输出到 debug.log，级别为 DEBUG
#     logging.basicConfig(filename="debug0.log",
#                         level=logging.DEBUG,
#                         format='%(asctime)s %(levelname)s: %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S')
#     logging.debug("日志初始化成功")

# def extract_number(filename):
#     try:
#         return int(filename.split('_')[-1].replace('.ckpt', ''))
#     except Exception as e:
#         logging.error(f"提取数字失败 {filename}: {e}")
#         return 0

# def parse_output(output):
#     """
#     解析 evaluate.py 的标准输出，提取需要的数据并返回字典
#     """
#     # 解析 WP results
#     wp_match = re.search(r'WP results:\nlandlord\s*:\s*Farmers\s*-\s*([0-9\.\-]+)\s*:\s*([0-9\.\-]+)', output)
#     wp_landlord = wp_match.group(1) if wp_match else None
#     wp_farmers = wp_match.group(2) if wp_match else None

#     # 解析 ADP results
#     adp_match = re.search(r'ADP results:\nlandlord\s*:\s*Farmers\s*-\s*([0-9\.\-]+)\s*:\s*([0-9\.\-]+)', output)
#     adp_landlord = adp_match.group(1) if adp_match else None
#     adp_farmers = adp_match.group(2) if adp_match else None

#     # 解析 Farmer Cooperation Statistics
#     coop_stats = {}
#     coop_match = re.findall(r'Average (.*?):\s*([0-9\.\-]+)', output)
#     for stat, value in coop_match:
#         coop_stats[stat] = value

#     # 返回一个字典，包含解析出来的数据
#     return {
#         'wp_landlord': wp_landlord,
#         'wp_farmers': wp_farmers,
#         'adp_landlord': adp_landlord,
#         'adp_farmers': adp_farmers,
#         **coop_stats
#     }

# def main():
#     setup_logging()
    
#     # ckpt 文件所在文件夹，使用相对路径
#     ckpt_folder = "./douzero_checkpoints/douzero/"
    
#     # 获取所有以 "landlord_down" 开头、以 ".ckpt" 结尾的文件
#     ckpt_files = [f for f in os.listdir(ckpt_folder)
#                   if f.startswith("landlord_down") and f.endswith(".ckpt")]
    
#     if not ckpt_files:
#         logging.debug("没有找到以 'landlord_down' 开头的 .ckpt 文件")
#         print("没有找到目标 ckpt 文件")
#         return

#     # 按文件名中数字部分排序
#     ckpt_files = sorted(ckpt_files, key=extract_number)
#     logging.debug(f"找到 {len(ckpt_files)} 个 ckpt 文件")
    
#     # 保存测试结果的 CSV 文件
#     results_csv = "evaluate_results.csv"
#     # CSV 头部信息
#     csv_fieldnames = ["ckpt_file", "frames_value", "command", "wp_landlord", "wp_farmers", "adp_landlord", "adp_farmers", "Average consecutive plays per game", "Average counter big cards per game", "Average successful coordination per game", "Average bomb/rocket plays per game"]
#     result_rows = []
    
#     # 检查文件是否存在且是否有表头
#     file_exists = os.path.exists(results_csv)
#     write_header = not file_exists

#     # 循环处理每个 ckpt 文件
#     for ckpt_file in ckpt_files:
#         full_ckpt_path = os.path.join(ckpt_folder, ckpt_file)
#         # 提取 frames_value，假设文件名格式为 landlord_down_weights_数字.ckpt
#         try:
#             frames_value = ckpt_file.split('_')[-1].replace('.ckpt', '')
#         except Exception as e:
#             logging.error(f"无法提取 frames_value from {ckpt_file}: {e}")
#             frames_value = ""
        
#         command = (
#             f"python3 evaluate.py "
#             f"--landlord baselines/sl/landlord.ckpt "
#             f"--landlord_up baselines/sl/landlord_up.ckpt "
#             f"--landlord_down {full_ckpt_path}"
#         )
#         logging.debug(f"开始测试：{full_ckpt_path}")
#         logging.debug(f"运行命令：{command}")
#         print(f"正在测试：{full_ckpt_path}")
        
#         try:
#             # 使用 subprocess.Popen 来运行命令，并确保等待进程完成
#             process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             stdout, stderr = process.communicate()  # 等待命令完成并获取输出
            
#             # 获取返回码
#             returncode = process.returncode
#             stdout = stdout.decode("utf-8").strip()
#             stderr = stderr.decode("utf-8").strip()
            
#             logging.debug(f"完成 {ckpt_file}，返回码：{returncode}")
#             if stdout:
#                 logging.debug(f"stdout: {stdout}")
#             if stderr:
#                 logging.error(f"stderr: {stderr}")
            
#             # 解析标准输出
#             result_data = parse_output(stdout)
#             result_data['ckpt_file'] = ckpt_file
#             result_data['frames_value'] = frames_value
#             result_data['command'] = command
            
#         except Exception as e:
#             logging.error(f"调用 {full_ckpt_path} 时出错: {e}")
#             result_data = {
#                 'ckpt_file': ckpt_file,
#                 'frames_value': frames_value,
#                 'command': command,
#                 'wp_landlord': None,
#                 'wp_farmers': None,
#                 'adp_landlord': None,
#                 'adp_farmers': None,
#                 'Average consecutive plays per game': None,
#                 'Average counter big cards per game': None,
#                 'Average successful coordination per game': None,
#                 'Average bomb/rocket plays per game': None
#             }
        
#         # 保存结果到列表中
#         result_rows.append(result_data)
        
#         # 可选：每次调用后稍作等待，便于观察和资源释放
#         time.sleep(5)
    
#     # 写入 CSV 文件保存测试结果
#     try:
#         with open(results_csv, "a", newline='', encoding="utf-8") as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            
#             if write_header:  # 如果文件不存在或没有表头，则写入表头
#                 writer.writeheader()

#             for row in result_rows:
#                 writer.writerow(row)
        
#         logging.debug(f"测试结果已保存到 {results_csv}")
#         print(f"测试结果保存到 {results_csv}")
#     except Exception as e:
#         logging.error(f"写入 CSV 文件失败: {e}")
#         print(f"写入 CSV 文件失败: {e}")

# if __name__ == "__main__":
#     main()
    
    
    
    
# #!/usr/bin/env python3
# import os
# import csv
# import subprocess
# import logging
# import time
# import re

# def setup_logging():
#     # 配置 logging 输出到 debug.log，级别为 DEBUG
#     logging.basicConfig(filename="debug.log",
#                         level=logging.DEBUG,
#                         format='%(asctime)s %(levelname)s: %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S')
#     logging.debug("日志初始化成功")

# def extract_number(filename):
#     try:
#         return int(filename.split('_')[-1].replace('.ckpt', ''))
#     except Exception as e:
#         logging.error(f"提取数字失败 {filename}: {e}")
#         return 0

# def parse_output(output):
#     """
#     解析 evaluate.py 的标准输出，提取需要的数据并返回字典
#     """
#     # 解析 WP results
#     wp_match = re.search(r'WP results:\nlandlord\s*:\s*Farmers\s*-\s*([0-9\.\-]+)\s*:\s*([0-9\.\-]+)', output)
#     wp_landlord = wp_match.group(1) if wp_match else None
#     wp_farmers = wp_match.group(2) if wp_match else None

#     # 解析 ADP results
#     adp_match = re.search(r'ADP results:\nlandlord\s*:\s*Farmers\s*-\s*([0-9\.\-]+)\s*:\s*([0-9\.\-]+)', output)
#     adp_landlord = adp_match.group(1) if adp_match else None
#     adp_farmers = adp_match.group(2) if adp_match else None

#     # 解析 Farmer Cooperation Statistics
#     coop_stats = {}
#     coop_match = re.findall(r'Average (.*?):\s*([0-9\.\-]+)', output)
#     for stat, value in coop_match:
#         coop_stats[stat] = value

#     # 返回一个字典，包含解析出来的数据
#     return {
#         'wp_landlord': wp_landlord,
#         'wp_farmers': wp_farmers,
#         'adp_landlord': adp_landlord,
#         'adp_farmers': adp_farmers,
#         **coop_stats
#     }

# def main():
#     setup_logging()
    
#     # ckpt 文件所在文件夹，使用相对路径
#     ckpt_folder = "./douzero_checkpoints/douzero/"
    
#     # 获取所有以 "landlord_down" 开头、以 ".ckpt" 结尾的文件
#     ckpt_files = [f for f in os.listdir(ckpt_folder)
#                   if f.startswith("landlord_down") and f.endswith(".ckpt")]
    
#     if not ckpt_files:
#         logging.debug("没有找到以 'landlord_down' 开头的 .ckpt 文件")
#         print("没有找到目标 ckpt 文件")
#         return

#     # 按文件名中数字部分排序
#     ckpt_files = sorted(ckpt_files, key=extract_number)
#     logging.debug(f"找到 {len(ckpt_files)} 个 ckpt 文件")
    
#     # 保存测试结果的 CSV 文件
#     results_csv = "evaluate_results.csv"
#     # CSV 头部信息
#     csv_fieldnames = ["ckpt_file", "frames_value", "command", "wp_landlord", "wp_farmers", "adp_landlord", "adp_farmers", "Average consecutive plays per game", "Average counter big cards per game", "Average successful coordination per game", "Average bomb/rocket plays per game"]
#     result_rows = []
    
#     # 循环处理每个 ckpt 文件
#     for ckpt_file in ckpt_files:
#         full_ckpt_path = os.path.join(ckpt_folder, ckpt_file)
#         # 提取 frames_value，假设文件名格式为 landlord_down_weights_数字.ckpt
#         try:
#             frames_value = ckpt_file.split('_')[-1].replace('.ckpt', '')
#         except Exception as e:
#             logging.error(f"无法提取 frames_value from {ckpt_file}: {e}")
#             frames_value = ""
        
#         command = (
#             f"python3 evaluate.py "
#             f"--landlord baselines/sl/landlord.ckpt "
#             f"--landlord_up baselines/sl/landlord_up.ckpt "
#             f"--landlord_down {full_ckpt_path}"
#         )
#         logging.debug(f"开始测试：{full_ckpt_path}")
#         logging.debug(f"运行命令：{command}")
#         print(f"正在测试：{full_ckpt_path}")
        
#         try:
#             # 使用 subprocess.Popen 来运行命令，并确保等待进程完成
#             process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             stdout, stderr = process.communicate()  # 等待命令完成并获取输出
            
#             # 获取返回码
#             returncode = process.returncode
#             stdout = stdout.decode("utf-8").strip()
#             stderr = stderr.decode("utf-8").strip()
            
#             logging.debug(f"完成 {ckpt_file}，返回码：{returncode}")
#             if stdout:
#                 logging.debug(f"stdout: {stdout}")
#             if stderr:
#                 logging.error(f"stderr: {stderr}")
            
#             # 解析标准输出
#             result_data = parse_output(stdout)
#             result_data['ckpt_file'] = ckpt_file
#             result_data['frames_value'] = frames_value
#             result_data['command'] = command
            
#         except Exception as e:
#             logging.error(f"调用 {full_ckpt_path} 时出错: {e}")
#             result_data = {
#                 'ckpt_file': ckpt_file,
#                 'frames_value': frames_value,
#                 'command': command,
#                 'wp_landlord': None,
#                 'wp_farmers': None,
#                 'adp_landlord': None,
#                 'adp_farmers': None,
#                 'Average consecutive plays per game': None,
#                 'Average counter big cards per game': None,
#                 'Average successful coordination per game': None,
#                 'Average bomb/rocket plays per game': None
#             }
        
#         # 保存结果到列表中
#         result_rows.append(result_data)
        
#         # 可选：每次调用后稍作等待，便于观察和资源释放
#         time.sleep(5)
    
#     # 写入 CSV 文件保存测试结果
#     try:
#         with open(results_csv, "w", newline='', encoding="utf-8") as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
#             writer.writeheader()
#             for row in result_rows:
#                 writer.writerow(row)
#         logging.debug(f"测试结果已保存到 {results_csv}")
#         print(f"测试结果保存到 {results_csv}")
#     except Exception as e:
#         logging.error(f"写入 CSV 文件失败: {e}")
#         print(f"写入 CSV 文件失败: {e}")

# if __name__ == "__main__":
#     main()