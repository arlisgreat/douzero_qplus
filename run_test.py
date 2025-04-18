#!/usr/bin/env python3
import os
import csv
import subprocess
import logging
import time
import re
import argparse
from typing import Dict

def setup_logging() -> None:
    """配置 debug.log 日志输出"""
    logging.basicConfig(
        filename="debug02.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    logging.debug("日志初始化成功")

def extract_number(filename: str) -> int:
    """从文件名中提取 frames 数字，失败返回 0"""
    try:
        return int(filename.split('_')[-1].replace('.ckpt', ''))
    except Exception as e:
        logging.error(f"提取数字失败 {filename}: {e}")
        return 0

def parse_output(output: str) -> Dict[str, str]:
    """解析 evaluate.py 标准输出，返回结果字典"""
    # WP
    wp_match = re.search(
        r'WP results:\s*landlord\s*:\s*Farmers\s*-\s*([0-9.\-]+)\s*:\s*([0-9.\-]+)', 
        output)
    # ADP
    adp_match = re.search(
        r'ADP results:\s*landlord\s*:\s*Farmers\s*-\s*([0-9.\-]+)\s*:\s*([0-9.\-]+)', 
        output)
    # 农民协作统计
    coop_stats = {}
    for stat, value in re.findall(r'Average (.*?):\s*([0-9.\-]+)', output):
        coop_stats[f"Average {stat}"] = value

    return {
        'wp_landlord': wp_match.group(1) if wp_match else None,
        'wp_farmers':  wp_match.group(2) if wp_match else None,
        'adp_landlord': adp_match.group(1) if adp_match else None,
        'adp_farmers':  adp_match.group(2) if adp_match else None,
        **coop_stats
    }

def main():
    # ---------- 参数 ----------
    parser = argparse.ArgumentParser(
        description="批量测试 DouZero landlord_down 权重并写入 CSV")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="仅测试 frames 大于此值的 ckpt（默认 0 = 全部）")
    args = parser.parse_args()
    start_frame = args.start_frame

    setup_logging()
    ckpt_folder = "./douzero_checkpoints/douzero_2/"

    # ---------- 收集 ckpt ----------
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_folder)
         if f.startswith("landlord_down") and f.endswith(".ckpt")],
        key=extract_number
    )
    if not ckpt_files:
        logging.warning("未找到 ckpt 文件")
        print("没有找到目标 ckpt 文件")
        return
    logging.debug(f"找到 {len(ckpt_files)} 个 ckpt 文件")

    # ---------- CSV ----------
    results_csv = "evaluate_results_2.csv"
    csv_fieldnames = [
        "ckpt_file", "frames_value", "command",
        "wp_landlord", "wp_farmers", "adp_landlord", "adp_farmers",
        "Average consecutive plays per game",
        "Average counter big cards per game",
        "Average successful coordination per game",
        "Average bomb/rocket plays per game"
    ]
    file_exists = os.path.exists(results_csv)
    write_header = not file_exists

    # ---------- 主循环 ----------
    for ckpt_file in ckpt_files:
        frames_val = extract_number(ckpt_file)
        # 只测试大于 start_frame 的文件
        if frames_val <= start_frame:
            continue

        full_ckpt_path = os.path.join(ckpt_folder, ckpt_file)
        command = (
            f"python3 evaluate.py "
            f"--landlord baselines/sl/landlord.ckpt "
            f"--landlord_up baselines/sl/landlord_up.ckpt "
            f"--landlord_down {full_ckpt_path}"
        )

        logging.debug(f"开始测试：{full_ckpt_path}")
        logging.debug(f"运行命令：{command}")
        print(f"正在测试：{full_ckpt_path}")

        # ---------- 调用 evaluate.py ----------
        try:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            returncode = proc.returncode

            stdout = stdout.decode("utf-8", errors="ignore").strip()
            stderr = stderr.decode("utf-8", errors="ignore").strip()

            logging.debug(f"完成 {ckpt_file}，返回码：{returncode}")
            if stdout:
                logging.debug(f"stdout: {stdout}")
            if stderr:
                logging.error(f"stderr: {stderr}")

            result_data = parse_output(stdout)
        except Exception as e:
            logging.error(f"调用 {ckpt_file} 出错: {e}")
            result_data = {}

        # ---------- 写 CSV ----------
        result_row = {
            'ckpt_file': ckpt_file,
            'frames_value': str(frames_val),
            'command': command,
            **{field: result_data.get(field) for field in csv_fieldnames
               if field not in ("ckpt_file", "frames_value", "command")}
        }

        try:
            with open(results_csv, "a", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False          # 防止重复写表头
                writer.writerow(result_row)

            logging.debug(f"已写入 {results_csv}")
            print(f"当前测试结果保存到 {results_csv}")
        except Exception as e:
            logging.error(f"写入 CSV 失败: {e}")
            print(f"写入 CSV 文件失败: {e}")

        # 可选：短暂休息，避免过度占用资源
        time.sleep(5)

if __name__ == "__main__":
    main()