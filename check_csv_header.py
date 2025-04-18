import csv

def check_csv_headers(file_path):
    """读取并打印 CSV 文件的列名（头部）"""
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # 读取并打印文件的第一行内容（列名）
            header = next(reader)
            print("CSV 文件的列名：", header)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    # 指定 evaluate_results.csv 文件的路径
    file_path = 'evaluate_results.csv'
    check_csv_headers(file_path)