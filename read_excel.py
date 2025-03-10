"""
Excel文件标题行读取工具
用于读取Excel文件的标题行（列名）并打印出来

作者：[Your Name]
日期：[Date]
"""

import os
import pandas as pd

def read_excel_headers(file_path):
    """
    读取Excel文件的标题行（列名）
    
    功能：
        - 检查文件是否存在
        - 读取Excel文件
        - 提取标题行（列名）
        - 打印标题行内容
    
    参数:
        file_path (str): Excel文件的路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return
        
        # 读取Excel文件
        print(f"正在读取文件: {file_path}")
        df = pd.read_excel(file_path)
        
        # 获取标题行（列名）
        headers = df.columns.tolist()
        
        # 打印标题行内容
        print("\nExcel文件标题行:")
        print("-" * 50)
        for i, header in enumerate(headers):
            print(f"列 {i+1}: {header}")
        print("-" * 50)
        
        # 打印为一行
        print("\n标题行内容(单行):")
        print(" ".join(str(header) for header in headers))
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    # 获取桌面路径
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    # 构建Excel文件路径 - 请替换为实际文件名
    excel_file = os.path.join(desktop_path, "狗牙儿.xlsx")
    
    # 读取并打印标题行
    read_excel_headers(excel_file) 