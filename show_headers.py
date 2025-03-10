"""
Excel文件标题行查看工具
用于读取并显示Excel文件的标题行（第一行）

作者：[Your Name]
日期：[Date]
"""

import pandas as pd
import os

def show_excel_headers(file_path):
    """
    读取并显示Excel文件的标题行（第一行）
    
    参数:
        file_path (str): Excel文件路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return
        
        # 读取Excel文件
        print(f"正在读取文件: {file_path}")
        df = pd.read_excel(file_path)
        
        # 获取并显示标题行
        headers = df.columns.tolist()
        
        print("\n文件标题行（第一行）:")
        print("-" * 100)
        for i, header in enumerate(headers):
            print(f"{i+1}. {header}")
        print("-" * 100)
        
        # 显示数据形状
        print(f"\n数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
        
        # 显示前几行数据类型
        print("\n数据类型预览:")
        print(df.dtypes)
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    # 设置文件路径
    file_path = "C:\\Users\\jiawang\\Desktop\\狗牙儿.xlsx"
    
    # 显示标题行
    show_excel_headers(file_path) 