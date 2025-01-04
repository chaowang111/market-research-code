"""
文件管理模拟系统
实现了一个基于命令行的文件系统，支持文件和目录的基本操作。
采用混合索引方式（直接索引+间接索引）管理文件块，使用位示图法管理磁盘空间。

主要功能：
1. 目录操作：创建、删除、切换、移动
2. 文件操作：创建、删除、复制、移动、重命名
3. 文件内容：读取、写入
4. 系统信息：磁盘使用情况、文件属性

所有操作均支持绝对路径和相对路径。


"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import math

# 系统常量定义
BLOCK_SIZE = 1024  # 每个盘块大小为1KB
TOTAL_BLOCKS = 1024  # 总共1024个盘块
MAX_DIRECT_BLOCKS = 10  # 直接索引块数量
MAX_INDIRECT_BLOCKS = 100  # 间接索引块可以指向的块数量

class Colors:
    """
    终端输出颜色定义类
    用于美化命令行界面，提供不同类型信息的颜色区分
    
    属性：
        HEADER: 紫色，用于标题
        BLUE: 蓝色，用于提示信息
        GREEN: 绿色，用于成功信息
        YELLOW: 黄色，用于警告信息
        RED: 红色，用于错误信息
        ENDC: 结束颜色标记
        BOLD: 粗体文本
    """
    HEADER = '\033[95m'  # 紫色，用于标题
    BLUE = '\033[94m'    # 蓝色，用于提示信息
    GREEN = '\033[92m'   # 绿色，用于成功信息
    YELLOW = '\033[93m'  # 黄色，用于警告信息
    RED = '\033[91m'     # 红色，用于错误信息
    ENDC = '\033[0m'     # 结束颜色
    BOLD = '\033[1m'     # 粗体

def print_color(text: str, color: str):
    """
    打印彩色文本的辅助函数
    
    参数:
        text (str): 要打印的文本内容
        color (str): 文本颜色（使用Colors类中定义的颜色代码）
    
    功能:
        将文本用指定的颜色打印到终端，并在结尾自动重置颜色
    """
    print(f"{color}{text}{Colors.ENDC}")

class Block:
    """
    磁盘块类，表示文件系统中的一个物理块
    
    属性:
        id (int): 块号，用于唯一标识该块
        is_used (bool): 标记该块是否被使用
        next_block (Optional[int]): 下一个块的编号（用于链接组织）
        data (Optional[bytes]): 块中存储的实际数据
        
    说明:
        - 每个块都有固定大小(BLOCK_SIZE)
        - 使用next_block实现块的链接，支持文件��储
        - 通过is_used标记实现空闲块的管理
    """
    def __init__(self, block_id: int):
        """
        初始化一个新的磁盘块
        
        参数:
            block_id (int): 块的唯一标识号
        """
        self.id = block_id
        self.is_used = False
        self.next_block = None
        self.data = None

class BlockManager:
    """
    磁盘块管理器，负责管理所有磁盘块的分配和释放
    使用位示图法管理空闲块，提供块的分配和回收功能
    
    属性:
        blocks (List[Block]): 所有磁盘块的列表
        free_blocks (List[int]): 空闲块号列表
        
    说明:
        - 使用位示图法管理空闲空间
        - 维护空闲块列表，支持快速分配和回收
        - 保证空间使用的连续性和效率
    """
    def __init__(self):
        """
        初始化块管理器
        创建指定数量的块并将其标记为未使用
        """
        self.blocks = [Block(i) for i in range(TOTAL_BLOCKS)]
        self.free_blocks = list(range(TOTAL_BLOCKS))
    
    def allocate_blocks(self, num_blocks: int) -> List[int]:
        """
        分配指定数量的块
        
        参数:
            num_blocks (int): 需要分配的块数量
            
        返回:
            List[int]: 分配的块号列表
            
        异常:
            Exception: 当没有足够的空闲块时抛出异常
            
        说明:
            - 检查是否有足够的空闲块
            - 从空闲列表中分配指定数量的块
            - 更新块的使用状态
        """
        if len(self.free_blocks) < num_blocks:
            raise Exception("没有足够的空闲块")
        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop(0)
            self.blocks[block_id].is_used = True
            allocated.append(block_id)
        return allocated

    def free_block(self, block_id: int):
        """
        释放指定的块
        
        参数:
            block_id (int): 要释放的块号
            
        说明:
            - 将块标记为未使用
            - 清除块的数据和链接信息
            - 将块号添加回空闲列表
        """
        if 0 <= block_id < TOTAL_BLOCKS:
            self.blocks[block_id].is_used = False
            self.blocks[block_id].data = None
            self.blocks[block_id].next_block = None
            self.free_blocks.append(block_id)

class FileSystemEntry:
    """
    文件系统条目，是目录和文件的父类
    提供了文件系统中所有条目共有的基本属性和方法
    
    属性:
        name (str): 条目名称
        creation_time (datetime): 创建时间
        parent (Optional[Directory]): 父目录的引用
        
    说明:
        - 作为文件和目录的基类，提供共同的属性和方法
        - 支持获取完整路径等基本操作
        - 维护层级结构关系
    """
    def __init__(self, name: str, parent=None):
        """
        初始化文件系统条目
        
        参数:
            name (str): 条目名称
            parent (Optional[Directory]): 父目录，默认为None
        """
        self.name = name
        self.creation_time = datetime.now()
        self.parent = parent
        
    def get_full_path(self) -> str:
        """
        获取条目的完整路径
        
        返回:
            str: 从根目录到当前条目的完整路径
            
        说明:
            - 递归向上查找父目录
            - 构建完整的路径字符串
            - 处理根目录的特殊情况
        """
        if self.parent is None:
            return self.name
        parent_path = self.parent.get_full_path()
        return os.path.join(parent_path, self.name)

class File(FileSystemEntry):
    """
    文件���，继承自FileSystemEntry
    实现了混合索引方式的文件组织
    
    属性:
        size (int): 文件大小（字节）
        direct_blocks (List[int]): 直接索引块列表，存储数据块的块号
        index_block (Optional[int]): 间接索引块的块号，存储指向数据块的指针
        indirect_data_blocks (List[int]): 间接索引指向的数据块列表
        content (str): 文件内容
        
    说明:
        - 使用混合索引方式管理文件块
        - 直接索引：直接存储数据块的块号（0-9）
        - 间接索引：使用一个索引块存储指针，指针指向数据块
        - 维护文件大小和内容信息
    """
    def __init__(self, name: str, size: int, parent=None):
        """
        初始化文件对象
        
        参数:
            name (str): 文件名
            size (int): 文件初始大小
            parent (Optional[Directory]): 父目录
        """
        super().__init__(name, parent)
        self.size = size
        self.direct_blocks = []  # 直接索引块，存储数据块号
        self.index_block = None  # 间接索引块号
        self.indirect_data_blocks = []  # 间接索引指向的数据块号列表
        self.content = ""  # 文件内容

    def show_properties(self, path: str):
        """
        显示文件或目录的详细属性信息
        
        参数:
            path (str): 文件或目录的路径
            
        说明:
            - 显示名称、类型、创建时间等基本信息
            - 对于文件显示：
              - 文件大小
              - 直接索引块（0-9）的使用情况
              - 间接索引块的结构（索引块及其指向的数据块）
            - 对于目录显示包含的项目数
        """
        parent, name = self.find_entry_by_path(path)
        if name not in parent.entries:
            raise Exception(f"文件或目录 {name} 不存在")
        
        entry = parent.entries[name]
        print_color("\n属性信息:", Colors.HEADER)
        print_color("-" * 60, Colors.HEADER)
        print(f"名称: {entry.name}")
        print(f"类型: {'目录' if isinstance(entry, Directory) else '文件'}")
        print(f"创建时间: {entry.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if isinstance(entry, File):
            print(f"大小: {entry.size} 字节")
            
            # 计算实际使用的数据块总数
            total_blocks = len(entry.direct_blocks) + len(entry.indirect_data_blocks)
            if entry.index_block is not None:
                total_blocks += 1  # 加上间接索引块
            print(f"占用总块数: {total_blocks}")
            
            # 显示直接索引块信息
            print("\n直接索引块 (0-9):")
            if entry.direct_blocks:
                for i in range(MAX_DIRECT_BLOCKS):
                    if i < len(entry.direct_blocks):
                        print(f"  索引{i}: 指向数据块 {entry.direct_blocks[i]}")
                    else:
                        print(f"  索引{i}: 未使用")
            else:
                print("  无直接索引块使用")
            
            # 显示间接索引块信息
            if entry.index_block is not None:
                print("\n间接索引结构:")
                print(f"  间接索引块: {entry.index_block}")
                print(f"  └─ 可存储指针数: {BLOCK_SIZE // 4}")  # 假设每个指针占4字节
                if entry.indirect_data_blocks:
                    print("  └─ 指向的数据块:")
                    for i, block in enumerate(entry.indirect_data_blocks):
                        print(f"     └─ 数据块{i}: {block}")
                else:
                    print("  └─ 暂无指向的数据块")
            else:
                print("\n间接索引块: 未使用")
        else:
            print(f"包含���目数: {len(entry.entries)}")
        print_color("-" * 60, Colors.HEADER)

    def create_file(self, path: str, size: int):
        """
        创建新文件，支持绝对路径和相对路径
        
        参数:
            path (str): 文件路径
            size (int): 文件大小（字节）
            
        说明:
            - 分配块时遵循以下顺序：
              1. 分配直接索引块号（0-9）
              2. 分配间接索引块号（紧接着直接索引块号）
              3. 分配数据块号（最后分配）
        """
        parent, file_name = self.find_entry_by_path(path)
        if file_name in parent.entries:
            raise Exception(f"文件 {file_name} 已存在")
        
        # 计算需要的块数
        needed_blocks = math.ceil(size / BLOCK_SIZE)
        if needed_blocks > MAX_DIRECT_BLOCKS + MAX_INDIRECT_BLOCKS:
            raise Exception("文件太大")
            
        # 创建新文件
        new_file = File(file_name, size, parent)
        
        # 分配块
        if needed_blocks <= MAX_DIRECT_BLOCKS:
            # 只需要直接索引块
            new_file.direct_blocks = self.block_manager.allocate_blocks(needed_blocks)
        else:
            # 需要直接索引块和间接索引块
            total_blocks_needed = needed_blocks + 1  # +1 是因为需要一个间接索引块
            all_blocks = self.block_manager.allocate_blocks(total_blocks_needed)
            
            # 1. 分配直接索引块（前10个块）
            new_file.direct_blocks = all_blocks[:MAX_DIRECT_BLOCKS]
            
            # 2. 分配间接索引块（第11个块）
            new_file.index_block = all_blocks[MAX_DIRECT_BLOCKS]
            
            # 3. 分配数据块（剩余的块）
            new_file.indirect_data_blocks = all_blocks[MAX_DIRECT_BLOCKS + 1:]
            
        parent.entries[file_name] = new_file

class Directory(FileSystemEntry):
    """
    目录类，继承自FileSystemEntry
    实现了树形目录结构
    
    属性:
        entries (Dict[str, FileSystemEntry]): 存储目录下的所有条目
        
    说明:
        - 使用字典存储子条目，支持快速查找
        - 维护目录树结构
        - 支持文件和子目录的管
    """
    def __init__(self, name: str, parent=None):
        """
        初始化目录对象
        
        参数:
            name (str): 目录名
            parent (Optional[Directory]): 父目录
        """
        super().__init__(name, parent)
        self.entries: Dict[str, FileSystemEntry] = {}

class FileSystem:
    """
    文件系统类，实现了所有文件系统操作
    包括文件和目录的创建、删除、移动、复制等功能
    
    属性:
        block_manager (BlockManager): 块管理器实例
        root (Directory): 根目录
        current_directory (Directory): 当前工作目录
        
    说明:
        - 供完整的文件系统操作接口
        - 管理系统资源和状态
        - 处理用户命令和错误
    """
    def __init__(self):
        """
        初始化文件系统
        创建根目录和块管理器
        """
        self.block_manager = BlockManager()
        self.root = Directory("/")
        self.current_directory = self.root
        print_color("欢迎使用文件管理系统！", Colors.GREEN)
        print_color("输入 'help' 获取帮助信息", Colors.BLUE)

    def get_current_path(self) -> str:
        """
        获取当前工作目录的完整路径
        
        返回:
            str: 当前目录的完整路径
        """
        return self.current_directory.get_full_path()

    def show_properties(self, path: str):
        """
        显示文件或目录的详细属性信息
        
        参数:
            path (str): 文件或目录的路径
            
        说明:
            - 显示名称、类型、��建时间等基本信息
            - 对于文件显示：
              - 文件大小
              - 直接索引块（0-9）的使用情况
              - 间接索引块的结构（索引块及其指向的数据块）
            - 对于目录显示包含的项目数
        """
        parent, name = self.find_entry_by_path(path)
        if name not in parent.entries:
            raise Exception(f"文件或目录 {name} 不存在")
        
        entry = parent.entries[name]
        print_color("\n属性信息:", Colors.HEADER)
        print_color("-" * 60, Colors.HEADER)
        print(f"名称: {entry.name}")
        print(f"类型: {'目录' if isinstance(entry, Directory) else '文件'}")
        print(f"创建时间: {entry.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if isinstance(entry, File):
            print(f"大小: {entry.size} 字节")
            
            # 计算实际使用的数据块总数
            data_blocks = len(entry.direct_blocks)
            if entry.index_block is not None:
                data_blocks += 1  # 加上间接索引块
            print(f"占用总块数: {data_blocks}")
            
            # 显示直接索引块信息
            print("\n直接索引块 (0-9):")
            if entry.direct_blocks:
                for i in range(MAX_DIRECT_BLOCKS):
                    if i < len(entry.direct_blocks):
                        print(f"  索引{i}: 指向数据块 {entry.direct_blocks[i]}")
                    else:
                        print(f"  索引{i}: 未使用")
            else:
                print("  无直接索引块使用")
            
            # 显示间接索引块信息
            if entry.index_block is not None:
                print("\n间接索引结构:")
                # 假设第一个间接索引块用于存储指针
                index_block = entry.index_block
                data_blocks = entry.indirect_data_blocks if len(entry.indirect_data_blocks) > 0 else []
                
                print(f"  间接索引块: {index_block}")
                print(f"  └─ 可存储指针数: {BLOCK_SIZE // 4}")  # 假设每个指针占4字节
                if data_blocks:
                    print("  └─ 指向的数据块:")
                    for i, block in enumerate(data_blocks):
                        print(f"     └─ 数据块{i}: {block}")
                else:
                    print("  └─ 暂无指向的数据块")
            else:
                print("\n间接索引块: 未使用")
        else:
            print(f"包含项目数: {len(entry.entries)}")
        print_color("-" * 60, Colors.HEADER)

    def show_help(self):
        """
        显示帮助信息
        列出所有可用的命令及其用法
        """
        help_text = """
可用命令：
    md <目录名>     - 创建新目录
    rd <目录名>     - 删除空目录
    cd <目录名>     - 切换当前目录 (使用 .. 返回上级目录, / 返回根目录)
    mv <源> <目标>  - 移动文件或目录
    cp <源> <目标>  - 复制文件
    dir            - 显示当前目录内容
    create <文件名> <大小> - 创建指定大小的文件（单位：字节）
    del <文件名>    - 删除文件
    rename <旧名称> <新名称> - 重命名文件或目录
    write <文件名> <内容> - 写入文件内容
    read <文件名>   - 读取文件内容
    disk           - 显示磁盘使用情况
    prop <名称>    - 显示文件或目录的详细属性
    help           - 显示此帮助信息
    exit           - 退出系统

注意事项：
    - 文件大小不能超过 {0} KB
    - 目录必须为空才能删除
    - 路径区分大小写
    - 支持相对路径和绝对路径
""".format(BLOCK_SIZE * (MAX_DIRECT_BLOCKS + MAX_INDIRECT_BLOCKS) / 1024)
        print_color(help_text, Colors.BLUE)

    def dir(self):
        """
        列出目录内容
        
        功能：
            - 显示当前目录的所有内容
            - 区分文件和目录
            - 显示条目的基本属性（类型、名称、大小、创建时间）
            - 格式化输出信息
        
        返回:
            List[str]: 格式化的目录内容列表
        """
        entries = []
        entries.append(f"当前目录: {self.get_current_path()}")
        entries.append("-" * 60)
        if not self.current_directory.entries:
            entries.append("(空目录)")
        else:
            entries.append(f"{'类型':<6} {'名称':<20} {'大小(字节)':<12} {'创建时间'}")
            entries.append("-" * 60)
            for name, entry in self.current_directory.entries.items():
                entry_type = "DIR" if isinstance(entry, Directory) else "FILE"
                size = getattr(entry, 'size', '-')
                creation_time = entry.creation_time.strftime("%Y-%m-%d %H:%M:%S")
                entries.append(f"{entry_type:<6} {name:<20} {size:<12} {creation_time}")
        return entries

    def md(self, path: str):
        """
        创建新目录
        
        功能：
            - 在指定路径创建新的空目录
            - 支持绝对路径和相对路径
            - 检查目录名是否已存在
            - 建立目录的父子关系
        
        参数:
            path (str): 要创建的目录的路径
        """
        parent, dir_name = self.find_entry_by_path(path)
        if dir_name in parent.entries:
            raise Exception(f"目录 {dir_name} 已存在")
        new_dir = Directory(dir_name, parent)
        parent.entries[dir_name] = new_dir

    def rd(self, path: str):
        """
        删除目录
        
        功能：
            - 删除指定路径的空目录
            - 支持绝对路径和相对路径
            - 检查目录是否为空
            - 检查目标是否为目录
            - 从父目录中移除目录条目
        
        参数:
            path (str): 要删除的目录的路径
        """
        parent, dir_name = self.find_entry_by_path(path)
        if dir_name not in parent.entries:
            raise Exception(f"目录 {dir_name} 不存在")
        dir_to_remove = parent.entries[dir_name]
        if not isinstance(dir_to_remove, Directory):
            raise Exception(f"{dir_name} 不是目录")
        if dir_to_remove.entries:
            raise Exception("目录不为空")
        del parent.entries[dir_name]

    def cd(self, path: str):
        """
        切换当前工作目录
        
        功能：
            - 改变当前工作目录到指定路径
            - 支持特殊���径：'/'（根目录）、'..'（上级目录）、'.'（当前目录）
            - 支持绝对路径和相对路径
            - 验证目标路径的有效性
            - 更新当前工作目录引用
        
        参数:
            path (str): 目标目录路径
        """
        if path == "/":
            self.current_directory = self.root
            return
            
        if path == "..":
            if self.current_directory.parent is not None:
                self.current_directory = self.current_directory.parent
            return
            
        parent, dir_name = self.find_entry_by_path(path)
        if dir_name:
            if dir_name not in parent.entries:
                raise Exception(f"目录 {path} 不存在")
            target = parent.entries[dir_name]
            if not isinstance(target, Directory):
                raise Exception(f"{path} 不是目录")
            self.current_directory = target
        else:
            self.current_directory = parent

    def create_file(self, path: str, size: int):
        """
        创建新文件
        
        功能：
            - 在指定路径创建指定大小的新文件
            - 分配所需的磁盘块
            - 支持混合索引（直接索引+间接索引）
            - 按顺序分配���号：
              1. 直接索引块（0-9）
              2. 间接索引块
              3. 数据块
            - 建立文件的索引结构
        
        参数:
            path (str): 文件路径
            size (int): 文件大小（字节）
        """
        parent, file_name = self.find_entry_by_path(path)
        if file_name in parent.entries:
            raise Exception(f"文件 {file_name} 已存在")
        
        # 计算需要的块数
        needed_blocks = math.ceil(size / BLOCK_SIZE)
        if needed_blocks > MAX_DIRECT_BLOCKS + MAX_INDIRECT_BLOCKS:
            raise Exception("文件太大")
            
        # 创建新文件
        new_file = File(file_name, size, parent)
        
        # 分配块
        if needed_blocks <= MAX_DIRECT_BLOCKS:
            # 只需要直接索引块
            new_file.direct_blocks = self.block_manager.allocate_blocks(needed_blocks)
        else:
            # 需要直接索引块和间接索引块
            total_blocks_needed = needed_blocks + 1  # +1 是因为需要一个间接索引块
            all_blocks = self.block_manager.allocate_blocks(total_blocks_needed)
            
            # 1. 分配直接索引块（前10个块）
            new_file.direct_blocks = all_blocks[:MAX_DIRECT_BLOCKS]
            
            # 2. 分配间接索引块（第11个块）
            new_file.index_block = all_blocks[MAX_DIRECT_BLOCKS]
            
            # 3. 分配数据块（剩余的块）
            new_file.indirect_data_blocks = all_blocks[MAX_DIRECT_BLOCKS + 1:]
            
        parent.entries[file_name] = new_file

    def delete_file(self, path: str):
        """
        删除文件
        
        功能：
            - 删除指定路径的文件
            - 释放文件占用的所有磁盘块
            - 释放顺序：
              1. 直接索引块
              2. 间接索引块
              3. 间接索引指向的数据块
            - 从父目录中移除文件条目
        
        参数:
            path (str): 要删除的文件路径
        """
        parent, file_name = self.find_entry_by_path(path)
        if file_name not in parent.entries:
            raise Exception(f"文件 {file_name} 不存在")
        file_to_delete = parent.entries[file_name]
        if not isinstance(file_to_delete, File):
            raise Exception(f"{file_name} 不是文件")
            
        # 释放直接索引块
        for block in file_to_delete.direct_blocks:
            self.block_manager.free_block(block)
            
        # 释放间接索引块和它指向的数据块
        if file_to_delete.index_block is not None:
            # 释放间接索引块
            self.block_manager.free_block(file_to_delete.index_block)
            # 释放间接索引指向的数据块
            for block in file_to_delete.indirect_data_blocks:
                self.block_manager.free_block(block)
            
        # 从父目录中删除文件条目
        del parent.entries[file_name]

    def show_disk_usage(self):
        """
        显示磁盘使用情况
        
        功能：
            - 统计并显示磁盘使用情况
            - 区分显示不同类型的块：
              1. 直接索引块
              2. 间接索引块
              3. 数据块
            - 显示空闲块信息
            - 计算并显示使用率
        
        返回:
            str: 格式化的磁盘使用情况信息
        """
        used_blocks = TOTAL_BLOCKS - len(self.block_manager.free_blocks)
        usage_percent = (used_blocks / TOTAL_BLOCKS) * 100
        
        # 收集不同类型的块号
        direct_blocks = set()  # 直接索引块
        index_blocks = set()   # 间接索引块
        data_blocks = set()    # 数据块
        
        # 遍历所有文件，收集块号信息
        def collect_blocks(directory):
            for entry in directory.entries.values():
                if isinstance(entry, File):
                    # 收集直接索引块
                    direct_blocks.update(entry.direct_blocks)
                    # 收集间接索引块
                    if entry.index_block is not None:
                        index_blocks.add(entry.index_block)
                        # 收集间接索引指向的数据块
                        data_blocks.update(entry.indirect_data_blocks)
                elif isinstance(entry, Directory):
                    collect_blocks(entry)
        
        # 从根目录开始收集
        collect_blocks(self.root)
        
        # 获取空闲块号
        free_block_numbers = sorted(self.block_manager.free_blocks)
        
        # 格式化输出信息
        info = [
            f"总块数: {TOTAL_BLOCKS}",
            f"已使用: {used_blocks} 块",
            "\n块使用情况:",
            f"  直接索引块: {', '.join(map(str, sorted(direct_blocks))) if direct_blocks else '无'}",
            f"  间接索引块: {', '.join(map(str, sorted(index_blocks))) if index_blocks else '无'}",
            f"  数据块: {', '.join(map(str, sorted(data_blocks))) if data_blocks else '无'}",
            f"\n空闲: {len(self.block_manager.free_blocks)} 块",
            f"空闲块号: {', '.join(map(str, free_block_numbers)) if free_block_numbers else '无'}",
            f"\n使用率: {usage_percent:.2f}%"
        ]
        
        return '\n'.join(info)

    def find_entry_by_path(self, path: str) -> tuple[FileSystemEntry, str]:
        """
        路径解析
        
        功能：
            - 解析路径字符串
            - 处理绝对路径和相对路径
            - 支持特殊路径符号（. 和 ..）
            - 验证路径的有效性
            - 返回父目录和目标名称
        
        参数:
            path (str): 要解析的路径
            
        返回:
            tuple[FileSystemEntry, str]: (父目录对象, 目标名称)
        """
        # 标准化路径分隔符
        path = path.replace('\\', '/')
        
        # 处理绝对路径
        if path.startswith('/'):
            current = self.root
            # 去掉开头的 '/'，如果路径只有 '/'，则 path 变为空字符串
            path = path[1:]
        else:
            current = self.current_directory
            
        # 如果路径为空，返回当前目录
        if not path:
            return current, ''
            
        # 分割路径
        parts = [p for p in path.split('/') if p]
        
        # 如果没有路径部分返回当前目录
        if not parts:
            return current, ''
            
        # 处理除最后一个部分外的所有部分
        parent = current
        for i, part in enumerate(parts[:-1]):
            if part == '..':
                if parent.parent:
                    parent = parent.parent
            elif part == '.':
                continue
            else:
                if part not in parent.entries:
                    raise Exception(f"路径 {'/'.join(parts[:i+1])} 不存在")
                next_entry = parent.entries[part]
                if not isinstance(next_entry, Directory):
                    raise Exception(f"{part} 不是目录")
                parent = next_entry
                    
        return parent, parts[-1]

    def is_subdirectory(self, parent: Directory, child: Directory) -> bool:
        """
        子目录检查
        
        功能：
            - 检查一个目录是否是另一个目录的子目录
            - 递归检查父子关系
            - 防止循环引用
            - 用于移动操作的验证
        
        参数:
            parent: 可能的父目录
            child: 可能的子目录
            
        返回:
            bool: 如果child是parent的子目录则返回True
        """
        current = child
        while current.parent is not None:
            if current.parent == parent:
                return True
            current = current.parent
        return False

    def mv(self, source: str, target: str):
        """
        移动文件或目录
        
        功能：
            - 将文件或目录移动到目标目录中
            - 支持绝对路径和相对路径
            - 更新父子关系
            - 保持原有名称
            - 验证源和目标的有效性
            - 确保目标目录不存在同名条目
        
        参数:
            source (str): 源文件或目录的路径
            target (str): 目标路径
        """
        # 解析源路径
        source_parent, source_name = self.find_entry_by_path(source)
        if source_name not in source_parent.entries:
            raise Exception(f"源文件/目录 {source} 不存在")
        entry = source_parent.entries[source_name]
        
        # 解析目标路径
        target_parent, target_name = self.find_entry_by_path(target)
        
        # 如果目标路径不包含文件名，使用原文件名
        if not target_name:
            target_dir = target_parent
            target_name = entry.name
        else:
            # 检查目标父目录是否存在且是目录
            if not isinstance(target_parent, Directory):
                raise Exception(f"目标路径 {target} 的父目录不存在或不是目录")
            target_dir = target_parent
        
        # 检查目标目录中是否已存在同名文件/目录
        if target_name in target_dir.entries:
            raise Exception(f"目标位置已存在同名文件/目录: {target_name}")
        
        # 如果是目录，检查是否试图将目录移动到其子目录中
        if isinstance(entry, Directory) and isinstance(target_dir, Directory):
            if self.is_subdirectory(entry, target_dir):
                raise Exception("不能将目录移动到其子目录中")
        
        # 执行移动操作
        del source_parent.entries[source_name]
        entry.parent = target_dir
        entry.name = target_name
        target_dir.entries[target_name] = entry
        
        print_color(f"已将 {source} 移动到 {target}", Colors.GREEN)

    def cp(self, source: str, target: str):
        """
        复制文件
        
        功能：
            - 创建文件的完整副本
            - 分配新的磁盘块
            - 复制文件内容和属性
            - 维持索引结构
            - 按顺序分配新块号
        
        参数:
            source (str): 源文件的路径
            target (str): 目标位置的路径
        """
        source_parent, source_name = self.find_entry_by_path(source)
        if source_name not in source_parent.entries:
            raise Exception(f"源文件 {source} 不存在")
            
        source_file = source_parent.entries[source_name]
        if not isinstance(source_file, File):
            raise Exception(f"{source} 不是文件")
            
        target_parent, target_name = self.find_entry_by_path(target)
        if not isinstance(target_parent, Directory):
            raise Exception(f"目标路径 {target} 不是目录")
            
        if target_name in target_parent.entries:
            raise Exception(f"目标文件 {target} 已存在")
            
        # 计算需要的总块数
        total_blocks = len(source_file.direct_blocks)
        if source_file.index_block is not None:
            total_blocks += 1 + len(source_file.indirect_data_blocks)  # 加1是因为需要一个间接索引块
            
        # 分配所有需要的块
        all_blocks = self.block_manager.allocate_blocks(total_blocks)
        
        # 创建新文件
        new_file = File(target_name, source_file.size, target_parent)
        
        if source_file.index_block is not None:
            # 间接索引的情况
            new_file.direct_blocks = all_blocks[:MAX_DIRECT_BLOCKS]
            new_file.index_block = all_blocks[MAX_DIRECT_BLOCKS]
            new_file.indirect_data_blocks = all_blocks[MAX_DIRECT_BLOCKS + 1:]
        else:
            # 只有直接索引的情况
            new_file.direct_blocks = all_blocks
            
        new_file.content = source_file.content
        target_parent.entries[target_name] = new_file
        print_color(f"已将 {source} 复制到 {target}", Colors.GREEN)

    def rename(self, old_path: str, new_name: str):
        """
        重命名文件或目录
        
        功能：
            - 修改文件或目录的名称
            - 保持其他属性不变
            - 验证新名称的可用性
            - 更新父目录的条目
        
        参数:
            old_path (str): 要重命名的文件或目录的路径
            new_name (str): 新名称
        """
        parent, old_name = self.find_entry_by_path(old_path)
        if old_name not in parent.entries:
            raise Exception(f"文件/目录 {old_name} 不存在")
        if new_name in parent.entries:
            raise Exception(f"文件/目录 {new_name} 已存在")
            
        entry = parent.entries[old_name]
        entry.name = new_name
        parent.entries[new_name] = entry
        del parent.entries[old_name]
        print_color(f"已将 {old_name} 重命名为 {new_name}", Colors.GREEN)

    def write(self, path: str, content: str):
        """
        写入文件内容
        
        功能：
            - 向指定文件写入内容
            - 自动调整文件大小
            - 必要时分配新的磁盘块
            - 更新文件的索引结构
            - 维护文件大小信息
        
        参数:
            path (str): 文件路径
            content (str): 要写入的内容
        """
        parent, file_name = self.find_entry_by_path(path)
        if file_name not in parent.entries:
            raise Exception(f"文件 {file_name} 不存在")
            
        file = parent.entries[file_name]
        if not isinstance(file, File):
            raise Exception(f"{file_name} 不是文件")
            
        file.content = content
        new_size = len(content.encode('utf-8'))
        needed_blocks = math.ceil(new_size / BLOCK_SIZE)
        
        if needed_blocks > len(file.direct_blocks) + len(file.indirect_data_blocks):
            additional_blocks = needed_blocks - (len(file.direct_blocks) + len(file.indirect_data_blocks))
            new_blocks = self.block_manager.allocate_blocks(additional_blocks)
            
            if needed_blocks <= MAX_DIRECT_BLOCKS:
                file.direct_blocks.extend(new_blocks)
            else:
                if len(file.direct_blocks) < MAX_DIRECT_BLOCKS:
                    remaining = MAX_DIRECT_BLOCKS - len(file.direct_blocks)
                    file.direct_blocks.extend(new_blocks[:remaining])
                    file.indirect_data_blocks.extend(new_blocks[remaining:])
                else:
                    file.indirect_data_blocks.extend(new_blocks)
                    
        file.size = new_size
        print_color(f"文件 {file_name} 写入成功", Colors.GREEN)

    def read(self, path: str):
        """
        读取文件内容
        
        功能：
            - 读取并显示指定文件的内容
            - 验证文件存在性和类型
            - 格式化显示文件内容
            - 支持绝对路径和相对路径
        
        参数:
            path (str): 文件路径
        """
        parent, file_name = self.find_entry_by_path(path)
        if file_name not in parent.entries:
            raise Exception(f"文件 {file_name} 不存在")
            
        file = parent.entries[file_name]
        if not isinstance(file, File):
            raise Exception(f"{file_name} 不是文件")
            
        print_color("\n文件内容:", Colors.HEADER)
        print_color("-" * 40, Colors.HEADER)
        print(file.content)
        print_color("-" * 40, Colors.HEADER)

def main():
    """
    主程序入口
    
    功能：
        - 初始化文件系统
        - 提供命令行交互界面
        - 解析用户输入的命令
        - 处理命令参数
        - 调用相应的文件系统操作
        - 错误处理和提示
        - 支持带引号的参数处理
    """
    fs = FileSystem()
    while True:
        try:
            current_path = fs.get_current_path()
            command = input(f"{Colors.GREEN}{current_path}{Colors.ENDC}> ").strip()
            if not command:
                continue
            
            # 处理引号的参数，支持包含空格的文件名或内容
            parts = []
            current_part = []
            in_quotes = False
            for char in command:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ' ' and not in_quotes:
                    if current_part:
                        parts.append(''.join(current_part))
                        current_part = []
                else:
                    current_part.append(char)
            if current_part:
                parts.append(''.join(current_part))
            
            if not parts:
                continue
                
            # 解析命令和参数
            cmd = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []

            # 命令处理
            if cmd == "exit":
                print_color("感谢使用！��见！", Colors.GREEN)
                break
            elif cmd == "help":
                fs.show_help()
            elif cmd == "md" and len(args) == 1:
                fs.md(args[0])
                print_color(f"目录 '{args[0]}' 创建成功", Colors.GREEN)
            elif cmd == "rd" and len(args) == 1:
                fs.rd(args[0])
                print_color(f"目录 '{args[0]}' 删除成功", Colors.GREEN)
            elif cmd == "cd" and len(args) == 1:
                fs.cd(args[0])
                print_color(f"当前目录: {fs.get_current_path()}", Colors.BLUE)
            elif cmd == "mv" and len(args) == 2:
                fs.mv(args[0], args[1])
            elif cmd == "cp" and len(args) == 2:
                fs.cp(args[0], args[1])
            elif cmd == "dir" and len(args) == 0:
                for entry in fs.dir():
                    print(entry)
            elif cmd == "create" and len(args) == 2:
                fs.create_file(args[0], int(args[1]))
                print_color(f"文件 '{args[0]}' 创建成功", Colors.GREEN)
            elif cmd == "del" and len(args) == 1:
                fs.delete_file(args[0])
                print_color(f"文件 '{args[0]}' 删除成功", Colors.GREEN)
            elif cmd == "rename" and len(args) == 2:
                fs.rename(args[0], args[1])
            elif cmd == "write" and len(args) >= 2:
                fs.write(args[0], ' '.join(args[1:]))
            elif cmd == "read" and len(args) == 1:
                fs.read(args[0])
            elif cmd == "disk":
                print_color(fs.show_disk_usage(), Colors.YELLOW)
            elif cmd == "prop" and len(args) == 1:
                fs.show_properties(args[0])
            else:
                print_color("无效命令或参数错误！输入 'help' 获取帮助。", Colors.RED)
        except Exception as e:
            print_color(f"错误: {str(e)}", Colors.RED)

if __name__ == "__main__":
    main()
