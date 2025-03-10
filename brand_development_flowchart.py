import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建画布
plt.figure(figsize=(14, 10))
ax = plt.gca()
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)

# 设置背景为网格状
ax.set_facecolor('#f9f9f9')
ax.grid(True, linestyle='--', alpha=0.7, color='#dddddd')

# 定义颜色方案
colors = {
    'background': '#f9f9f9',
    'main_box': '#ffcccb',  # 淡粉色
    'left_box': '#fffacd',  # 淡黄色
    'right_box': '#ffe4c4',  # 淡橙色
    'middle_box': '#e6e6fa',  # 淡紫色
    'arrow': '#708090',  # 深灰色
    'bracket': '#696969'  # 暗灰色
}

# 绘制左侧方框
left_boxes = [
    {'pos': [1, 7.5, 2.5, 0.8], 'text': '产品价格实惠', 'fontsize': 12},
    {'pos': [1, 6, 2.5, 0.8], 'text': '国货老品牌', 'fontsize': 12},
    {'pos': [1, 4.5, 2.5, 0.8], 'text': '国家推行食品\n"三减二健"政策', 'fontsize': 12}
]

for box in left_boxes:
    rect = patches.FancyBboxPatch(
        (box['pos'][0], box['pos'][1]), box['pos'][2], box['pos'][3],
        boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
        facecolor=colors['left_box'], edgecolor='none', alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(box['pos'][0] + box['pos'][2]/2, box['pos'][1] + box['pos'][3]/2, 
            box['text'], ha='center', va='center', fontsize=box['fontsize'], 
            fontweight='bold', color='#333333')

# 绘制右侧方框
right_boxes = [
    {'pos': [10.5, 7.5, 2.5, 0.8], 'text': '包装简陋', 'fontsize': 12},
    {'pos': [10.5, 6, 2.5, 0.8], 'text': '品牌宣传力度低', 'fontsize': 12},
    {'pos': [10.5, 4.5, 2.5, 0.8], 'text': '创新发展力步骤', 'fontsize': 12}
]

for box in right_boxes:
    rect = patches.FancyBboxPatch(
        (box['pos'][0], box['pos'][1]), box['pos'][2], box['pos'][3],
        boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
        facecolor=colors['right_box'], edgecolor='none', alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(box['pos'][0] + box['pos'][2]/2, box['pos'][1] + box['pos'][3]/2, 
            box['text'], ha='center', va='center', fontsize=box['fontsize'], 
            fontweight='bold', color='#333333')

# 绘制中间上方方框
main_box = patches.FancyBboxPatch(
    (5, 7), 4, 0.8,
    boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
    facecolor=colors['main_box'], edgecolor='none', alpha=0.9
)
ax.add_patch(main_box)
ax.text(7, 7.4, '狗牙儿品牌发展必要性', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#333333')

# 绘制中间下方椭圆
ellipse = patches.Ellipse(
    (7, 3.5), 6, 1.2,
    facecolor=colors['main_box'], edgecolor='none', alpha=0.9
)
ax.add_patch(ellipse)
ax.text(7, 3.5, '狗牙儿品牌与年轻市场的双向对话', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#333333')

# 绘制左下方框
left_bottom_box = patches.FancyBboxPatch(
    (1.5, 3.5), 2, 0.8,
    boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
    facecolor=colors['middle_box'], edgecolor='none', alpha=0.9
)
ax.add_patch(left_bottom_box)
ax.text(2.5, 3.9, '调查意图', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#333333')

# 绘制右下方框
right_bottom_box = patches.FancyBboxPatch(
    (10.5, 3.5), 2, 0.8,
    boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
    facecolor=colors['middle_box'], edgecolor='none', alpha=0.9
)
ax.add_patch(right_bottom_box)
ax.text(11.5, 3.9, '受访人群', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#333333')

# 绘制左侧现状背景框
left_bg = patches.FancyBboxPatch(
    (3.7, 4.3), 1.3, 4,
    boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
    facecolor=colors['right_box'], edgecolor='none', alpha=0.8
)
ax.add_patch(left_bg)
ax.text(4.35, 6.3, '现\n状\n背\n景', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#333333')

# 绘制右侧现存问题框
right_bg = patches.FancyBboxPatch(
    (9, 4.3), 1.3, 4,
    boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
    facecolor=colors['right_box'], edgecolor='none', alpha=0.8
)
ax.add_patch(right_bg)
ax.text(9.65, 6.3, '现\n存\n问\n题', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#333333')

# 绘制左下方虚线框
left_dotted_box = patches.FancyBboxPatch(
    (1, 1.5), 6, 1.2,
    boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
    facecolor='none', edgecolor='#333333', alpha=0.5, linestyle='dotted', linewidth=2
)
ax.add_patch(left_dotted_box)

# 绘制右下方虚线框
right_dotted_box = patches.FancyBboxPatch(
    (7.5, 1.5), 5, 1.2,
    boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
    facecolor='none', edgecolor='#333333', alpha=0.5, linestyle='dotted', linewidth=2
)
ax.add_patch(right_dotted_box)

# 绘制左下方小方框
left_small_boxes = [
    {'pos': [1.5, 2, 1.2, 0.6], 'text': '购买意愿'},
    {'pos': [3, 2, 1.2, 0.6], 'text': '购买偏好'},
    {'pos': [4.5, 2, 1.2, 0.6], 'text': '购买途径'},
    {'pos': [6, 2, 1.2, 0.6], 'text': '购买频率'}
]

for box in left_small_boxes:
    rect = patches.FancyBboxPatch(
        (box['pos'][0], box['pos'][1]), box['pos'][2], box['pos'][3],
        boxstyle=patches.BoxStyle("Round", pad=0.2, rounding_size=0.1),
        facecolor=colors['left_box'], edgecolor='none', alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(box['pos'][0] + box['pos'][2]/2, box['pos'][1] + box['pos'][3]/2, 
            box['text'], ha='center', va='center', fontsize=10, 
            fontweight='bold', color='#333333')

# 绘制右下方小方框
right_small_box = patches.FancyBboxPatch(
    (8, 2), 4, 0.6,
    boxstyle=patches.BoxStyle("Round", pad=0.2, rounding_size=0.1),
    facecolor=colors['right_box'], edgecolor='none', alpha=0.9
)
ax.add_patch(right_small_box)
ax.text(10, 2.3, '河北省及其周边省份大学生', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='#333333')

# 绘制左侧大括号
left_bracket_verts = [
    (3.5, 7.9),  # 起点
    (3.3, 7.9),  # 控制点
    (3.3, 6.3),  # 控制点
    (3.5, 6.3),  # 中点上
    (3.3, 6.3),  # 控制点
    (3.3, 4.7),  # 控制点
    (3.5, 4.7),  # 终点
]
left_bracket_codes = [
    Path.MOVETO,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
]
left_bracket_path = Path(left_bracket_verts, left_bracket_codes)
left_bracket_patch = patches.PathPatch(
    left_bracket_path, facecolor='none', edgecolor=colors['bracket'], lw=2
)
ax.add_patch(left_bracket_patch)

# 绘制右侧大括号
right_bracket_verts = [
    (10.5, 7.9),  # 起点
    (10.7, 7.9),  # 控制点
    (10.7, 6.3),  # 控制点
    (10.5, 6.3),  # 中点上
    (10.7, 6.3),  # 控制点
    (10.7, 4.7),  # 控制点
    (10.5, 4.7),  # 终点
]
right_bracket_codes = [
    Path.MOVETO,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
]
right_bracket_path = Path(right_bracket_verts, right_bracket_codes)
right_bracket_patch = patches.PathPatch(
    right_bracket_path, facecolor='none', edgecolor=colors['bracket'], lw=2
)
ax.add_patch(right_bracket_patch)

# 绘制箭头
arrows = [
    {'start': (5, 7.4), 'end': (4.5, 7.4)},  # 左侧背景到主框
    {'start': (9, 7.4), 'end': (9.5, 7.4)},  # 右侧背景到主框
    {'start': (7, 6.8), 'end': (7, 4.2)},    # 主框到椭圆
    {'start': (4, 3.5), 'end': (4.5, 3.5)},  # 左下框到椭圆
    {'start': (9.5, 3.5), 'end': (10, 3.5)}, # 椭圆到右下框
    {'start': (2.5, 3.3), 'end': (2.5, 2.8)},# 左下框到虚线框
    {'start': (11.5, 3.3), 'end': (11.5, 2.8)}# 右下框到虚线框
]

for arrow in arrows:
    ax.annotate('', 
                xy=arrow['end'], 
                xytext=arrow['start'],
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))

# 移除坐标轴
plt.axis('off')

# 添加标题
plt.title('狗牙儿品牌发展流程图', fontsize=18, fontweight='bold', pad=20)

# 保存图片
plt.tight_layout()
plt.savefig('brand_development_flowchart.png', dpi=300, bbox_inches='tight')
plt.show() 