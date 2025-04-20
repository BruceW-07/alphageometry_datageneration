import re
from collections import defaultdict
from itertools import combinations
import are_same_figure as same

# 所有构造名称及编号（根据你给的定义）
definitions = [
    "angle_bisector", "angle_mirror", "circle", "circumcenter", "eq_quadrangle",
    "eq_trapezoid", "eq_triangle", "eqangle2", "eqdia_quadrangle", "eqdistance",
    "foot", "free", "incenter", "incenter2", "excenter", "excenter2",
    "centroid", "ninepoints", "intersection_cc", "intersection_lc",
    "intersection_ll", "intersection_lp", "intersection_lt", "intersection_pp",
    "intersection_tt", "iso_triangle", "lc_tangent", "midpoint", "mirror",
    "nsquare", "on_aline", "on_aline2", "on_bline", "on_circle", "on_line",
    "on_pline", "on_tline", "orthocenter", "parallelogram", "pentagon",
    "psquare", "quadrangle", "r_trapezoid", "r_triangle", "rectangle", "reflect",
    "risos", "s_angle", "segment", "shift", "square", "isquare", "trapezoid",
    "triangle", "triangle12", "2l1c", "e5128", "3peq", "trisect", "trisegment",
    "on_dia", "ieq_triangle", "on_opline", "cc_tangent0", "cc_tangent",
    "eqangle3", "tangent", "on_circum", "eqangle", "eqratio", "perp", "para", "cong",
    "cyclic", "coll", "midp"
]
construct_to_id = {name: idx + 1 for idx, name in enumerate(definitions)}

construct_pattern = re.compile(r'=\s*(\w+)')
target_pattern = re.compile(r'\?\s*(\w+)')

structure_map = defaultdict(list)
geometry_map = {}

def process_geometry_block(data_num, content):
    if '?' not in content:
        print(f"⚠️ 数据 {data_num} 缺少问号，跳过")
        return
    before_q, after_q = content.split('?', 1)
    constructs = construct_pattern.findall(before_q)
    construct_ids = sorted(construct_to_id[c] for c in constructs if c in construct_to_id)

    target_match = target_pattern.search('?' + after_q)
    if not target_match:
        print(f"⚠️ 数据 {data_num} 缺少目标构造，跳过")
        return
    target_keyword = target_match.group(1)
    if target_keyword not in construct_to_id:
        print(f"⚠️ 数据 {data_num} 的目标构造 {target_keyword} 不在定义中")
        return
    target_id = construct_to_id[target_keyword]

    key = (tuple(construct_ids), target_id)
    structure_map[key].append(data_num)
    geometry_map[data_num] = content

# 读取数据
with open("geometry_depth15_pr.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
while i < len(lines):
    if lines[i].isdigit():
        data_num = int(lines[i])
        i += 1
        if i < len(lines):
            content = lines[i]
            process_geometry_block(data_num, content)
    i += 1

# 输出判断为真正“相同图形”的编号对
with open("same_figures.txt", "w", encoding="utf-8") as out_file:
    out_file.write("✅ 以下编号对应的图形真正相等：\n\n")
    for key, data_nums in structure_map.items():
        if len(data_nums) > 1:
            for a, b in combinations(data_nums, 2):
                fig1 = geometry_map[a]
                fig2 = geometry_map[b]
                if same.are_same_figure(fig1, fig2):
                    out_file.write(f"{a} 和 {b} 是相同图形\n")

print("✅ 真正相等的图形对已写入 same_figures.txt")
