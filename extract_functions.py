import re
import os

# 文件路径
file_path = '/home/tttcat/data_generation/alphageometry_my/numericals.py'

# 读取文件内容
with open(file_path, 'r') as f:
    content = f.read()

# 使用正则表达式找出所有sketch_开头的函数定义
sketch_funcs = re.findall(r'def\s+(sketch_\w+)\s*\(', content)

# 去掉sketch_前缀
func_names = [func.replace('sketch_', '') for func in sketch_funcs]

# 按字母顺序排序(可选)
func_names.sort()

print(f"从numericals.py提取的numerics函数: {len(func_names)}个")

# 读取defs.txt文件并处理construction和numerics
defs_path = '/home/tttcat/data_generation/alphageometry_my/defs.txt'
constructions_data = []  # 用于存储每个construction及其numerics信息

try:
    with open(defs_path, 'r') as f:
        lines = f.readlines()
    
    # 每6行一个construction
    for i in range(0, len(lines), 6):
        if i+5 < len(lines):
            construction_name = lines[i].strip().split()[0]  # 第一行第一个单词为construction名称
            
            # 提取第5行的numerics（索引为i+4）
            numerics_line = lines[i+4].strip()
            numerics = []
            if numerics_line:
                # 处理每个逗号分隔的numeric，取第一项作为名称
                numeric_items = numerics_line.split(',')
                for item in numeric_items:
                    item = item.strip()
                    if item:
                        numeric_name = item.split()[0]  # 取第一项
                        numerics.append(numeric_name)
            
            # 存储construction及其numerics
            constructions_data.append({
                'name': construction_name,
                'numerics': numerics
            })
    
    # 检查每个construction的numerics是否都在func_names中
    satisfied_constructions = []
    unsatisfied_constructions = []
    
    for construction in constructions_data:
        all_numerics_found = True
        missing_numerics = []
        
        for numeric in construction['numerics']:
            if numeric not in func_names:
                all_numerics_found = False
                missing_numerics.append(numeric)
        
        if all_numerics_found:
            satisfied_constructions.append(construction['name'])
        else:
            unsatisfied_constructions.append({
                'name': construction['name'],
                'missing_numerics': missing_numerics
            })
    
    # 输出满足条件的construction
    print("\n满足条件的constructions（所有numerics都在numericals.py中）:")
    satisfied_output = "ALL = [\n"
    for cons in satisfied_constructions:
        satisfied_output += f"    '{cons}',\n"
    satisfied_output += "]"
    print(satisfied_output)
    print(f"共{len(satisfied_constructions)}个满足条件的constructions")
    
    # 输出不满足条件的construction及其缺失的numerics
    print("\n不满足条件的constructions（部分numerics不在numericals.py中）:")
    for cons in unsatisfied_constructions:
        print(f"- {cons['name']}: 缺失的numerics: {', '.join(cons['missing_numerics'])}")
    print(f"共{len(unsatisfied_constructions)}个不满足条件的constructions")
    
    # 在imo_ag_30.txt和jgex_ag_231.txt中搜索不满足条件的construction
    search_files = [
        '/home/tttcat/data_generation/alphageometry_my/imo_ag_30.txt',
        '/home/tttcat/data_generation/alphageometry_my/jgex_ag_231.txt'
    ]
    
    for search_file in search_files:
        if os.path.exists(search_file):
            print(f"\n在{os.path.basename(search_file)}中搜索不满足条件的constructions:")
            
            with open(search_file, 'r') as f:
                file_content = f.read()
            
            for cons in unsatisfied_constructions:
                if cons['name'] in file_content:
                    print(f"- {cons['name']}: 已找到")
                else:
                    print(f"- {cons['name']}: 未找到")
        else:
            print(f"\n文件{search_file}不存在")
            
except FileNotFoundError as e:
    print(f"错误: 文件不存在 - {e}")
except Exception as e:
    print(f"发生错误: {e}")