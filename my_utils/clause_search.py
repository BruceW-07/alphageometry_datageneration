import os
import re

def read_problems(filename):
    """读取问题文件，返回问题名和描述的列表"""
    problems = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    name = lines[i].strip()
                    desc = lines[i + 1].strip()
                    problems.append((name, desc))
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 不存在")
    except Exception as e:
        print(f"读取文件 '{filename}' 时出错: {e}")
    return problems

def analyze_clause(clause, problem_name):
    """分析一个子句，检查是否满足条件"""
    try:
        if '=' not in clause:
            return None
        
        parts = clause.split('=')
        if len(parts) != 2:
            return None
        
        points_part = parts[0].strip()
        constructions_part = parts[1].strip()
        
        # 识别点 - 点是使用空格分隔的
        points = points_part.split()
        
        # 计算构造的数量
        constructions = constructions_part.split(',')
        
        # print(f"length of points: {len(points)}, length of constructions: {len(constructions)}")

        if len(points) >= 2 and len(constructions) >= 2:
            return {
                'problem': problem_name,
                'clause': clause,
                'points': points,
                'constructions': constructions,
                'points_count': len(points),
                'constructions_count': len(constructions)
            }
    except Exception as e:
        print(f"分析子句时出错 '{clause}': {e}")
    
    return None

def search_clauses(filename):
    """在文件中搜索满足条件的子句"""
    results = []
    problems = read_problems(filename)
    
    for name, desc in problems:
        try:
            # 分割陈述和结论
            if '?' in desc:
                statement, _ = desc.split('?', 1)
            else:
                statement = desc
            
            # 分割子句
            clauses = statement.split(';')
            
            for clause in clauses:
                result = analyze_clause(clause.strip(), name)
                if result:
                    results.append(result)
        except Exception as e:
            print(f"处理问题 '{name}' 时出错: {e}")
    
    return results

def main():
    base_dir = '/home/tttcat/data_generation/alphageometry_my'
    files = [
        os.path.join(base_dir, 'imo_ag_30.txt'),
        os.path.join(base_dir, 'jgex_ag_231.txt')
    ]
    all_results = []
    
    for filename in files:
        print(f"处理文件 {filename}...")
        results = search_clauses(filename)
        all_results.extend(results)
    
    if not all_results:
        print("未找到满足条件的子句")
        return
    
    print(f"\n找到 {len(all_results)} 个满足条件的子句:")
    for i, result in enumerate(all_results, 1):
        print(f"\n{i}. 问题: {result['problem']}")
        print(f"   子句: {result['clause']}")
        print(f"   点数量: {result['points_count']}, 点: {', '.join(result['points'])}")
        print(f"   构造数量: {result['constructions_count']}, 构造: {', '.join(result['constructions'])}")

if __name__ == "__main__":
    main()
