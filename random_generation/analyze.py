#!/usr/bin/env python3
import re
import collections
import os
import glob
import argparse
import statistics
import matplotlib.pyplot as plt
import numpy as np

def analyze_problem_file(filename):
    """Analyze a problem file and extract statistics."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    problems = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            problem_name = lines[i].strip()
            problem_desc = lines[i + 1].strip()
            problems.append((problem_name, problem_desc))
    
    # Initialize statistics counters
    all_clause_counts = []
    all_construction_counts = []
    construction_names = collections.Counter()
    
    for problem_name, problem_desc in problems:
        # Split into statement and goal
        parts = problem_desc.split('?', 1)
        if len(parts) != 2:
            statement = problem_desc
        else:
            statement = parts[0].strip()
        
        # Count clauses
        clauses = statement.split(';')
        num_clauses = len(clauses)
        all_clause_counts.append(num_clauses)
        
        # Count constructions and their names
        total_constructions = 0
        for clause in clauses:
            if '=' not in clause:
                continue
            
            construction_part = clause.split('=', 1)[1].strip()
            constructions = construction_part.split(',')
            total_constructions += len(constructions)
            
            for construction in constructions:
                parts = construction.strip().split()
                if parts:  # Make sure there's at least one part (the name)
                    name = parts[0]
                    construction_names[name] += 1
        
        all_construction_counts.append(total_constructions)
    
    # Return statistics
    return {
        'problem_count': len(problems),
        'clause_counts': all_clause_counts,
        'construction_counts': all_construction_counts,
        'construction_names': construction_names
    }

def write_statistics_to_file(stats, output_file):
    """Write statistics to a file."""
    base_filename = os.path.splitext(output_file)[0]
    
    with open(output_file, 'w') as f:
        f.write(f"Total problems analyzed: {stats['problem_count']}\n")
        
        # Clause statistics
        if stats['clause_counts']:
            avg_clauses = sum(stats['clause_counts']) / len(stats['clause_counts'])
            min_clauses = min(stats['clause_counts'])
            max_clauses = max(stats['clause_counts'])
            median_clauses = statistics.median(stats['clause_counts'])
            
            # 计算众数
            mode_clauses = "N/A"
            if stats['clause_counts']:
                try:
                    mode_clauses = statistics.mode(stats['clause_counts'])
                except statistics.StatisticsError:
                    # 如果有多个众数
                    counter = collections.Counter(stats['clause_counts'])
                    max_count = max(counter.values())
                    modes = [k for k, v in counter.items() if v == max_count]
                    mode_clauses = ', '.join(map(str, modes))
            
            f.write("\nClause statistics:\n")
            f.write(f"  Average clauses per problem: {avg_clauses:.2f}\n")
            f.write(f"  Median clauses: {median_clauses}\n")
            f.write(f"  Mode clauses: {mode_clauses}\n")
            f.write(f"  Minimum clauses: {min_clauses}\n")
            f.write(f"  Maximum clauses: {max_clauses}\n")
        
        # Construction statistics
        if stats['construction_counts']:
            avg_constructions = sum(stats['construction_counts']) / len(stats['construction_counts'])
            min_constructions = min(stats['construction_counts'])
            max_constructions = max(stats['construction_counts'])
            median_constructions = statistics.median(stats['construction_counts'])
            
            # 计算众数
            mode_constructions = "N/A"
            if stats['construction_counts']:
                try:
                    mode_constructions = statistics.mode(stats['construction_counts'])
                except statistics.StatisticsError:
                    # 如果有多个众数
                    counter = collections.Counter(stats['construction_counts'])
                    max_count = max(counter.values())
                    modes = [k for k, v in counter.items() if v == max_count]
                    mode_constructions = ', '.join(map(str, modes))
            
            f.write("\nConstruction statistics:\n")
            f.write(f"  Average constructions per problem: {avg_constructions:.2f}\n")
            f.write(f"  Median constructions: {median_constructions}\n")
            f.write(f"  Mode constructions: {mode_constructions}\n")
            f.write(f"  Minimum constructions: {min_constructions}\n")
            f.write(f"  Maximum constructions: {max_constructions}\n")
        
        # Construction name distribution
        if stats['construction_names']:
            f.write("\nConstruction name distribution:\n")
            total_constructions = sum(stats['construction_names'].values())
            
            # Sort by frequency (most common first)
            for name, count in stats['construction_names'].most_common():
                percentage = (count / total_constructions) * 100
                f.write(f"  {name}: {count} occurrences ({percentage:.2f}%)\n")
    
    # 生成构造函数分布图表
    if stats['construction_names']:
        plt.figure(figsize=(12, 8))
        
        # 获取前20个最常见的构造函数
        top_constructions = stats['construction_names'].most_common(20)
        names = [name for name, _ in top_constructions]
        counts = [count for _, count in top_constructions]
        
        # 创建条形图
        bars = plt.bar(range(len(names)), counts)
        plt.xlabel('Construction Names')
        plt.ylabel('Frequency')
        plt.title('Top 20 Construction Functions')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        
        # 添加计数标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height+0.1,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        chart_path = f"{base_filename}_chart.png"
        plt.savefig(chart_path)
        plt.close()
        
        # # 如果有很多构造函数，创建饼图展示前10个和其他
        # if len(stats['construction_names']) > 10:
        #     plt.figure(figsize=(10, 10))
            
        #     top10 = stats['construction_names'].most_common(10)
        #     names = [name for name, _ in top10]
        #     counts = [count for _, count in top10]
            
        #     # 添加"其他"类别
        #     other_count = sum(stats['construction_names'].values()) - sum(counts)
        #     if other_count > 0:
        #         names.append('Other')
        #         counts.append(other_count)
            
        #     # 创建饼图
        #     plt.pie(counts, labels=names, autopct='%1.1f%%',
        #            shadow=True, startangle=90)
        #     plt.axis('equal')  # 确保饼图是圆的
        #     plt.title('Distribution of Construction Functions')
            
        #     # 保存饼图
        #     pie_chart_path = f"{base_filename}_pie_chart.png"
        #     plt.savefig(pie_chart_path)
        #     plt.close()

def main(dir):
    """Main function to analyze problem files."""
    files_to_analyze = [
        '../imo_ag_30.txt',
        '../jgex_ag_231.txt',
    ]
    files = glob.glob(os.path.join(dir, r'geometry_depth[0-9]*_pr.txt'))
    files_to_analyze += files
    # 创建输出目录（如果不存在）
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in files_to_analyze:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue
        
        stats = analyze_problem_file(filename)
        
        # 将统计结果写入文件
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        output_file = os.path.join(output_dir, f"{base_filename}_stats.txt")
        write_statistics_to_file(stats, output_file)
    
    # # Combine statistics for both files
    # if all(os.path.exists(f) for f in files_to_analyze):
    #     combined_stats = {
    #         'problem_count': 0,
    #         'clause_counts': [],
    #         'construction_counts': [],
    #         'construction_names': collections.Counter()
    #     }
        
    #     for filename in files_to_analyze:
    #         stats = analyze_problem_file(filename)
    #         combined_stats['problem_count'] += stats['problem_count']
    #         combined_stats['clause_counts'].extend(stats['clause_counts'])
    #         combined_stats['construction_counts'].extend(stats['construction_counts'])
    #         combined_stats['construction_names'].update(stats['construction_names'])
        
    #     # 将组合统计结果写入文件
    #     combined_output_file = os.path.join(output_dir, "stats_combined.txt")
    #     write_statistics_to_file(combined_stats, combined_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='dataset')
    args = parser.parse_args()
    main(args.dir)