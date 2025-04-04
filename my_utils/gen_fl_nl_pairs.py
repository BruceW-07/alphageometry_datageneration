import os
import csv
import glob
import argparse

def extract_fl_nl_pairs(input_dir, output_file):
    """
    从指定目录下的所有CSV文件中提取fl_statement和nl_statement，
    并以<fl>...\n<nl>....\n\n的格式写入到输出文件中，
    同时进行去重处理
    """
    # 确保输入目录路径以斜杠结尾
    if not input_dir.endswith('/'):
        input_dir += '/'
    
    # 获取所有CSV文件的路径
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"警告：在 {input_dir} 中没有找到CSV文件")
        return
    
    # 记录处理的行数和去重后的行数
    total_rows = 0
    unique_rows = 0
    duplicate_rows = 0
    
    # 用于去重的集合
    unique_pairs = set()
    
    try:
        # 打开输出文件
        with open(output_file, 'w', encoding='utf-8') as outf:
            # 处理每个CSV文件
            for csv_file in csv_files:
                print(f"处理文件: {csv_file}")
                
                try:
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        
                        for row in reader:
                            # 提取fl_statement和nl_statement
                            fl_statement = row.get('fl_statement', '')
                            nl_statement = row.get('nl_statement', '')
                            
                            # 构造唯一键用于去重
                            pair_key = (fl_statement, nl_statement)
                            
                            # 检查是否为重复项
                            if pair_key not in unique_pairs:
                                # 写入输出文件
                                outf.write(f"<fl>{fl_statement}\n<nl>{nl_statement}\n\n")
                                unique_pairs.add(pair_key)
                                unique_rows += 1
                            else:
                                duplicate_rows += 1
                            
                            total_rows += 1
                            
                except Exception as e:
                    print(f"处理文件 {csv_file} 时出错: {str(e)}")
                    continue
        
        print(f"处理完成，共读取了 {total_rows} 对fl-nl对")
        print(f"去重后保留了 {unique_rows} 对唯一的fl-nl对")
        print(f"发现并移除了 {duplicate_rows} 对重复的fl-nl对")
        
    except Exception as e:
        print(f"写入输出文件 {output_file} 时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='提取CSV文件中的fl_statement和nl_statement')
    parser.add_argument('--input-dir', '-i', type=str, 
                        default="../random_generation/dataset/geometry_w_proof_mcq_depth5/",
                        help='包含CSV文件的输入目录')
    parser.add_argument('--output-file', '-o', type=str, 
                        default="fl_nl_pairs.txt",
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    extract_fl_nl_pairs(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
