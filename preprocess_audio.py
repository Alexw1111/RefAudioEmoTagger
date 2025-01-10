import os
import logging
import argparse
from pydub import AudioSegment
import glob
import re

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_filename(filename):
    """清理文件名中的无效字符，并限制长度"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename.strip(' .'))[:255]

def rename_wav_with_lab(directory):
    """使用.lab文件中的信息重命名对应的.wav文件"""
    lab_files = glob.glob(os.path.join(directory, "**", "*.lab"), recursive=True)
    renamed_count = 0

    logging.info(f"找到 {len(lab_files)} 个 .lab 文件。")

    for lab_file in lab_files:
        wav_file = os.path.splitext(lab_file)[0] + ".wav"

        if not os.path.exists(wav_file):
            logging.warning(f"找不到对应的WAV文件: {wav_file}")
            continue

        with open(lab_file, 'r', encoding='utf-8') as f:
            new_name_parts = [line.strip() for line in f if line.strip()]
            if not new_name_parts:
                logging.warning(f"lab文件 {lab_file} 为空或无效")
                continue

        new_name = sanitize_filename('_'.join(new_name_parts))
        new_wav_file = os.path.join(os.path.dirname(wav_file), f"{new_name}.wav")

        if new_wav_file != wav_file:
            try:
                os.rename(wav_file, new_wav_file)
                renamed_count += 1
                logging.info(f"重命名: {wav_file} -> {new_wav_file}")
                os.remove(lab_file)
                logging.info(f"删除 .lab 文件: {lab_file}")
            except OSError as e:
                logging.error(f"重命名或删除文件时出错: {e}")

    logging.info(f"共重命名了 {renamed_count} 个文件")
    return renamed_count

def rename_wav_with_list(list_file, wav_folder):
    """使用.list文件中的信息重命名.wav文件"""
    renamed_count = 0
    
    if not os.path.exists(list_file):
        logging.error(f"指定的.list文件不存在: {list_file}")
        return renamed_count

    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 4:
            logging.warning(f"格式错误: {line}")
            continue

        wav_name = os.path.basename(parts[0])
        new_name = sanitize_filename(parts[-1])
        wav_files = glob.glob(os.path.join(wav_folder, "**", wav_name), recursive=True)

        if not wav_files:
            logging.warning(f"找不到音频文件: {wav_name}")
            continue

        wav_file = wav_files[0]
        new_wav_file = os.path.join(os.path.dirname(wav_file), f"{new_name}.wav")

        if new_wav_file != wav_file:
            try:
                os.rename(wav_file, new_wav_file)
                renamed_count += 1
                logging.info(f"重命名: {wav_file} -> {new_wav_file}")
            except OSError as e:
                logging.error(f"重命名文件时出错: {e}")

    logging.info(f"共重命名了 {renamed_count} 个文件")
    return renamed_count

def filter_audio(src_folder, dst_folder=None, min_duration=3, max_duration=10, copy_parent_folder=False):
    """根据音频时长过滤文件"""
    if not os.path.exists(src_folder):
        logging.error(f"源文件夹不存在: {src_folder}")
        return src_folder

    audio_files = glob.glob(os.path.join(src_folder, "**", "*.wav"), recursive=True)
    filtered_folder = dst_folder or src_folder

    for src_path in audio_files:
        dst_path = (os.path.join(os.path.dirname(src_path), f"filtered_{os.path.basename(src_path)}")
                    if dst_folder is None else os.path.join(dst_folder, os.path.relpath(src_path, src_folder)))
        
        if copy_parent_folder and dst_folder:
            parent_folder = os.path.basename(src_folder)
            dst_parent_folder = os.path.join(dst_folder, parent_folder)
            dst_path = os.path.join(dst_parent_folder, os.path.relpath(src_path, src_folder))

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        duration = AudioSegment.from_wav(src_path).duration_seconds
        if min_duration <= duration <= max_duration:
            AudioSegment.from_wav(src_path).export(dst_path, format="wav")
            logging.info(f"已复制: {src_path} -> {dst_path}")
        else:
            logging.warning(f"跳过: {src_path} (时长: {duration:.2f}秒)")

    logging.info(f"过滤后的音频已保存在 {filtered_folder}")
    return filtered_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='重命名并筛选音频文件')
    parser.add_argument('src_folder', help='源文件夹路径')
    parser.add_argument('-dst', '--dst_folder', help='目标文件夹路径')
    parser.add_argument('-min', '--min_duration', type=float, default=3, help='最小时长(秒), 默认为3秒')
    parser.add_argument('-max', '--max_duration', type=float, default=10, help='最大时长(秒)')
    parser.add_argument('-d', '--disable_filter', action='store_true', help='禁用音频筛选')
    parser.add_argument('-r', '--rename_method', choices=['lab', 'list'], required=True, help='重命名方式：根据.lab文件或.list文件')

    args = parser.parse_args()

    # 先执行重命名操作
    if args.rename_method == 'lab':
        renamed_count = rename_wav_with_lab(args.src_folder)
    elif args.rename_method == 'list':
        list_file = input("请输入.list文件路径: ")
        renamed_count = rename_wav_with_list(list_file, args.src_folder)

    logging.info(f"重命名完成，共重命名 {renamed_count} 个文件")

    # 然后进行音频筛选
    if not args.disable_filter:
        filter_audio(args.src_folder, args.dst_folder, args.min_duration, args.max_duration, copy_parent_folder=True)
        logging.info(f"音频文件筛选完成，保存在 {args.dst_folder or args.src_folder}")
    else:
        logging.info("音频筛选已禁用")

    logging.info("处理完成")
