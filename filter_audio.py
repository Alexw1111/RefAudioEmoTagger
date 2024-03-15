import os
import logging
import argparse
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_audio(src_folder, dst_folder, min_duration=3, max_duration=10):
    if not os.path.exists(src_folder):
        logging.error(f"Source folder does not exist: {src_folder}")
        return

    os.makedirs(dst_folder, exist_ok=True)

    for root, _, files in os.walk(src_folder):
        for file in filter(lambda f: f.endswith('.wav'), files):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_folder, file)

            duration = AudioSegment.from_wav(src_path).duration_seconds
            if min_duration <= duration <= max_duration:
                os.rename(src_path, dst_path)
                logging.info(f"Moved {src_path} to {dst_path}")
            else:
                logging.warning(f"Skipped {src_path} (duration: {duration:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter audio files by duration')
    parser.add_argument('src_folder', help='Path to the source folder')
    parser.add_argument('dst_folder', help='Path to the destination folder')
    parser.add_argument('-min', '--min_duration', type=float, default=3, help='Minimum duration in seconds (default: 3)')
    parser.add_argument('-max', '--max_duration', type=float, default=10, help='Maximum duration in seconds (default: 10)')

    args = parser.parse_args()

    filter_audio(args.src_folder, args.dst_folder, args.min_duration, args.max_duration)
