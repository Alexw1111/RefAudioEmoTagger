import os
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchaudio
import glob
import pandas as pd
import torch
import asyncio
import gc
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    model_config = ConfigDict(arbitrary_types_allowed=True)

async def process_batch(batch_audio_paths, recognizer):
    waveforms, sample_rates = zip(*[torchaudio.load(path) for path in batch_audio_paths])
    resampled_waveforms = [torchaudio.transforms.Resample(sr, 16000)(waveform) for waveform, sr in zip(waveforms, sample_rates)]
    results = recognizer(resampled_waveforms, sampled_rate=16000, granularity="utterance", extract_embedding=False)
    
    processed_results = []
    for audio_path, result in zip(batch_audio_paths, results):
        scores = result['scores']
        labels = result['labels']
        max_score_index = scores.index(max(scores))
        max_score_label = labels[max_score_index]
        processed_results.append((audio_path, max_score_label, scores[max_score_index]))
    
    return processed_results
def audio_path_generator(folder_path):
    audio_paths = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)
    for audio_path in audio_paths:
        yield audio_path

async def process_audio_files(folder_path, recognizer, batch_size=64, max_workers=4):
    if not os.path.exists(folder_path):
        logging.error(f"目录不存在：{folder_path}")
        return None

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch = []
        for audio_path in audio_path_generator(folder_path):
            batch.append(audio_path)
            if len(batch) == batch_size:
                results.extend(await process_batch(batch, recognizer))
                batch = []
                gc.collect()  # 主动调用垃圾回收

        if batch:
            results.extend(await process_batch(batch, recognizer))

    logging.info(f"Processed files in {folder_path}, total time: {time.time() - start_time:.2f} seconds")
    return results

async def main(args):
    emotion_recognizer = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_plus_large"
    )
    
    audio_emotion_results = await process_audio_files(args.folder_path, emotion_recognizer, args.batch_size, args.max_workers)

    if audio_emotion_results is None:
        return

    df = pd.DataFrame(audio_emotion_results, columns=['AudioPath', 'AudioEmotion', 'Confidence'])
    df['ParentFolder'] = df['AudioPath'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    output_file = args.output_file
    df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用emotion2vec+模型识别音频文件中的情感')
    parser.add_argument('--folder_path', type=str, required=True, help='包含音频文件的文件夹路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件的路径')
    parser.add_argument('--batch_size', type=int, default=64, help='推理的批量大小')
    parser.add_argument('--max_workers', type=int, default=4, help='并行处理的最大工作线程数')
    args = parser.parse_args()
    asyncio.run(main(args))
