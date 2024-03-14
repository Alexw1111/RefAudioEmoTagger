import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import glob
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionRecognitionPipeline:
    def __init__(self, model_path, model_revision="v2.0.4", device='cuda:0'):
        self.device = device
        self.pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_path,
            model_revision=model_revision,
            device=device
        )

    def batch_infer(self, audio_paths):
        rec_results = self.pipeline(audio_paths, granularity="utterance", extract_embedding=False)
        return rec_results

def get_top_emotion_with_confidence(recognition_results):
    emotions_with_confidence = []
    for result in recognition_results:
        labels = result['labels']
        scores = result['scores']
        max_score_index = scores.index(max(scores))
        emotion = labels[max_score_index].split('/')[0]  # 只取中文部分
        confidence = scores[max_score_index]
        emotions_with_confidence.append((emotion, confidence))
    return emotions_with_confidence

def process_batch_and_write_results(batch_audio_paths, recognizer, output_file):
    recognition_results = recognizer.batch_infer(batch_audio_paths)
    top_emotions_with_confidence = get_top_emotion_with_confidence(recognition_results)

    with open(output_file, 'a', encoding='utf-8') as f:
        for audio_path, (top_emotion, confidence) in zip(batch_audio_paths, top_emotions_with_confidence):
            parent_folder_name = os.path.basename(os.path.dirname(audio_path))
            write_str = f"{audio_path}|{parent_folder_name}|{top_emotion}|{confidence:.2f}\n"
            f.write(write_str)
            logging.info(f"写入文件：{write_str.strip()}")

def process_audio_files(folder_path, output_file, recognizer, batch_size=10, max_workers=4):
    if not os.path.exists(folder_path):
        logging.error(f"目录不存在：{folder_path}")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("AudioPath|ParentFolder|Emotion|Confidence\n")
        
    audio_paths = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith('.wav')]
    batches = [audio_paths[i:i + batch_size] for i in range(0, len(audio_paths), batch_size)]

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch_and_write_results, batch, recognizer, output_file) for batch in batches]
        
        for _ in as_completed(futures):
            pass
    
    total_inference_time = time.time() - start_time
    logging.info(f"Processed {len(audio_paths)} files in {folder_path}, total inference time: {total_inference_time:.2f} seconds")

# 示例初始化和使用代码
emotion_recognizer = EmotionRecognitionPipeline(model_path="iic/emotion2vec_base_finetuned", device='cuda:0')
folder_path = 'your_folder_path'
output_file = 'inference_results.list'
process_audio_files(folder_path, output_file, emotion_recognizer, batch_size=10, max_workers=4)