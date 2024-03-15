import os
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionRecognitionPipeline:
    def __init__(self, model_path="iic/emotion2vec_base_finetuned", model_revision="v2.0.4", device='cuda:0'):
        self.pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_path,
            model_revision=model_revision,
            device=device
        )

    def batch_infer(self, audio_paths):
        return self.pipeline(audio_paths, granularity="utterance", extract_embedding=False)

def get_top_emotion_with_confidence(recognition_results):
    return [(result['labels'][result['scores'].index(max(result['scores']))].split('/')[0], 
             max(result['scores'])) for result in recognition_results]

def process_batch_and_write_results(batch_audio_paths, recognizer, output_file):
    top_emotions_with_confidence = get_top_emotion_with_confidence(recognizer.batch_infer(batch_audio_paths))
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows([[audio_path, os.path.basename(os.path.dirname(audio_path)), 
                          top_emotion, f"{confidence:.2f}"] 
                         for audio_path, (top_emotion, confidence) in 
                         zip(batch_audio_paths, top_emotions_with_confidence)])
    logging.info(f"Processed {len(batch_audio_paths)} files")

def process_audio_files(folder_path, output_file, recognizer, batch_size=10, max_workers=4):
    if not os.path.exists(folder_path):
        logging.error(f"目录不存在：{folder_path}")
        return

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(["AudioPath", "ParentFolder", "Emotion", "Confidence"])

    audio_paths = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if
                   file.endswith('.wav')]
    batches = [audio_paths[i:i + batch_size] for i in range(0, len(audio_paths), batch_size)]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch_and_write_results, batch, recognizer, output_file) for batch in batches]
        for _ in as_completed(futures):
            pass

    logging.info(f"Processed {len(audio_paths)} files in {folder_path}, "
                 f"total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion recognition for audio files')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing audio files')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')

    args = parser.parse_args()

    emotion_recognizer = EmotionRecognitionPipeline()
    process_audio_files(args.folder_path, args.output_file, emotion_recognizer)
