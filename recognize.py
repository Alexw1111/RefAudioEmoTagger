import os
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchaudio
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionRecognitionPipeline:
    def __init__(self, model_path="iic/emotion2vec_base_finetuned", model_revision="v2.0.4", device='cuda:0'):
        self.device = device
        self.target_sample_rate = 16000  # 目标采样率为16000 Hz
        self.pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_path,
            model_revision=model_revision,
            device=device
        )

    def batch_infer(self, audio_paths):
        waveforms, sample_rates = zip(*map(torchaudio.load, audio_paths))
        
        resampled_waveforms = []
        for waveform, sample_rate in zip(waveforms, sample_rates):
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            resampled_waveforms.append(waveform.to(self.device))

        rec_results = self.pipeline(resampled_waveforms, sample_rate=self.target_sample_rate, granularity="utterance", extract_embedding=False)
        return rec_results

def get_top_emotion_with_confidence(recognition_results):
    return [(result['labels'][result['scores'].index(max(result['scores']))].split('/')[0],
             max(result['scores'])) for result in recognition_results]

def process_batch_and_write_results(batch_audio_paths, recognizer, output_file):
    top_emotions_with_confidence = get_top_emotion_with_confidence(recognizer.batch_infer(batch_audio_paths))

    with open(output_file, 'a', encoding='utf-8') as f:
        f.writelines(f"{audio_path}|{os.path.basename(os.path.dirname(audio_path))}|{top_emotion}|{confidence:.2f}\n"
                     for audio_path, (top_emotion, confidence) in zip(batch_audio_paths, top_emotions_with_confidence))

    logging.info(f"Processed {len(batch_audio_paths)} files")

def process_audio_files(folder_path, output_file, recognizer, batch_size=10, max_workers=4):
    if not os.path.exists(folder_path):
        logging.error(f"目录不存在：{folder_path}")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("AudioPath|ParentFolder|Emotion|Confidence\n")

    audio_paths = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)
    batches = [audio_paths[i:i + batch_size] for i in range(0, len(audio_paths), batch_size)]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch_and_write_results, batch, recognizer, output_file) for batch in batches]
        for _ in as_completed(futures):
            pass

    logging.info(f"Processed {len(audio_paths)} files in {folder_path}, "
                 f"total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='识别音频文件中的情感')
    parser.add_argument('--folder_path', type=str, required=True, help='包含音频文件的文件夹路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件的路径')
    parser.add_argument('--model_revision', type=str, default="v2.0.4", help='情感识别模型的修订版本')
    parser.add_argument('--batch_size', type=int, default=10, help='推理的批量大小')
    parser.add_argument('--max_workers', type=int, default=4, help='并行处理的最大工作线程数')

    args = parser.parse_args()

    emotion_recognizer = EmotionRecognitionPipeline(model_revision=args.model_revision)
    process_audio_files(args.folder_path, args.output_file, emotion_recognizer, args.batch_size, args.max_workers)
