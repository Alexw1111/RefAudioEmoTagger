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

EMOTION_MAPPING = {  #使音频情感分类和文本情感分类输出一致
    '恐惧': '恐惧',
    '愤怒': '生气',
    '厌恶': '厌恶', 
    '喜好': '开心',
    '悲伤': '难过',
    '高兴': '开心',
    '惊讶': '吃惊'
}

class EmotionRecognitionPipeline:
    def __init__(self, model_path="iic/emotion2vec_base_finetuned", model_revision="v2.0.4", device='cuda:0', target_sample_rate=16000):
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_path,
            model_revision=model_revision,
            device=device
        )

    def batch_infer(self, audio_paths):
        waveforms, sample_rates = zip(*map(torchaudio.load, audio_paths))
        resampled_waveforms = [self._resample_waveform(waveform, sample_rate) for waveform, sample_rate in zip(waveforms, sample_rates)]
        return self.pipeline(resampled_waveforms, sample_rate=self.target_sample_rate, granularity="utterance", extract_embedding=False)

    def _resample_waveform(self, waveform, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        return waveform.to(self.device)

def get_top_emotion_with_confidence(recognition_results):
    return [(result['labels'][result['scores'].index(max(result['scores']))].split('/')[0], max(result['scores'])) for result in recognition_results]

def process_batch_and_write_results(batch_audio_paths, recognizer, output_file):
    top_emotions_with_confidence = get_top_emotion_with_confidence(recognizer.batch_infer(batch_audio_paths))
    with open(output_file, 'a', encoding='utf-8') as f:
        f.writelines(f"{audio_path}|{os.path.basename(os.path.dirname(audio_path))}|{top_emotion}|{confidence:.2f}\n" for audio_path, (top_emotion, confidence) in zip(batch_audio_paths, top_emotions_with_confidence))
    logging.info(f"Processed {len(batch_audio_paths)} files")

def process_audio_files(folder_path, output_file, recognizer, batch_size=10, max_workers=4):
    if not os.path.exists(folder_path):
        logging.error(f"目录不存在：{folder_path}")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("AudioPath|ParentFolder|AudioEmotion|Confidence\n")

    audio_paths = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)
    batches = [audio_paths[i:i + batch_size] for i in range(0, len(audio_paths), batch_size)]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch_and_write_results, batch, recognizer, output_file) for batch in batches]
        for _ in as_completed(futures):
            pass

    logging.info(f"Processed {len(audio_paths)} files in {folder_path}, total time: {time.time() - start_time:.2f} seconds")

def process_text_emotion(audio_emotion_file, text_emotion_file, text_classifier):
    with open(audio_emotion_file, 'r', encoding='utf-8') as f_in, open(text_emotion_file, 'w', encoding='utf-8') as f_out:
        header = f_in.readline().strip()
        f_out.write(f"{header}|TextEmotion\n")
        for line in f_in:
            audio_path, *rest = line.strip().split('|')
            text = os.path.splitext(os.path.basename(audio_path))[0]
            text_emotion = text_classifier(input=text)[0]['label']
            mapped_emotion = EMOTION_MAPPING.get(text_emotion, text_emotion)
            f_out.write(f"{line.strip()}|{mapped_emotion}\n")

def main(args):
    emotion_recognizer = EmotionRecognitionPipeline(model_revision=args.model_revision)
    process_audio_files(args.folder_path, args.output_file, emotion_recognizer, args.batch_size, args.max_workers)

    text_classifier = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-large', model_revision='v1.0.0')
    text_emotion_file = f"{os.path.splitext(args.output_file)[0]}_with_text_emotion.csv"
    process_text_emotion(args.output_file, text_emotion_file, text_classifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='识别音频文件中的情感')
    parser.add_argument('--folder_path', type=str, required=True, help='包含音频文件的文件夹路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件的路径')
    parser.add_argument('--model_revision', type=str, default="v2.0.4", help='情感识别模型的修订版本')
    parser.add_argument('--batch_size', type=int, default=10, help='推理的批量大小')
    parser.add_argument('--max_workers', type=int, default=4, help='并行处理的最大工作线程数')
    args = parser.parse_args()
    main(args)
