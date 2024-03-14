import os
import csv
import shutil
from pydub import AudioSegment

def classify_audio_emotion(log_file, dataset_path):
    character_emotions = {}

    with open(log_file, 'r', encoding='utf-8') as file:
        next(csv.reader(file, delimiter='|'))  # 跳过标题行
        for audio_path, character, emotion in csv.reader(file, delimiter='|'):
            character_emotions.setdefault(character, {}).setdefault(emotion, []).append(audio_path)

    for character, emotions in character_emotions.items():
        character_folder = os.path.join(dataset_path, character)
        os.makedirs(character_folder, exist_ok=True)

        for emotion, audio_files in emotions.items():
            emotion_folder = os.path.join(character_folder, emotion)
            os.makedirs(emotion_folder, exist_ok=True)

            for audio_file in audio_files:
                src_path = os.path.join(dataset_path, audio_file)
                audio_name = os.path.basename(audio_file)
                new_audio_name = f"【{emotion}】{audio_name}"
                dst_path = os.path.join(emotion_folder, new_audio_name)

                if os.path.exists(src_path):
                    duration = AudioSegment.from_wav(src_path).duration_seconds
                    if 3 <= duration <= 10:
                        shutil.move(src_path, dst_path)
                        print(f"Moved {audio_file} to {dst_path}")
                    else:
                        os.remove(src_path)
                        print(f"Deleted {audio_file} (duration: {duration:.2f}s)")
                else:
                    print(f"File not found: {src_path}")

log_file = r''
dataset_path = r''
classify_audio_emotion(log_file, dataset_path)
