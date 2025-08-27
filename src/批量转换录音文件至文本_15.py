
import os
import torch
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import List, Dict

def split_audio(audio_array: List[float], sample_rate: int, chunk_length: int = 30) -> List[List[float]]:
    """
    Split long audio into smaller chunks of fixed length.
    """
    chunk_samples = chunk_length * sample_rate
    total_samples = len(audio_array)
    return [audio_array[i:i + chunk_samples] for i in range(0, total_samples, chunk_samples)]

def transcribe_audio_with_timestamps(model, processor, audio_path, output_path, device, chunk_length: int = 30):
    """
    Transcribe long audio files with timestamps.
    """
    if os.path.exists(output_path):
        print(f"Skipping existing file: {output_path}")
        return

    print(f"Processing: {audio_path}")
    speech_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_chunks = split_audio(speech_array, sr, chunk_length)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    current_time = 0.0
    results = []

    for chunk in tqdm(audio_chunks, desc="Processing chunks", unit="chunk"):
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        chunk_duration = len(chunk) / sr
        start_time = current_time
        end_time = current_time + chunk_duration
        current_time = end_time

        results.append({
            "start": f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02}",
            "end": f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02}",
            "text": transcription.strip()
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"[{result['start']} - {result['end']}] {result['text']}\n")

def batch_transcribe(model_name_or_path, input_dir, output_dir, chunk_length: int = 30):
    """
    Batch process audio files in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    processor = WhisperProcessor.from_pretrained(model_name_or_path)

    audio_files = [os.path.join(root, file) for root, _, files in os.walk(input_dir)
                   for file in files if file.lower().endswith(('.wav', '.mp3', '.m4a'))]

    for audio_file in audio_files:
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_file))[0] + "_transcription.txt")
        transcribe_audio_with_timestamps(model, processor, audio_file, output_file, device, chunk_length)

if __name__ == "__main__":
    model_path = r"E:/笔记/ai模型/whisper-small"
    input_directory = r"E:/运营商IT系统/工作文档/o域/政务云二期/过程文档"
    output_directory = r"E:/运营商IT系统/工作文档/o域/政务云二期/过程文档/转写结果"

    batch_transcribe(model_path, input_directory, output_directory)
