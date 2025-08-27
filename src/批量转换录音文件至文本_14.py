import os
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

def split_audio(audio_array: np.ndarray, sample_rate: int, chunk_length: int = 30) -> List[np.ndarray]:
    """
    将长音频分割成30秒的片段
    """
    chunk_samples = chunk_length * sample_rate
    total_samples = len(audio_array)
    
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio_array[start:end]
        chunks.append(chunk)
    
    return chunks


def transcribe_long_audio(model, processor, audio_path, output_path, device, chunk_length: int = 30):
    """
    转写长音频文件，自动分段处理，并增加时间戳
    """
    if os.path.exists(output_path):
        print(f"跳过已存在结果: {os.path.basename(audio_path)}")
        return

    print(f"正在处理: {os.path.basename(audio_path)}")

    try:
        # 读取音频
        speech_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(speech_array) / sr
        print(f"音频时长: {audio_duration:.2f}秒")

        # 分割音频
        audio_chunks = split_audio(speech_array, sr, chunk_length)
        print(f"分割成 {len(audio_chunks)} 个片段")

        # 中文转写设置
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

        all_segments = []
        current_time = 0.0

        for i, chunk in enumerate(audio_chunks):
            print(f"  处理片段 {i+1}/{len(audio_chunks)}...")

            # 处理当前片段
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 生成文本
            generated_ids = model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                num_beams=5,
                temperature=0.1
            )

            # 解码结果
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 计算时间戳（近似）
            chunk_duration = len(chunk) / sr
            segment_start = current_time
            segment_end = current_time + chunk_duration

            # 格式化时间戳为 HH:MM:SS
            start_time_formatted = f"{int(segment_start // 3600):02}:{int((segment_start % 3600) // 60):02}:{int(segment_start % 60):02}"
            end_time_formatted = f"{int(segment_end // 3600):02}:{int((segment_end % 3600) // 60):02}:{int(segment_end % 60):02}"

            all_segments.append({
                'start': start_time_formatted,
                'end': end_time_formatted,
                'text': transcription.strip()
            })

            current_time = segment_end

        # 保存结果
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"音频文件: {os.path.basename(audio_path)}\n")
            f.write(f"总时长: {audio_duration:.2f}秒\n")
            f.write(f"分段数: {len(audio_chunks)}\n")
            f.write("=" * 50 + "\n\n")

            for seg in all_segments:
                f.write(f"[{seg['start']}-{seg['end']}] {seg['text']}\n")

        print(f"✓ 完成: {os.path.basename(audio_path)}")
        return True

    except Exception as e:
        print(f"✗ 处理失败 {os.path.basename(audio_path)}: {e}")
        return False


def batch_transcribe_long_audio(model_name_or_path, input_path, output_directory, chunk_length: int = 30):
    """
    批量转写长音频文件
    """
    os.makedirs(output_directory, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载模型
    print("正在加载模型...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    print("模型加载完成")

    # 收集音频文件
    audio_files = []
    supported_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".m4a")
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(supported_extensions):
            audio_files.append(input_path)
    else:
        for root, dirs, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(supported_extensions):
                    audio_files.append(os.path.join(root, f))

    if not audio_files:
        print("未找到音频文件")
        return

    print(f"找到 {len(audio_files)} 个音频文件")

    success_count = 0
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] ", end="")
        output_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_transcribed.txt"
        output_path = os.path.join(output_directory, output_filename)
        
        if transcribe_long_audio(model, processor, audio_path, output_path, device, chunk_length):
            success_count += 1

    print(f"\n处理完成! 成功: {success_count}/{len(audio_files)}")

# 使用示例
if __name__ == "__main__":
    model_path = "openai/whisper-small"
    audio_input = r"E:/运营商IT系统/工作文档/o域/政务云二期/过程文档"
    output_dir = r"E:/运营商IT系统/工作文档/o域/政务云二期/过程文档/转写结果"
    
    batch_transcribe_long_audio(model_path, audio_input, output_dir)
