import os
import glob
from shutil import copyfile
import pandas as pd
import random
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.metrics.wer import word_error_rate

random.seed(26)


import jiwer
import jiwer.transforms as tr
from packaging import version
import importlib.metadata as importlib_metadata
import random
import transformers
import torch
import torchaudio
import numpy as np

def set_seed(random_seed=26):
    # set deterministic inference
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    transformers.set_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


set_seed()


def return_existing_files(files, Exps_maps):
    new_files = []
    for f in files:
        exist = True
        out_gt_path = os.path.join(output_dir, os.path.basename(f).replace(ext, "_gt"+ext))
        # check GT file
        if not os.path.isfile(f):
            continue

        for key in Exps_maps:
            exp_path = Exps_maps[key]
            f_exp = f.replace(GT_dir, exp_path)
            if not os.path.isfile(f_exp):
                exist = False
        if exist:
            new_files.append(f)
    return new_files


output_dir = "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/Comparision_demo/mls/"

# output_dir = "/home/ecasanova/Projects/Papers/INTERSPEECH-Codec/NanoCodec/audios_demo/ZS-TTS/"

num_files_per_language = 6

ext = ".flac"
between_both_models = True
worst_cer = True

Exps_maps = {
    "1.89kbps-Low-Frame-rate-Speech-Codec": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/Low_Frame-rate_Speech_Codec_2k_codes_21Hz/",
    "1.78kbps-Ours": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/ml-model-INTERSPEECH_2025_abblations_12.5Hz_13_codebooks_2016_codes_enc_non_causal_dec_causa_1.78kbps/",
    "1.1kbps-Mimi": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/Mimi-8-codebooks/",
    "1.1kbps-Ours": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/ml-model-INTERSPEECH_2025_abblations_12.5Hz_8_codebooks_2016_codes_pad_fix_enc_non_causal_dec_causal_1.1kbps/",
    "1.1kbps-Ours-25-FPS": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/ml-model-INTERSPEECH_2025_abblations_25Hz_4_codebooks_2016_codes_pad_fix_enc_non_causal_dec_causal/",
    "1.1kbps-Ours-6.25-FPS": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/ml-model-INTERSPEECH_2025_abblations_6.25Hz_16_codebooks_2016_codes_pad_fix_enc_non_causal_dec_causal/",
    "0.9kbps-WavTokenizer": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/wavtokenizer_large_speech_320_24khz_75FPS/",
    "0.85kbps-TS3-Codec": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/TS3-Codec-X2-0.85kbps/",
    "0.8kbps-Ours": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/ml-model-INTERSPEECH_2025_abblations_12.5Hz_4_codebooks_65.536_codes_enc_non_causal_dec_causal_0.8kbps/",
    "0.7kbps-TAAE": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/StableCodec_0.7kbps/",
    "0.6kbps-Ours": "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/MLS_44kHz/ml-model-INTERSPEECH_2025_abblations_12.5Hz_4_codebooks_4032_codes_enc_non_causal_dec_causal_0.6kbps/",
}



os.makedirs(output_dir, exist_ok=True)

# get ranking from the first
first_exp_path = Exps_maps[list(Exps_maps.keys())[0]]
wer_file_path = os.path.join(first_exp_path, "metrics/wer.json")
items = read_manifest(wer_file_path)[:-1]


df = pd.DataFrame(items)

samples = df.sample(frac=1)

samples_languages = {}
for i, sample in samples.iterrows():
    GT_sample = sample["target_audio"]

    if "MLS_44kHz" in GT_sample:
        language = GT_sample.split("mls_")[1].split("/")[0]
    else:
        language = "en"

    out_gt_path = os.path.join(output_dir, language, os.path.basename(GT_sample).replace(ext, "_GT"+ext))

    ref_audio_dirname = os.path.dirname(sample["pred_audio"])


    if language in samples_languages:
        # print(language, len(samples_languages[language]))
        if len(samples_languages[language]) >= num_files_per_language:
            continue

        speaker_in_set = False
        for s in samples_languages[language]:
            speaker = os.path.basename(s).split("_")[0]
            if speaker in GT_sample:
                speaker_in_set = True
        if speaker_in_set:
            continue

        samples_languages[language].append(GT_sample)
    else:
        samples_languages[language] = [GT_sample]

    os.makedirs(os.path.dirname(out_gt_path), exist_ok=True)
    # copy GT file
    copyfile(GT_sample, out_gt_path)
    for key in Exps_maps.keys():
        exp_path = Exps_maps[key]
        f_exp = sample["pred_audio"].replace(first_exp_path, exp_path)
        out_exp_path = os.path.join(output_dir, language, os.path.basename(f_exp).replace(ext, "_"+key+ext))
        os.makedirs(os.path.dirname(out_exp_path), exist_ok=True)
        copyfile(f_exp, out_exp_path)

print("Samples per each  CML language:")
for lang in samples_languages:
    print(lang, len(samples_languages[lang]))
print("Audios saved at: ", output_dir)

ext = ".wav"
# create samples for DAPS
output_dir = "/home/ecasanova/Projects/Evaluations/MMS_ASR_interspeech_paper_eval_16khz/Comparision_demo/daps/"


for exp in Exps_maps:
    Exps_maps[exp] = Exps_maps[exp].replace("MLS_44kHz", "daps_f10_m10_spks")

# print(Exps_maps)
os.makedirs(output_dir, exist_ok=True)

# get ranking from the first
first_exp_path = Exps_maps[list(Exps_maps.keys())[0]]
wer_file_path = os.path.join(first_exp_path, "metrics/wer.json")
items = read_manifest(wer_file_path)[:-1]


df = pd.DataFrame(items)

samples = df.sample(frac=1)
samples = samples[:num_files_per_language]

for i, sample in samples.iterrows():
    GT_sample = sample["target_audio"]

    out_gt_path = os.path.join(output_dir, os.path.basename(GT_sample).replace(ext, "_GT"+ext))

    ref_audio_dirname = os.path.dirname(sample["pred_audio"])
    os.makedirs(os.path.dirname(out_gt_path), exist_ok=True)
    # copy GT file
    copyfile(GT_sample, out_gt_path)
    for key in Exps_maps.keys():
        exp_path = Exps_maps[key]
        f_exp = sample["pred_audio"].replace(first_exp_path, exp_path)
        out_exp_path = os.path.join(output_dir, os.path.basename(f_exp).replace(ext, "_"+key+ext))
        os.makedirs(os.path.dirname(out_exp_path), exist_ok=True)
        copyfile(f_exp, out_exp_path)


print("Audios saved at: ", output_dir)




