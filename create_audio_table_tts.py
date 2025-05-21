import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import pandas as pd
import os

import random
from glob import glob
from tqdm import tqdm

import shutil


import numpy as np
# set seed to ensures reproducibility
def set_seed(random_seed=1234):
    # set deterministic inference
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)

set_seed()



# samples_path = "/home/ecasanova/Projects/Papers/ICASSP-2025-21Hz-codec/NeMo-Speech-Codec/audios_demo/T5-TTS_22kHz/"
samples_path = "/home/ecasanova/Projects/Papers/INTERSPEECH-Codec/Interspeech25TTSEvals/Interspeech25Evals_with_RTF_norm/"
output_audio_path = "/home/ecasanova/Projects/Papers/INTERSPEECH-Codec/NanoCodec/audios_demo/ZS-TTS/"

num_samples_per_exp = 20
extension = ".wav"



def create_html_table(dic):


    
    df = pd.DataFrame.from_dict(dic, orient='columns')
    # print(df)
    # df = pd.concat([pd.DataFrame.from_dict(aux_list, orient='columns'), df], ignore_index=True)
    # df = df.sort_values('Model Name')
    
    # html = df.pivot_table(values=['generated_wav'], index=["Model Name"], columns=['Speaker Name'], aggfunc='sum').to_html()

    html = df.pivot_table(values=['Samples'], index=["Codec"], columns=['Sample ID'], aggfunc='sum').to_html()

    # added audio 
    html = html.replace("<td>audios_demo/", '<td><audio controls style="width: 110px;" src="audios_demo/')
    html = html.replace(extension+"</td>", extension+'"></audio></td>')
    # remove begging name order id
    for key in model_map:
        exp_name = model_map[key]
        html = html.replace(exp_name, exp_name[2:])

    html  = html.replace("@", "")

    print(html)









from glob import glob


all_samples = glob(samples_path + '**/*'+extension, recursive=True)
all_samples.sort()
random.shuffle(all_samples)


model_map = {
    "GT": "0 Ground truth",
    "Reference": "1 Reference",
    "ICML_CFG_DC_koel_onlyphoneme_epoch161_Temp0.6_Topk80_Cfg_True_2.5_svmodel_wavlm_libri_unseen_edresson_phoneme": "2 LFSC 12.5 FPS 1.89 kbps",
    "21fps_causal_epoch203_Temp0.6_Topk80_Cfg_True_2.5_svmodel_wavlm_libri_unseen_edresson_phoneme": "3 NanoCodec 21.5 FPS 1.89 kbps",
    "12.5codec_epoch248_Temp0.6_Topk80_Cfg_True_2.5_svmodel_wavlm_libri_unseen_edresson_phoneme": "4 NanoCodec 12.5 FPS 1.78 kbps",
    "12.5codec_epoch154_1.1kbps_8_codebooks_Temp0.6_Topk80_Cfg_True_2.5_svmodel_wavlm_libri_unseen_edresson_phoneme": "5 NanoCodec 12.5 FPS 1.1 kbps",
    "koel_12.5_FPS_causal_13codebooks_context10s_epoch_129_Temp0.6_Topk80_Cfg_True_2.5_svmodel_wavlm_libri_unseen_edresson_phoneme": "6 NanoCodec 12.5 FPS 1.78 kbps 10s context"
}


final_samples = []
sample_id = 0
sample_names = []
for sample in all_samples:

    sample_name = sample.split("/")[-1]
    if sample_name in sample_names:
        continue

    if "predicted" not in sample_name: # consider only tts samples
        continue

    if sample_id > num_samples_per_exp:
        continue
    
    cur_sample_model_name = os.path.basename(sample.split("/audio/")[-2])
    print(cur_sample_model_name)

    for model_name in model_map:
        
        if model_name == "GT":
            audio_path = sample.replace("predicted_", "target_")
            new_audio_path = audio_path.replace(samples_path, output_audio_path).replace(cur_sample_model_name, model_name)
        elif model_name == "Reference":
            audio_path = sample.replace("predicted_", "context_")
            new_audio_path = audio_path.replace(samples_path, output_audio_path).replace(cur_sample_model_name, model_name)
        else:
            audio_path = sample.replace(cur_sample_model_name, model_name)
            new_audio_path = audio_path.replace(samples_path, output_audio_path)

        os.makedirs(os.path.dirname(new_audio_path),exist_ok=True)
        print("Copying file:", audio_path, new_audio_path)
        shutil.copyfile(audio_path, new_audio_path)

        new_audio_path_local_path = "audios_demo/" + new_audio_path.split("audios_demo/")[-1]
        print(new_audio_path_local_path)
        dic = {"Codec": model_map[model_name], "Samples": new_audio_path_local_path, "Sample ID": sample_id}
        # if "GT" not in model_name and "Reference" not in model_name:
        #     exit()
        final_samples.append(dic)

    sample_id += 1
    sample_names.append(sample_name)



create_html_table(final_samples)
