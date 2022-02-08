# -*- coding: utf-8 -*-
# @Time     : 2021/10/8 13:37
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com

import logging
import sys
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import os
import json
from tqdm import tqdm
from POI_utils.utils import choose_best_match
import gc
from fuzzywuzzy import process

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data_original = []

# TODO 读取数据集
print("使用wx排序的input")
sorted_text_infos = open("./user_data/processed_data/data_txt_sorted_train_A.txt", 'r', encoding='utf-8').readlines()
# print("使用wx排序并且aug后的input")
# sorted_text_infos = open("../data/POI/data_txt_aug.txt", 'r', encoding='utf-8').readlines()

for line in tqdm(sorted_text_infos):
    idx, inp, label = line.strip('\n').split('\t')
    sample = ["排序", inp, label]
    train_data_original.append((idx, sample))

# json_data_train = json.load(open("../data/POI/train_label_public.json", 'r', encoding='utf-8'))
# poi_info = json_data_train["data"]
# bad_num = 0
# TODO 匹配拼接
# print("使用匹配拼接")
# for block_id, block_info in tqdm(poi_info.items()):
#     img_id = block_info["image_id"]
#     label = block_info["name"]
#     text_infos = block_info["texts"]
#     text_list = list(set([text_info["text"] for text_info in text_infos]))
#     text_list_filter = [text for text in text_list if text in label]
#     text_list_fuzzy_filter = [match[0] for match in process.extractBests(label, text_list, limit=3)]
#     if len("".join(text_list_filter)) == len(label) or label in text_list_filter:
#         text_list_sorted = sorted(text_list_filter, reverse=True)
#     else:
#         text_list_sorted = sorted(text_list_fuzzy_filter, reverse=True)
#         print(f"block_id: {block_id}\n input: {text_list}\n label: {label}\n fuzzy match input: {text_list_fuzzy_filter}")
#         bad_num += 1
#
#     input_text = " | ".join(text_list_sorted)
#     sample = ["排序", input_text, label]
#     train_data_original.append((block_id, sample))

# print(f"Bad Num: {bad_num}")

# TODO 完全拼接
# print("使用完全拼接")
# for block_id, block_info in tqdm(poi_info.items()):
#     # img_id = block_info["image_id"]
#     label = block_info["name"]
#     text_infos = block_info["texts"]
#     text_list = list(set([text_info["text"] for text_info in text_infos]))
#     text_list_sorted = sorted(text_list, reverse=True)
#     input_text = " | ".join(text_list_sorted)
#     sample = ["排序", input_text, label]
#     train_data_original.append((block_id, sample))


import random
random.seed(2021)
random.shuffle(train_data_original)

# TODO 使用80%训练集模型
# print("使用80%训练数据集训练！")
# train_data = train_data_original[:int(0.8*len(train_data_original))]
# TODO 用全量数据集
print("使用全量数据集训练！")
train_data = train_data_original

train_data_original_blk_id = [t[0] for t in train_data]
train_data_original_text = [t[1] for t in train_data]

train_df = pd.DataFrame(train_data_original_text)
train_df.columns = ["prefix", "input_text", "target_text"]

# TODO print input len
input_max_len_train = max([len(t) for t in list(train_df['input_text'])])
target_max_len_train = max([len(t) for t in list(train_df['target_text'])])
print(f"max_len of train_input: {input_max_len_train}")
print(f"max_len of train_target: {target_max_len_train}")

eval_data = train_data_original[int(0.8*len(train_data_original)):]
print(f"EVAL NUM: {len(eval_data)}")
eval_data_original_blk_id = [t[0] for t in eval_data]
eval_data_original_text = [t[1] for t in eval_data]

eval_df = pd.DataFrame(eval_data_original_text)
eval_df.columns = ["prefix", "input_text", "target_text"]

# TODO print input len
input_max_len_eval = max([len(t) for t in list(eval_df['input_text'])])
target_max_len_eval = max([len(t) for t in list(eval_df['target_text'])])
print(f"max_len of eval_input: {input_max_len_eval}")
print(f"max_len of eval_target: {target_max_len_eval}")


# TODO 读取wx排序后的inpt_text
print("读取wx排序后的测试集输入！")
test_data_original = []
block_id_list = []
# sorted_test_text_infos = open("./user_data/processed_data/data_txt_sorted_test_A.txt", 'r', encoding='utf-8').readlines()  # TODO Test A
sorted_test_text_infos = open("./user_data/processed_data/data_txt_sorted_test_B.txt", 'r', encoding='utf-8').readlines()  # TODO Test B
for line in tqdm(sorted_test_text_infos):
    block_id, inp_test = line.strip('\n').split('\t')
    block_id_list.append(block_id)
    test_data_original.append(inp_test)

# TODO 读取测试数据集
# test_data_original = []
# block_id_list = []
# json_data_test = json.load(open("../data/POI/TestA_Preporcess_public.json", 'r', encoding='utf-8'))
# poi_info = json_data_test["data"]
# for block_id, block_info in tqdm(poi_info.items()):
#     text_infos = block_info["texts"]
#     # text_list = list(set([text_info["text"] for text_info in text_infos]))  # TODO 这里使用用去重还是不去重，要实验得出结论
#     text_list = [text_info["text"] for text_info in text_infos]
#     text_list_sorted = sorted(text_list, reverse=True)
#     input_text = " | ".join(text_list_sorted)
#     block_id_list.append(block_id)
#     test_data_original.append(input_text)

# TODO 读取MMBT_filter_result测试数据集
# import pickle
# MMBT_file = "../result/POI/MMBT/MMBT_sub_img-checkpoint-10092-epoch-4_sorted_top_7.pkl"
# print(f"Reading result from MMBT...\n file dir: {MMBT_file}")
# test_data_original = pickle.load(open(MMBT_file, 'rb'))

to_predict = ["排序: " + t for t in test_data_original]

# TODO print input len
input_max_len_test = max([len(t) for t in to_predict])

print(f"max_len of test_input: {input_max_len_test}")
# TODO MMBT Result
# input_max_len_for_t5 = max([input_max_len_train, input_max_len_eval]) + 10

input_max_len_for_t5 = max([input_max_len_train, input_max_len_eval, input_max_len_test]) + 10
generate_max_len_for_t5 = max([target_max_len_train, target_max_len_eval]) + 10
print(f"T5 max input len: {input_max_len_for_t5}")
print(f"T5 max generate len: {generate_max_len_for_t5}")

model_args = T5Args()
cache_dir = "./user_data/cache_dir/mt5-base"
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
model_args.manual_seed = 2021
model_args.config = {"cache_dir": "./user_data/cache_dir/mt5-base"}
model_args.do_sample = False
model_args.train_batch_size = 6
model_args.eval_batch_size = 8
model_args.max_seq_length = input_max_len_for_t5
model_args.max_length = generate_max_len_for_t5
model_args.best_model_dir = "./user_data/saved_model/T5_POI_wx_input_all_data_aug/outputs/best_model"
model_args.output_dir = "./user_data/saved_model/T5_POI_wx_input_all_data_aug/outputs/"
model_args.cache_dir = "./user_data/cache_dir/T5_POI_wx_input_cache"
model_args.evaluate_during_training = True
model_args.fp16 = False
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 300
model_args.evaluate_during_training_verbose = True
model_args.no_save = False
model_args.save_steps = -1

model_args.gradient_accumulation_steps = 1
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.use_multiprocessed_decoding = False


# model_args.save_steps = (model_args.num_train_epochs * 1440) // (model_args.train_batch_size * 20) // model_args.gradient_accumulation_steps
print(f"SAVING STEPS:{model_args.save_steps}")
model_args.evaluate_generated_text = True
model_args.save_model_every_epoch = True
model_args.evaluate_during_training_steps = -1
model_args.save_eval_checkpoints = False
model_args.learning_rate = 1e-4

model_args.repetition_penalty = 1.0
# model_args.length_penalty = 0.6

model_args.scheduler = "polynomial_decay_schedule_with_warmup"
# model_args.scheduler = "constant_schedule_with_warmup"
model_args.use_early_stopping = False
model_args.num_beams = 1
model_args.top_k = 50
model_args.top_p = 0.95

# model = T5Model("mt5", "lemon234071/t5-base-Chinese", args=model_args)
#  TODO Retraining
# print("Continue training")
# model = T5Model("mt5", "../saved_model/T5_POI_wx_input_all_data_aug/outputs/checkpoint-775000-epoch-75", args=model_args)


def count_matches(labels, preds):
    for blk_id, inp, label, pred in tqdm(zip(eval_data_original_blk_id, list(eval_df["input_text"]), labels, preds)):
        # gc.collect()
        # pred = choose_best_match(pred, inp)

        if label == pred:
            print("-" * 106)
            print("*" * 50 + "↓正确↓" + "*" * 50)
            print("-" * 106)
        else:
            print("-"*106)
        print(f"[blk_id]: {blk_id}")
        print(f"[input]: {inp}")
        print(f"[label]: {label}")
        print(f"[pred]: {pred}")
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


# model.train_model(train_df, eval_data=eval_df, matches=count_matches)

# tmp = json.load(open("../result/POI/Cicero0_100902.json", "r", encoding="utf8"))
# json.dump(tmp, open("../result/POI/Cicero0_1009_01_fuzzy_match.json", 'w+', encoding='utf-8'), ensure_ascii=True, indent=4)

# 加载训练好的模型
model_args.eval_batch_size = 200
# TODO 使用全量数据集模型

base_dir = "./user_data/best_models"

ensemble_model_list = [os.path.join(base_dir, file)for file in os.listdir(base_dir)]
print(ensemble_model_list)
# ensemble_model_list = [
#     "checkpoint-121212-epoch-111",
#     "checkpoint-137592-epoch-35",
#     "checkpoint-137124-epoch-34",
#     "checkpoint-136656-epoch-33",
#     "checkpoint-136188-epoch-32",
#     "checkpoint-135720-epoch-31",
#     "checkpoint-135252-epoch-30",
#     "checkpoint-134784-epoch-29",
#     "checkpoint-134316-epoch-28",
#     "checkpoint-133848-epoch-27",
#     "checkpoint-133380-epoch-26"
# ]
for i, checkpoint in enumerate(ensemble_model_list):
    # if i <= 0:
    #     continue
    print(f"使用全量数据集模型{checkpoint}！")
    model = T5Model("mt5", checkpoint, args=model_args)
    # model = T5Model("mt5", "../saved_model/T5_POI_wx_input_all_data_aug/outputs/checkpoint-465000-epoch-36", args=model_args)
    # TODO 使用80%训练集模型
    # print("使用80%训练数据集模型！")
    # model = T5Model("t5", "../saved_model/T5_POI_wx_input/outputs/checkpoint-701200-epoch-91", args=model_args)
    # print(f"EVAL...")
    # print(model.eval_model(eval_df, matches=count_matches))

    print(f"PREDICT...")
    predictions = model.predict(to_predict)


    # # TODO 不做模糊匹配修正
    # print("原始输出...")
    # result_json_file = dict()
    # for block_id, pred in zip(block_id_list, predictions):
    #     result_json_file[block_id] = pred
    # json.dump(result_json_file, open("../result/POI/2021-11-17-01-re_train_wx_input_all_data-checkpoint-465000-epoch-36.json", 'w+', encoding='utf-8'), ensure_ascii=True, indent=4)
    # import json
    # result_json_file = json.load(open("../result/POI/ensemble_ten_files_big_than_71_base_29-03_before_fuzzy.json", "r", encoding="utf-8"))
    # predictions = [result_json_file[blk_id] for blk_id in block_id_list]
    # TODO 进行模糊匹配修正
    print("模糊匹配修正输出...")
    from POI_utils.utils_2 import get_inside
    result_json_file = dict()
    change_num = 0
    for block_id, pred, inp in tqdm(zip(block_id_list, predictions, test_data_original)):
        gc.collect()
        # pred_fix = choose_best_match(pred, inp, k=1, p=4, diff=2)
        if "·" in pred:
            pred_fix = pred
        else:
            pred_fix = get_inside(inp, pred)
        if pred_fix != pred:
            # print("-*"*50)
            # print(f"blk: {block_id}")
            # print(f"pre: {pred}")
            # print(f"pred_fix: {pred_fix}")
            # print("-*"*50)

            change_num += 1
        result_json_file[block_id] = pred_fix
    print(f"Changed num: {change_num}")
    # json.dump(result_json_file, open("../result/POI/2021-11-17-01-re_train_wx_input_all_data-checkpoint-465000-epoch-36-fuzzy_match-top_p_4.json", 'w+', encoding='utf-8'), ensure_ascii=True, indent=4)
    json.dump(result_json_file, open(f"./user_data/result_single_models/result-{i}.json", 'w+', encoding='utf-8'), ensure_ascii=True, indent=4)

# TODO Ensemble
best_single_model_result = json.load(open("./user_data/result_single_models/result-7.json", 'r', encoding="utf-8"))
base_dir = "./user_data/result_single_models"
base_file_list = [os.path.join(base_dir, file)for file in os.listdir(base_dir)]
print(base_file_list)
change_info = dict()

result_json_file_ensemble = dict()
for json_file in base_file_list:
    result_json_file = json.load(open(json_file, 'r', encoding='utf-8'))
    for blk_id, poi_name in result_json_file.items():
        result_json_file_ensemble[blk_id] = result_json_file_ensemble.get(blk_id, []) + [poi_name]
    assert len(result_json_file_ensemble) == len(result_json_file)

for blk_id, poi_name_list in result_json_file_ensemble.items():
    best_choice_poi_name = max(poi_name_list, key=poi_name_list.count)
    if poi_name_list.count(best_choice_poi_name) == poi_name_list.count(best_single_model_result[blk_id]):  # 如果投票结果中单模最好结果处于并列第一，则保留单模结果
        best_choice_poi_name = best_single_model_result[blk_id]
    if best_choice_poi_name != best_single_model_result[blk_id]:
        change_info[blk_id] = change_info.get(blk_id, []) + [best_single_model_result[blk_id], best_choice_poi_name]
    result_json_file_ensemble[blk_id] = best_choice_poi_name

print(f"Changed poi name num: {len(change_info)}")

json.dump(result_json_file_ensemble, open("./prediction_result/result.json", "w+", encoding="utf-8"), ensure_ascii=True, indent=4)
