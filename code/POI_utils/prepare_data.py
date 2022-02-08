#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/12 10:00
# @Author  : Sunris
# @File    : prepare_data.py
import json
import cv2
import numpy as np
from tqdm import tqdm

from utils_2 import caculate_area, contour_num_used, bins_tokenizer


"""
    准备训练集，按照外界多边形排序
"""
with open('./raw_data/train_label_public.json', encoding='utf-8') as f:
    txts = json.load(f)
# print(txts)
# print(type(txts))

print("=========准备sorted训练集=========")
print()
train_dict = {}
file_out_train_1 = open('./user_data/processed_data/data_txt_sorted_train_A.txt', 'w+', encoding='utf-8')
file_out_train_2 = open('./user_data/processed_data/data_txt_sorted_train_A_aug.txt', 'w+', encoding='utf-8')
for img_id in tqdm(txts["data"]):
    # img_path = 'data_images/desensitized_train_images/' + txts["data"][img_id]["image_id"] + '.jpg'
    # print(img_path)
    text_and_area = []  # 用来储存小框对应的面积

    for text in txts["data"][img_id]["texts"]:
        # text储存二元组[text, area]
        text_and_area.append([text["text"], caculate_area(text["contour"], text["text"])])

    text_and_area.sort(key=lambda x: x[1], reverse=True)
    input_seq = [seq for seq, area in text_and_area]
    input_seq = '|'.join(input_seq)  # 输入的句子
    label_name = txts["data"][img_id]["name"]  # 输出的标签
    use_num = contour_num_used(input_seq, label_name)  # 判断使用的小框个数，如果小框个数大于3就做数据增强
    write_line = txts["data"][img_id]["image_id"] + '\t' + input_seq + '\t' + label_name + '\n'
    file_out_train_1.write(write_line)
    if use_num < 3:
        write_line_aug = txts["data"][img_id]["image_id"] + '\t' + input_seq + '\t' + label_name + '\n'
    else:
        write_line_aug = txts["data"][img_id]["image_id"] + '\t' + input_seq + '\t' + label_name + '\n' \
                    + txts["data"][img_id]["image_id"] + '\t' + input_seq + '\t' + label_name + '\n'
    file_out_train_2.write(write_line_aug)

# with open('data_processed/data_txt_sorted_pos_token.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_dict, json_file, indent=5)

"""
    准备A榜测试，按照外界多边形排序
"""
with open('./raw_data/TestA_Preporcess_public.json', 'r', encoding='utf-8') as file:
    txts_test = json.load(file)
    # print(txts_test)
    # print(type(txts_test))

print("=========准备A榜测试集=========")
print()
test_dict = {}
file_out_test_A = open('./user_data/processed_data/data_txt_sorted_test_A.txt', 'w+', encoding='utf-8')
for img_id in tqdm(txts_test["data"]):
    input_seq = []
    # img_path = './raw_data/desensitized_TestA_images/' + txts_test["data"][img_id]["image_id"] + '.jpg'
    # print(img_path)
    text_and_area = []
    board_contour = txts_test["data"][img_id]["board_contour"]

    for text in txts_test["data"][img_id]["texts"]:
        x_pos, y_pos = bins_tokenizer(text["contour"], board_contour, 100)
        pos_tokens_x = [x_pos] * len(text["text"])
        pos_tokens_y = [y_pos] * len(text["text"])
        # text储存四元组[text, area, pos_token_x, pos_token_y]
        text_and_area.append([text["text"], caculate_area(text["contour"], text["text"]), pos_tokens_x, pos_tokens_y])

    input_seq = [seq for seq, area, pos_x, pos_y in text_and_area]
    input_seq = '|'.join(input_seq)  # 输入的句子
    input_pos = [[pos_x, pos_y] for seq, area, pos_x, pos_y in text_and_area]
    input_pos_x = []
    input_pos_y = []
    for x, y in input_pos:
        input_pos_x += x
        input_pos_x += [-9999]
        input_pos_y += y
        input_pos_y += [-9999]
    input_pos_x = input_pos_x[:-1]
    input_pos_y = input_pos_y[:-1]

    # print('input seq:', input_seq, len(input_seq))
    # print('label name:', label_name, len(label_name))
    # print('input pos x:', input_pos_x[:-1], len(input_pos_x[:-1]))
    # print('input pos y:', input_pos_y[:-1], len(input_pos_y[:-1]))

    test_dict[img_id] = {"input_seq": input_seq,
                         "input_pos_x": input_pos_x,
                         "input_pos_y": input_pos_y}
    write_line = img_id + '\t' + input_seq + '\t' + str(input_pos_x) + '\t' + str(input_pos_y) + '\n'
    file_out_test_A.write(write_line)
#
# with open('data_processed/data_test_txt_sorted_pos_token.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_dict, json_file, indent=5)


"""
    准备B榜测试，按照外界多边形排序
"""
with open('./raw_data/public_TestB.json', 'r', encoding='utf-8') as file:
    txts_test = json.load(file)
    # print(txts_test)
    # print(type(txts_test))

print("=========准备B榜测试集=========")
print()
test_dict = {}
file_out_test_B = open('./user_data/processed_data/data_txt_sorted_test_B.txt', 'w+', encoding='utf-8')
for img_id in tqdm(txts_test["data"]):
    input_seq = []
    # img_path = 'data_images/desensitized_TestA_images/' + txts_test["data"][img_id]["image_id"] + '.jpg'
    # print(img_path)
    text_and_area = []
    board_contour = txts_test["data"][img_id]["board_contour"]

    for text in txts_test["data"][img_id]["texts"]:
        x_pos, y_pos = bins_tokenizer(text["contour"], board_contour, 100)
        pos_tokens_x = [x_pos] * len(text["text"])
        pos_tokens_y = [y_pos] * len(text["text"])
        # text储存四元组[text, area, pos_token_x, pos_token_y]
        text_and_area.append([text["text"], caculate_area(text["contour"], text["text"]), pos_tokens_x, pos_tokens_y])
    text_and_area.sort(key=lambda x: x[1], reverse=True)
    input_seq = [seq for seq, area, pos_x, pos_y in text_and_area]
    input_seq = '|'.join(input_seq)  # 输入的句子
    input_pos = [[pos_x, pos_y] for seq, area, pos_x, pos_y in text_and_area]
    input_pos_x = []
    input_pos_y = []
    for x, y in input_pos:
        input_pos_x += x
        input_pos_x += [-9999]
        input_pos_y += y
        input_pos_y += [-9999]
    input_pos_x = input_pos_x[:-1]
    input_pos_y = input_pos_y[:-1]

    # print('input seq:', input_seq, len(input_seq))
    # print('label name:', label_name, len(label_name))
    # print('input pos x:', input_pos_x[:-1], len(input_pos_x[:-1]))
    # print('input pos y:', input_pos_y[:-1], len(input_pos_y[:-1]))

    test_dict[img_id] = {"input_seq": input_seq,
                         "input_pos_x": input_pos_x,
                         "input_pos_y": input_pos_y}
    write_line = img_id + '\t' + input_seq + '\n'
    file_out_test_B.write(write_line)

# with open('data_processed/data_test_txt_sorted_pos_token.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_dict, json_file, indent=5)