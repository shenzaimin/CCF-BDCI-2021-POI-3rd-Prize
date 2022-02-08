from itertools import permutations
from fuzzywuzzy import process, fuzz
import gc
import numpy as np
import cv2


def get_permutations(text, limit=3):
    """
    获取排列数目小于等于limit元素的全排列
    :param text:
    :param limit: 排列数目上限
    :return:
    """
    gc.collect()
    str_list = text.split("|")
    str_list = list(set(str_list))
    combine_str_list = []
    length = len(str_list)
    for i in range(length):
        if i > limit:
            continue
        permute_list = list(permutations(str_list, i+1))
        for sub_list in permute_list:
            combine_str = "".join(sub_list)
            combine_str_list.append(combine_str)
    return combine_str_list


def choose_best_match(target_str, text_str, k=1, p=3):
    """

    :param target_str:
    :param text_str:
    :param k: 模糊匹配选取结果数目
    :param p: 全排列的上限
    :return:
    """
    comb_str_list = get_permutations(text_str, limit=p)
    comb_str_list = [string for string in comb_str_list if abs(len(target_str)-len(string)) <= 2]
    if len(comb_str_list) == 0:
        return target_str
    # print(len(combine_str_list))

    return process.extractBests(target_str, comb_str_list, limit=k)[0][0]

def get_inside(text, label):
    gc.collect()
    str_list = text.split("|")
    str_list = list(set(str_list))
    ratio_str_list = []
    length = len(str_list)
    limit = 3
    # 筛除无关候选框
    for i in range(length):
        if fuzz.token_sort_ratio(label, str_list[i]) == 100:
            ratio_str_list.append(str_list[i])
            break
        if fuzz.token_sort_ratio(label, str_list[i]) != 0:
            ratio_str_list.append(str_list[i])
    # 全排列候选框
    combine_str_list = []
    combine_str_dict = {}
    if len(ratio_str_list) == 0:
        pred_name = label
        combine_str_dict[pred_name] = 0
        return pred_name
    for i in range(len(ratio_str_list)):
        if i > limit:
            continue
        permute_list = list(permutations(ratio_str_list, i+1))
        for sub_list in permute_list:
            combine_str = "".join(sub_list)
            combine_str_dict[combine_str] = i+1
            combine_str_list.append(combine_str)
    # 得到框数
    pred_name = process.extractOne(label, combine_str_list, scorer=fuzz.ratio)[0]
    return pred_name
    
    
def contour_num_used(inp_str, tar_str):
    inp_str = inp_str.split("|")
    num = 0
    for s in inp_str:
        if s in tar_str:
            num += 1
    return num
    
def caculate_area(contour, text):
    contour = np.array(contour)
    area = cv2.contourArea(contour, oriented=False)
    area = np.divide(area, len(text))

    return area
    
"""
    展示一张图片
"""


def just_for_show(img_path, contour):
    image = cv2.imread(img_path)
    polygon = np.array(contour, np.int32)
    cv2.polylines(image, [polygon], True, (0, 255, 255))
    cv2.imshow('show a image:', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
    求最小外接矩形,返回中心点和外接矩形的四个角点
"""


def min_rect(board_contour):
    board_contour = np.array(board_contour)
    rect = cv2.minAreaRect(board_contour)  # 最小外接矩形
    box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
    centor = rect[0]
    return centor, box


"""
    按照面积排序，准备样本
"""
# def caculate_area(img_path, countor, text):
#     image = cv2.imread(img_path)
#     polygon = np.array(countor, dtype=np.int32)[np.newaxis, ...]  # 必须多扩充一维
#     im = np.zeros(image.shape[:2], dtype='uint8')
#     polygon_mask = cv2.fillPoly(im, polygon, 255)
#
#     area = np.divide(np.sum(np.greater(polygon_mask, 0)), len(text))  # 用外接多边形的面积除以字数
#
#     return area


def caculate_area(contour, text):
    contour = np.array(contour)
    area = cv2.contourArea(contour, oriented=False)
    area = np.divide(area, len(text))

    return area


"""
    输入一条线段的两个端点，和一个直线外的点，输出点到直线的距离
"""


def get_distance_from_point_to_line(point, line_point1, line_point2):
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]

    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))  # 根据点到直线的距离公式计算距离
    return distance


# def point_distance_line(point, line_point1, line_point2):
#     vec1 = line_point1 - point
#     vec2 = line_point2 - point
#     distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1-line_point2)
#     return distance


"""
    计算两点之间的距离
"""


def distance(point1, point2):
    return np.linalg.norm(point1-point2, ord=2)


"""
    输入数据中的大框和小框，返回小框在大框中的位置编号。
    通过对大框切分bins计算得出
"""


def bins_tokenizer(contour, board_contour, num_of_bins):
    """
    :param contour: 小框的多边形
    :param board_contour: 大框多边形
    :param num_of_bins: 分箱数量
    :return: x方向编码，y方向编码
    """
    _, board_box = min_rect(board_contour)
    p1, p2, p3, p4 = board_box  # p1,p2,p3,p4是大框的四个角点

    centor, _ = min_rect(contour)  # centor是小框的中心点

    x_distance = get_distance_from_point_to_line(centor, p1, p4)  # x方向距离为 centor到矩形“左边”的距离
    y_distance = get_distance_from_point_to_line(centor, p1, p2)  # y方向距离为 centor到矩形“上边”的距离
    x_width = distance(p1, p2)  # x方向上大框的宽
    y_height = distance(p1, p4)  # y方向上大框的高
    x_pos = np.floor((x_distance / x_width)*num_of_bins)  # 计算centor到左边距离占左边高度的百分比，乘以分箱数量，得到的就是分到第几个箱子
    y_pos = np.floor((y_distance / y_height)*num_of_bins)  # 计算centor到上边距离占上边宽度的百分比，乘以分箱数量，得到的就是分到第几个箱子

    return int(x_pos), int(y_pos)





def get_angles(pos, i, d_model):
    """
    计算位置编码中，相应位置上的angles
    :param pos: 一句话中第pos个字，即位置
    :param i: 每个字的embedding 第i个元素
    :param d_model: embedding的维数
    :return: pos上第i个位置的值
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# 输入一串pos_token编码，输出该编码映射到d_model维空间的编码矩阵。
def pos_embedding(pos_tokens, d_model):
    angle_rads = []
    for pos in pos_tokens:
        angle_rad = get_angles(pos, np.arange(d_model), d_model)  # [np.newaxis, :]
        angle_rads.append(angle_rad)

    angle_rads = np.array(angle_rads)[np.newaxis, ...]

    return angle_rads


"""
    输入一个列表和一个数字，输出列表按照这个数字切分后的小列表
"""


def split_list_by_num(lst, num):
    i = 0
    j = 0
    return_lst = []
    while i < len(lst):
        if lst[i] == num:
            return_lst.append(lst[j:i])
            j = i+1
        i += 1
    return_lst.append(lst[j:i])
    return return_lst




if __name__ == '__main__':
    input_text = "万达不锈钢门窗|广州铝材厂|服务电话|更可|拉闸门|钢结构|137272278|老品牌|防盗门|阳光房|纱窗|扶手|不锈钢|吊轨门|1947|推拉门窗|since|三轨门窗|编号|平开门窗|木纹门窗|qz7001|铝|有限公司"
    target = "诚达不锈钢门窗"
    # start_time = time.time()
    for i in range(1):
        best_match = choose_best_match(target, input_text, k=1, p=3)
    # end_time = time.time()
    # print(end_time-start_time)
    print(best_match)

    # # 预测数据
    # perdict_label = []
    # with open('perdict_v2.txt', 'r', encoding='utf-8') as file_perdict:
    #     lines = file_perdict.readlines()
    #     for line in lines:
    #         perdict_label.append(line)

    # # 测试集数据
    # test_input = []
    # with open('data_test_txt.txt', 'r', encoding='utf-8') as file_test:
    #     lines = file_test.readlines()
    #     for line in lines:
    #         test_input.append(line)

    # 训练集数据
    # train_input = []
    # with open('data_txt.txt', 'r', encoding='utf-8') as file_train:
    #     lines = file_train.readlines()
    #     for line in lines:
    #         train_input.append(line)

    # result_dict = {}
    # for i in range(len(test_input)):
    #     img_id, input_seq = test_input[i].split('\t')
    #     pred_img_id, target = perdict_label[i].split('\t')
    #     input_seq = input_seq.replace('\n', '')
    #     target = target.replace('\n', '')
    #     pred_name = choose_best_match(target, input_seq, k=1, p=3)
    #     result_dict[img_id] = pred_name
    
    # result_dict = {}
    # for i in range(len(train_input)):
    #     input_seq, target = train_input[i].split('\t')
    #     input_seq = input_seq.replace('\n', '')
    #     target = target.replace('\n', '')
    #     print(target)
    #     pred_name = choose_best_match(target, input_seq, k=1, p=4)
    #     result_dict[target] = pred_name

    # with open('rematch_train_label.txt', 'w', encoding='utf-8') as pr:
    #     for i in result_dict:
    #         pr.write(i + '\t' + result_dict[i] + '\n' )