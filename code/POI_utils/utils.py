from itertools import permutations
from fuzzywuzzy import process
import time
import gc


def get_permutations(text, limit=3):
    """
    获取排列数目小于等于limit元素的全排列
    :param text:
    :param limit: 排列数目上限
    :return:
    """
    gc.collect()
    if " | " in text:
        str_list = text.split(" | ")
    else:
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


def choose_best_match(target_str, text_str, k=1, p=3, diff=2):
    """

    :param target_str:
    :param text_str:
    :param k: 模糊匹配选取结果数目
    :param p: 全排列的上限
    :param diff: 长度差异限制
    :return:
    """
    comb_str_list = get_permutations(text_str, limit=p)
    comb_str_list_filter = [string for string in comb_str_list if abs(len(target_str)-len(string)) <= diff]
    if len(comb_str_list_filter) == 0:
        return process.extractBests(target_str, comb_str_list, limit=k)[0][0]
        #return process.extractBests(target_str, get_permutations(text_str, limit=1), limit=k)[0][0]

        # return target_str
    # print(len(combine_str_list))

    return process.extractBests(target_str, comb_str_list_filter, limit=k)[0][0]


if __name__ == '__main__':
    input_text = "鲜扎送 | 纯粹精酿 | 精酿啤酒屋新售品 | 优鲜送达 | 优布劳精酿啤酒屋 | 优布劳精酿 | urbrew | 33331田"
    target = "优布劳精酿"
    start_time = time.time()
    for i in range(1):
        best_match = choose_best_match(target, input_text, k=1, p=3)
    end_time = time.time()
    print(end_time-start_time)
    # print(best_match)