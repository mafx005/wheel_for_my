# -*- coding: utf-8 -*-
# @Time : 2021/3/1 17:11
# @Author : M
# @FileName: save_bert_embed.py
# @Dec : 将bert词向量保存为npy形式，内存加载
import numpy as np
from tensorflow.python import pywrap_tensorflow

ckpt_input_path = './data/models/tf_model.ckpt'
ckpt_output_path = './data/models/bert.npy'


def save_bert_embeding_to_npy(input_file, output_file):
    reader = pywrap_tensorflow.NewCheckpointReader(input_file)
    param_dict = reader.get_variable_to_shape_map()  # 读取 ckpt中的参数的维度的
    emd = reader.get_tensor('bert/embeddings/word_embeddings')  # 得到ckpt中指定的tensor
    print(len(emd))
    print(emd[:5])
    param = np.array(emd)
    np.save(output_file, param)
    return


# save_bert_embeding_to_npy(ckpt_input_path, ckpt_output_path)