import random


def graph_split(graph_list, shot_num):
    r""" Splitting the graphs in :obj:`graph_list` into
    training data, testing data and validating data and then shuffle
    their data respectively. :obj:`[1:shot_num]` will be testing data,
    while :obj:`1/9` of the rest will be testing data and :obj:`8/9` of the
    rest will be validating data.

    Args:
          graph_list (list): A list containing all the graphs needed to be split.
          shot_num (int): :obj:`[1:shot_num]` will be separated into testing data list, and the remaining will be used to generate testing and validating data.
      """

    class_datasets = {}
    for data in graph_list:
        label = data.y
        if label not in class_datasets:
            class_datasets[label] = []
        class_datasets[label].append(data)

    train_data = []
    remaining_data = []
    for label, data_list in class_datasets.items():
        train_data.extend(data_list[:shot_num])
        random.shuffle(train_data)
        remaining_data.extend(data_list[shot_num:])

    # 将剩余的数据 1：9 划分为测试集和验证集
    random.shuffle(remaining_data)
    val_dataset_size = len(remaining_data) // 9
    val_dataset = remaining_data[:val_dataset_size]
    test_dataset = remaining_data[val_dataset_size:]
    return train_data, test_dataset, val_dataset
