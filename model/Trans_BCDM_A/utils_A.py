import torch
import scipy.io as sio
import numpy as np
from sklearn import preprocessing

############################ Loss cdd ########################
def cdd(output_t1,output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss

############################ Data procesing ########################
def cubeData(src_path, src_label_path, tgt_path, tgt_label_path, dataset):
    if dataset == 'huston':
        temp = sio.loadmat(src_path)
        print(temp.keys())
        data1 = temp['ori_data']
        print(data1.shape)
        temp = sio.loadmat(src_label_path)
        print(temp.keys())
        gt1 = temp['map']
        temp = sio.loadmat(tgt_path)
        print(temp.keys())
        data2 = temp['ori_data']
        print(data2.shape)
        temp = sio.loadmat(tgt_label_path)
        print(temp.keys())
        gt2 = temp['map']
    elif dataset == 'pavia':
        temp = sio.loadmat(src_path)
        print(temp.keys())
        data1 = temp['pavia']
        print(data1.shape)
        temp = sio.loadmat(src_label_path)
        print(temp.keys())
        gt1 = temp['pavia_gt_7']
        temp = sio.loadmat(tgt_path)
        print(temp.keys())
        data2 = temp['paviaU']
        print(data2.shape)
        temp = sio.loadmat(tgt_label_path)
        print(temp.keys())
        gt2 = temp['paviaU_gt_7']
    elif dataset == 'indian':
        temp = sio.loadmat(src_path)
        print(temp.keys())
        data1 = temp['DataCube1']
        print(data1.shape)
        gt1 = temp['gt1']
        data2 = temp['DataCube2']
        print(data2.shape)
        temp = sio.loadmat(tgt_label_path)
        gt2 = temp['gt2']
    elif dataset == 'shanghai-hangzhou':
        temp = sio.loadmat(src_path)
        print(temp.keys())
        data1 = temp['DataCube1']
        print(data1.shape)
        gt1 = temp['gt1']
        data2 = temp['DataCube2']
        print(data2.shape)
        temp = sio.loadmat(tgt_label_path)
        gt2 = temp['gt2']


    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))  # (111104,204)
    data_scaler_s = preprocessing.scale(data_s)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1],data1.shape[2])

    data_t = data2.reshape(np.prod(data2.shape[:2]), np.prod(data2.shape[2:]))  # (111104,204)
    data_scaler_t = preprocessing.scale(data_t)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_t = data_scaler_t.reshape(data2.shape[0], data2.shape[1],data2.shape[2])

    return Data_Band_Scaler_s, gt1, Data_Band_Scaler_t, gt2

# 获取高光谱数据
def cubeData1(img_path, label_path, dataset):
    if dataset == 'houston':
        temp1 = sio.loadmat(img_path)
        print(temp1.keys())
        data1 = temp1[list(temp1.keys())[-1]]
        print(data1.shape)
        temp2 = sio.loadmat(label_path)
        print(temp2.keys())
        gt1 = temp2[list(temp2.keys())[-1]]
        print(gt1.shape)
    elif dataset == 'pavia':
        temp1 = sio.loadmat(img_path)
        print(temp1.keys())
        data1 = temp1[list(temp1.keys())[-1]]
        print(data1.shape)
        temp2 = sio.loadmat(label_path)
        print(temp2.keys())
        gt1 = temp2[list(temp2.keys())[-1]]
        print(gt1.shape)
    else:
        raise Exception('this dataset not support')
    # 6.5 之前没有。可能是因为从别的文件中复制过来的，那个没使用归一化？
    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))  # (111104,204)
    data_scaler_s = preprocessing.scale(data_s)  # 标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1], data1.shape[2])

    return Data_Band_Scaler_s, gt1
    # return Data_Band_Scaler_s, gt1



def get_all_data(All_data, All_label, HalfWidth):
    print('get_all_data() run...')
    print('The original data shape:', All_data.shape)
    nBand = All_data.shape[2]

    data = np.pad(All_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(All_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    num_class = int(np.max(label))
    print(f'num_class : {num_class}')

    for i in range(num_class):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices

    for i in range(num_class):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of all data:', len(train_indices))
    nTest = len(train_indices)
    index = np.zeros([nTest], dtype=np.int64)
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTest):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')
    return processed_data, processed_label


def get_sample_data(Sample_data, Sample_label, HalfWidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:',Sample_data.shape)
    nBand = Sample_data.shape[2]

    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    m = int(np.max(label))
    print(f'num_class : {m}')

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices[:num_per_class]

    for i in range(m):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label

def get_sample_data_without_train_val(Sample_data, Sample_label, HalfWidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:', Sample_data.shape)
    # 波段数
    nBand = Sample_data.shape[2]
    # 数据变成了 214 958 48 原因是因为之后要将有标记的点旁边的值一起取出来
    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')

    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    # 返回索引
    [Row, Column] = np.nonzero(label)
    # 类别数
    m = int(np.max(label))
    print(f'num_class : {m}')
    for i in range(m):
        # ravel: return a 1D array
        # 类别数 从 1 到 m 0代表的应该是默认情况
        # 这个索引的值，是column和row的索引值，下面的式子
        # [j for j, x in enumerate(len(Row)) if label[Row[j], Column[j]] == i + 1]
        # 等价的
        # 本质就是把所有不为0的点，都拿出来查一遍
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        # num_per_class 有什么作用？ 选择训练和测试的数目
        per_class_num = int(len(indices)) if num_per_class <= 0 else num_per_class
        train[i] = indices[:per_class_num]
        # val[i] = indices[num_per_class:]

    for i in range(m):
        train_indices += train[i]
        # val_indices += val[i]
    #     再次打乱
    np.random.shuffle(train_indices)
    # np.random.shuffle(val_indices)

    # #val
    # print('the number of val data:', len(val_indices))
    # nVAL = len(val_indices)
    # val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    # val_label = np.zeros([nVAL], dtype=np.int64)
    # RandPerm = val_indices
    # RandPerm = np.array(RandPerm)
    #
    # for i in range(nVAL):
    #     # 将原数据的不为0的点作为中心，一个5*5的块，整体作为数据。
    #     val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
    #                                               Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
    #                                               :],
    #                                               (2, 0, 1))
    #     val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    # val_label = val_label - 1

    # train
    # 和之前test的做同样的处理
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label

def get_sample_data_spatial_spectral(Sample_data, Sample_label, half_width_spatial,half_width_spectral, num_per_class):
    '''

    :param Sample_data:
    :param Sample_label:
    :param half_width_spatial:
    :param half_width_spectral:
    :param num_per_class:
    :return:空间 光谱 标签
    '''
    print('get_sample_data() run...')
    print('The original sample data shape:', Sample_data.shape)
    # 波段数
    nBand = Sample_data.shape[2]
    # 数据变成了 214 958 48 原因是因为之后要将有标记的点旁边的值一起取出来
    # 把数据分为光谱信息和空间信息
    max_half_width = max(half_width_spectral,half_width_spatial)
    data_spatial_spectral = np.pad(Sample_data, ((max_half_width, max_half_width), (max_half_width, max_half_width),
                                        (0, 0)), mode='constant')

    label = np.pad(Sample_label, half_width_spatial, mode='constant')

    train = {}
    train_indices = []
    # 返回索引
    [Row, Column] = np.nonzero(label)
    # 类别数
    m = int(np.max(label))
    print(f'num_class : {m}')
    for i in range(m):
        # ravel: return a 1D array
        # 类别数 从 1 到 m 0代表的应该是默认情况
        # 这个索引的值，是column和row的索引值，下面的式子
        # [j for j, x in enumerate(len(Row)) if label[Row[j], Column[j]] == i + 1]
        # 等价的
        # 本质就是把所有不为0的点，都拿出来查一遍
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        # num_per_class 有什么作用？ 选择训练和测试的数目
        per_class_num = int(len(indices)) if num_per_class<=0 else num_per_class
        train[i] = indices[:per_class_num]
        # val[i] = indices[num_per_class:]

    for i in range(m):
        train_indices += train[i]
        # val_indices += val[i]
    #     再次打乱
    np.random.shuffle(train_indices)
    # np.random.shuffle(val_indices)

    # #val
    # print('the number of val data:', len(val_indices))
    # nVAL = len(val_indices)
    # val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    # val_label = np.zeros([nVAL], dtype=np.int64)
    # RandPerm = val_indices
    # RandPerm = np.array(RandPerm)
    #
    # for i in range(nVAL):
    #     # 将原数据的不为0的点作为中心，一个5*5的块，整体作为数据。
    #     val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
    #                                               Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
    #                                               :],
    #                                               (2, 0, 1))
    #     val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    # val_label = val_label - 1

    # train
    # 和之前test的做同样的处理
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data_spatial = np.zeros([nTrain, nBand, 2 * half_width_spatial + 1, 2 * half_width_spatial + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    processed_data_spectral = np.zeros([nTrain, nBand, 2 * half_width_spectral + 1, 2 * half_width_spectral + 1], dtype=np.float32)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data_spatial[i, :, :, :] = np.transpose(data_spatial_spectral[Row[RandPerm[i]] - half_width_spatial: Row[RandPerm[i]] + half_width_spatial + 1, \
                                                  Column[RandPerm[i]] - half_width_spatial: Column[RandPerm[i]] + half_width_spatial + 1,
                                                  :],
                                                  (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
        processed_data_spectral[i, :, :, :] = np.transpose(
            data_spatial_spectral[Row[RandPerm[i]] - half_width_spectral: Row[RandPerm[i]] + half_width_spectral + 1, \
            Column[RandPerm[i]] - half_width_spectral: Column[RandPerm[i]] + half_width_spectral + 1,
            :],
            (2, 0, 1))
    processed_label = processed_label - 1
    # processed_label_spectral = processed_label_spectral - 1

    print('sample data spatial shape', processed_data_spatial.shape)
    print('sample data spectral shape', processed_data_spectral.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    # 返回空间 光谱 标签
    return processed_data_spatial, processed_data_spectral, processed_label