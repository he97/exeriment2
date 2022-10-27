import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class HsiMaskGenerator:
    #     我要做的 应该就是在48个channel 随机选择几个channel就可以了
    def __init__(self, mask_ratio=0.6, in_channel=48,mask_patch_size = 1):
        self.mask_ratio = mask_ratio
        self.in_channel = in_channel
        assert in_channel % mask_patch_size == 0, 'in_channel not match mask_patch_size'
        self.mask_patch_size = mask_patch_size
        self.patch_count = self.in_channel // self.mask_patch_size
        self.mask_count = int(np.ceil(self.patch_count * self.mask_ratio))

    #     def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
    #         self.input_size = input_size
    #         self.mask_patch_size = mask_patch_size
    #         self.model_patch_size = model_patch_size
    #         self.mask_ratio = mask_ratio

    #         assert self.input_size % self.mask_patch_size == 0
    #         assert self.mask_patch_size % self.model_patch_size == 0

    #         self.rand_size = self.input_size // self.mask_patch_size
    #         # 尺度 一个mask点代表几个原本的图片的点
    #         self.scale = self.mask_patch_size // self.model_patch_size

    #         self.token_count = self.rand_size ** 2
    #         # 有多少个mask的区域
    #         self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
    def __call__(self):
        mask_idx = np.random.permutation(self.patch_count)[:self.mask_count]
        # 建立一个所有token的零数组
        mask = np.zeros(self.patch_count, dtype=int)
        # 被选取中的mask区域置1
        mask[mask_idx] = 1
        # mask = mask.transpose()
        # mask = mask[:, np.newaxis]
        return mask

class HsiMaskTensorDataSet(Dataset):
    def __init__(self, data, label, transform=None):
        """
        dataset_type: ['train', 'test']
        """

        self.transform = transform
        self.label = label
        self.data = data

    def __getitem__(self, index):
        img = self.data[index]
        mask = self.transform()
        _ = self.label[index]
        return img, mask, _

    def __len__(self):
        return self.data.size(0)


def to_group(tensor, config):
    [B, C, H, W] = tensor.shape
    tensor = tensor.reshape(B, C, -1)
    mask_patch_size = config.DATA.MASK_PATCH_SIZE
    assert C % mask_patch_size == 0, 'can not devide channels to groups'
    d = C // mask_patch_size
    tensor = tensor.reshape(B, d, -1)
    return tensor

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

def get_mask_dataloader(config,size):
    all_samples = torch.randn(size)
    all_labels = torch.full((size[0],), 1.0)

    transform = HsiMaskGenerator(config.DATA.MASK_RATIO, all_samples.shape[1],
                                 mask_patch_size=config.DATA.MASK_PATCH_SIZE)
    all_samples = to_group(all_samples, config)
    dataset = HsiMaskTensorDataSet(torch.tensor(all_samples), torch.tensor(all_labels), transform=transform)
    data_loader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=0, sampler=None,
                             pin_memory=True, drop_last=True)
    return data_loader

def get_tensor_dataset(size=(256, 48, 5, 5), tensor_type='randn', have_label=True):
    """
    通过size和类型设置张量。
    eye 不能用 没改了
    :param size:
    :param tensor_type:
    :param have_label:
    :return:
    """
    vector = torch.randn(size)
    if tensor_type == 'eye':
        vector = torch.eye(size)
    elif tensor_type == 'randn':
        vector = torch.randn(size)
    elif tensor_type == 'randint':
        vector = torch.randint(1, 3, size)
    elif tensor_type == 'full':
        vector = torch.full(size, 1)
    elif tensor_type == 'arange':
        s = 1
        for x in size:
            s *= x
        vector = torch.arange(s)
        vector.reshape(size)
    vector = vector.type(torch.FloatTensor)
    if have_label:
        label = torch.randint(6, (size[0],))
        label = label.type(torch.FloatTensor)
        return TensorDataset(vector, label)
    else:
        return TensorDataset(vector)