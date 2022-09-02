import torch.optim as optim


def build_optimizer(config, model, pretrain):
    if pretrain:
        return optim.Adam(list(model.parameters()), lr=config.TRAIN.PRETRAIN_LR.LR)
    else:
        return optim.Adam(list(model.parameters()), lr=config.TRAIN.FINETUNE_LR.LR)
