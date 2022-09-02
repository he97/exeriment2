from model.finetune_model import build_finetune_G
from model.pretrain_model import build_model


def get_pretrain_model(config):
    return build_model(config)

def get_finetune_G(config):
    return build_finetune_G(config)