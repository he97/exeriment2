from einops import rearrange
from torch import nn
from torch.autograd import Function

# BCDM Classifier
from model import Encoder


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None
def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)
class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, num_unit=2048, prob=0.2, middle=1000):
        super(ResClassifier, self).__init__()
        layers1 = []
        # currently 10000 units
        layers1.append(nn.Dropout(p=prob))
        fc11 = nn.Linear(num_unit, middle)
        self.fc11 = fc11
        layers1.append(fc11)
        layers1.append(nn.BatchNorm1d(middle, affine=True))
        layers1.append(nn.ReLU(inplace=True))
        # layers1.append(nn.Dropout(p=prob))
        fc12 = nn.Linear(middle, middle)
        self.fc12 = fc12
        layers1.append(fc12)
        layers1.append(nn.BatchNorm1d(middle, affine=True))
        layers1.append(nn.ReLU(inplace=True))
        fc13 = nn.Linear(middle, num_classes)
        self.fc13 = fc13
        layers1.append(fc13)
        self.classifier1 = nn.Sequential(*layers1)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        y = self.classifier1(x)
        return y



class Classifier(nn.Module):
    def __init__(self, num_classes=12, in_unit=2048, prob=0.2, middle=1000, middle_depth=2):
        super(Classifier, self).__init__()
        layers1 = []
        # currently 10000 units
        layers1.append(nn.BatchNorm1d(in_unit, affine=True))
        layers1.append(nn.Dropout(p=prob))
        fc11 = nn.Linear(in_unit, middle)
        # self.fc11 = fc11
        layers1.append(fc11)
        layers1.append(nn.BatchNorm1d(middle, affine=True))
        layers1.append(nn.ReLU(inplace=True))
        # layers1.append(nn.Dropout(p=prob))
        for _ in range(middle_depth):
            fc12 = nn.Linear(middle, middle)
            # self.fc12 = fc12
            layers1.append(fc12)
            layers1.append(nn.BatchNorm1d(middle, affine=True))
            layers1.append(nn.ReLU(inplace=True))
        fc13 = nn.Linear(middle, num_classes)
        # self.fc13 = fc13
        layers1.append(fc13)
        self.classifier1 = nn.Sequential(*layers1)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        y = self.classifier1(x)
        return y

class FResClassifier(nn.Module):
    def __init__(self, num_classes=12, num_unit=2048, prob=0.5, middle=1000, middle_depth=2):
        super(FResClassifier, self).__init__()
        layers1 = []
        # currently 10000 units
        # layers1.append(nn.BatchNorm1d(num_unit, affine=True))
        layers1.append(nn.BatchNorm1d(num_unit))
        layers1.append(nn.Linear(num_unit,middle))
        layers1.append(nn.BatchNorm1d(middle,affine=True))
        layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.Dropout(p=prob))
        # layers1.append(nn.Dropout(p=prob))
        for _ in range(middle_depth):
            layers1.append(nn.Linear(middle, middle))
            layers1.append(nn.BatchNorm1d(middle, affine=True))
            layers1.append(nn.ReLU(inplace=True))
            # layers1.append(nn.Dropout(p=prob))
        layers1.append(nn.Dropout(p=prob))
        layers1.append(nn.Linear(middle, num_classes))
        # self.fc13 = fc13
        # layers1.append(fc13)
        self.classifier1 = nn.Sequential(*layers1)
    def forward(self, x):
        y = self.classifier1(x)
        return y

class PengResClassifier(nn.Module):
    def __init__(self, num_classes=12, in_unit=2048, prob=0.5, middle=1000, middle_depth=2):
        middle = in_unit//2
        super(PengResClassifier, self).__init__()
        layers1 = []
        # currently 10000 units
        # layers1.append(nn.BatchNorm1d(num_unit, affine=True))
        # bn+dropout
        layers1.append(nn.BatchNorm1d(in_unit))
        layers1.append(nn.Dropout(p=prob))
        # in_unit->1/2 in_unit
        layers1.append(nn.Linear(in_unit, in_unit//2))
        layers1.append(nn.BatchNorm1d(middle,affine=True))
        layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.Dropout(p=prob))
        # (1/2 in_unit->middle)*middle_depth times
        # layers1.append(nn.Dropout(p=prob))
        for _ in range(middle_depth):
            layers1.append(nn.Linear(middle, middle))
            layers1.append(nn.BatchNorm1d(middle, affine=True))
            layers1.append(nn.ReLU(inplace=True))
            # layers1.append(nn.Dropout(p=prob))
        # layers1.append(nn.Dropout(p=prob))
        layers1.append(nn.Linear(middle, num_classes))
        # self.fc13 = fc13
        # layers1.append(fc13)
        self.classifier1 = nn.Sequential(*layers1)
    def forward(self, x):
        y = self.classifier1(x)
        return y


class AttentionClassifier(nn.Module):
    def __init__(self, num_classes=12, in_unit=2048, attention=None,prob=0.2, middle=1000, middle_depth=2):
        assert isinstance(attention,Encoder)
        super(AttentionClassifier, self).__init__()
        self.attention_layer = attention
        layers1 = []
        # currently 10000 units
        # layers1.append(nn.BatchNorm1d(in_unit, affine=True))
        layers1.append(nn.Dropout(p=prob))
        # layers1.append(attention)
        fc11 = nn.Linear(in_unit, middle)
        # self.fc11 = fc11
        layers1.append(fc11)
        layers1.append(nn.BatchNorm1d(middle, affine=True))
        layers1.append(nn.ReLU(inplace=True))
        # layers1.append(nn.Dropout(p=prob))
        for _ in range(middle_depth):
            fc12 = nn.Linear(middle, middle)
            # self.fc12 = fc12
            layers1.append(fc12)
            layers1.append(nn.BatchNorm1d(middle, affine=True))
            layers1.append(nn.ReLU(inplace=True))
        fc13 = nn.Linear(middle, num_classes)
        # self.fc13 = fc13
        layers1.append(fc13)
        self.classifier1 = nn.Sequential(*layers1)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = rearrange(x,'B F -> B 1 F')
        x = self.attention_layer(x)
        x = rearrange(x,'B H F -> B (H F)')
        y = self.classifier1(x)
        return y

if __name__ == '__main__':
    a = Classifier(num_classes=24,middle_depth=3)
    for name, value in Classifier(num_classes=24,middle_depth=3).named_parameters():
        print(f'name:{name},value:{value}')
