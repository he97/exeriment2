from torch import nn
from torch.autograd import Function

# BCDM Classifier
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
    def __init__(self, num_classes=12, num_unit=2048, prob=0.2, middle=1000, middle_depth=2):
        super(Classifier, self).__init__()
        layers1 = []
        # currently 10000 units
        layers1.append(nn.BatchNorm1d(num_unit, affine=True))
        layers1.append(nn.Dropout(p=prob))
        fc11 = nn.Linear(num_unit, middle)
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
    def __init__(self, num_classes=12, num_unit=2048, prob=0.2, middle=1000, middle_depth=2):
        super(FResClassifier, self).__init__()
        layers1 = []
        # currently 10000 units
        # layers1.append(nn.BatchNorm1d(num_unit, affine=True))
        layers1.append(nn.Linear(num_unit,middle))
        layers1.append(nn.BatchNorm1d(middle,affine=True))
        layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.Dropout(p=prob))
        # layers1.append(nn.Dropout(p=prob))
        for _ in range(middle_depth):
            layers1.append(nn.Linear(middle, middle))
            layers1.append(nn.BatchNorm1d(middle, affine=True))
            layers1.append(nn.ReLU(inplace=True))
            layers1.append(nn.Dropout(p=prob))
        fc13 = nn.Linear(middle, num_classes)
        # self.fc13 = fc13
        # layers1.append(fc13)
        self.classifier1 = nn.Sequential(*layers1)
    def forward(self, x):
        y = self.classifier1(x)
        return y
if __name__ == '__main__':
    a = Classifier(num_classes=24,middle_depth=3)
    for name, value in Classifier(num_classes=24,middle_depth=3).named_parameters():
        print(f'name:{name},value:{value}')
