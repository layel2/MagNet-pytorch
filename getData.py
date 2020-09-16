import torchvision
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MNISTdata():
    def __init__(self):
        self.num_channel = 1
        self.img_size = 28
        self.num_lables = 10
        data_train = torchvision.datasets.MNIST('./',train=True,download=True,transform=torchvision.transforms.ToTensor())
        data_test = torchvision.datasets.MNIST('./',train=False,download=True,transform=torchvision.transforms.ToTensor())
        
        self.train_data = torch.mul(data_train.data,1/255).reshape((-1,1,28,28)).type(torch.float32)
        self.train_labels = data_train.targets
        self.test_data = torch.mul(data_test.data,1/255).reshape((-1,1,28,28)).type(torch.float32)
        self.test_labels = data_test.targets


class dataSet(torch.utils.data.Dataset):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data = self.data[index]
        labels = self.labels[index]
        return data,labels

class normalMnist():
    def __init__(self, data_type = "test",loader_batch=128):
        data = MNISTdata()
        if data_type == 'train' :
            self.data = data.train_data
            self.labels = data.train_labels
        else :
            self.data = data.test_data
            self.labels = data.test_labels

        self.loader = torch.utils.data.DataLoader(dataSet(self.data, self.labels),batch_size=loader_batch)

class attackMnist():
    def __init__(self,attack_model,attack_method="FGSM",eps=0.3,data_type = "test",rand_seed=0,rand_min=0,rand_max=1):
        data = MNISTdata()
        if data_type == 'train' :
            self.data = data.train_data
            self.labels = data.train_labels
        else :
            self.data = data.test_data
            self.labels = data.test_labels

        if isinstance(eps,str):
            #eps random
            None
        else :
            x_atk = FGSM(attack_model,loss_fn=torch.nn.NLLLoss(),eps=eps).perturb(self.data.to(device),self.labels.to(device))
            self.data = x_atk.cpu()

        
        self.loader = dataSet(self.data, self.labels)


from advertorch.attacks.base import Attack,LabelMixin
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm

# modify from advertorch.attacks.FGSM
class FGSM(Attack,LabelMixin):

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False, getAtkpn=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(FGSM, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.getAtkpn = getAtkpn

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = xadv.grad.detach().sign()
        
        if self.getAtkpn:
            xadv = grad_sign
        else :
            xadv = xadv + self.eps * grad_sign
            xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.detach()