import torch
import numpy as np
#from scipy.stats import entropy
from numpy.linalg import norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AEDectector():
    def __init__(self,model,load_path=None,p=1):
        if load_path != None :
            model.load(load_path)

        self.model = model.model
        self.path = load_path
        self.p = p

    def mark(self,X):
        X = X.to(device)
        diff = torch.abs(X - self.model(X))
        marks = torch.mean(torch.pow(diff, self.p), axis=(1,2,3))
        return marks

    def print(self):
        return "AEDetector:" + self.path.split("/")[-1]

class IdReformer():
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer.
        Reforms an example to itself.
        """
        self.path = path
        self.heal = lambda X: X

    def print(self):
        return "IdReformer:" + self.path

class SimpleReformer():
    def __init__(self,model,load_path=None):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.

        path: Path to the autoencoder used.
        """
        self.model = model.model
        self.path = load_path

    def heal(self, X):
        X = self.model(X)
        return torch.clamp(X, 0.0, 1.0)

    def print(self):
        return "SimpleReformer:" + self.path.split("/")[-1]

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

class Classifier():
    def __init__(self,clf):
        self.model = clf

    def classify(self,X):
        return self.model(X)

class Operator():
    def __init__(self,data,classifier,det_dict,reformer):
        self.data = data
        self.classifier = classifier
        self.det_dict = det_dict
        self.reformer = reformer
        self.normal = self.operate(self.data)

    def operate(self,untrusted_obj):
        X = untrusted_obj.data.to(device)
        y_true = untrusted_obj.labels.to(device)

        X_prime = self.reformer.heal(X)
        y = torch.argmax(self.classifier(X),axis=1)
        y_judge = (y == y_true[:len(X_prime)])
        y_prime = torch.argmax(self.classifier(X_prime),axis=1)
        y_prime_judge = (y_prime == y_true[:len(X_prime)])

        return np.array(list(zip(y_judge,y_prime_judge)))

    def filter(self, X, thrs):
        collector = dict()
        all_pass = np.array(range(10000))
        for name,detector in self.det_dict.items():
            marks = detector.mark(X).cpu().detach().numpy()
            idx_pass = np.argwhere(marks < thrs[name].cpu().detach().numpy())
            collector[name] = len(idx_pass)
            all_pass = np.intersect1d(all_pass, idx_pass)
        return all_pass, collector

    def get_thrs(self, drop_rate):
        """
        Get filtering threshold by marking validation set.
        """
        thrs = dict()
        for name, detector in self.det_dict.items():
            num = int(len(self.data.data) * drop_rate[name])
            marks = detector.mark(self.data.data)
            marks,_ = torch.sort(marks)
            thrs[name] = marks[-num]
        return thrs

class AttackData():
    def __init__(self,examples,labels,name=""):
        if isinstance(examples, str): self.data = utils.load_obj(examples)
        else: 
            self.data = examples
        self.labels = labels
        self.name = name

    def print(self):
        return "Attack:"+self.name

class Evauator():
    def __init__(self, operator, untrusted_data, graph_dir="./graph"):

        self.operator = operator
        self.untrusted_data = untrusted_data
        self.graph_dir = graph_dir
        self.data_package = operator.operate(untrusted_data)

    def bind_operator(self, operator):
        self.operator = operator
        self.data_package = operator.operate(self.untrusted_data)

    def load_data(self, data):
        self.untrusted_data = data
        self.data_package = self.operator.operate(self.untrusted_data)

    def get_normal_acc(self, normal_all_pass):

        normal_tups = self.operator.normal
        num_normal = len(normal_tups)
        filtered_normal_tups = normal_tups[normal_all_pass]

        both_acc = sum(1 for _, XpC in filtered_normal_tups if XpC)/num_normal
        det_only_acc = sum(1 for XC, XpC in filtered_normal_tups if XC)/num_normal
        ref_only_acc = sum([1 for _, XpC in normal_tups if XpC])/num_normal
        none_acc = sum([1 for XC, _ in normal_tups if XC])/num_normal

        return both_acc, det_only_acc, ref_only_acc, none_acc

    def get_attack_acc(self, attack_pass):
        attack_tups = self.data_package
        num_untrusted = len(attack_tups)
        filtered_attack_tups = attack_tups[attack_pass]

        both_acc = 1 - sum(1 for _, XpC in filtered_attack_tups if not XpC)/num_untrusted
        det_only_acc = 1 - sum(1 for XC, XpC in filtered_attack_tups if not XC)/num_untrusted
        ref_only_acc = sum([1 for _, XpC in attack_tups if XpC])/num_untrusted
        none_acc = sum([1 for XC, _ in attack_tups if XC])/num_untrusted
        return both_acc, det_only_acc, ref_only_acc, none_acc

