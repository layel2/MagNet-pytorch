import torch
import getData
from worker import *
from defensive_models import *
import mnist_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

clf = mnist_model.getmodel(load_path="mnist_model/mnistModel_A_sd.pth").to(device)
data = getData.normalMnist()
data_atk = getData.attackMnist(attack_model=clf,eps=0.3)

dae1 = DenoisingAutoEncoder_1()
dae1.load('./saved_model/model1.pth')

dae2 = DenoisingAutoEncoder_2()
dae2.load('./saved_model/model2.pth')

detector_1 = AEDectector(dae1,p=2)
detector_2 = AEDectector(dae2,p=1)
reformer = SimpleReformer(dae1)
id_reformer = IdReformer()

detector_dict = dict()
detector_dict["1"] = detector_1
detector_dict["2"] = detector_2

operator = Operator(data, clf, detector_dict, reformer)
evau = Evauator(operator,data_atk)
thrs = evau.operator.get_thrs({'1':0.001,'2':0.001})
all_pass = evau.operator.filter(evau.operator.data.data,thrs)

result = evau.get_attack_acc(evau.operator.filter(data_atk.data,thrs)[0])

print('No defense accuracy : ',result[3])
print('Reformer only accuracy : ',result[2])
print('Detector only accuracy : ',result[1])
print('Both detector and reformer accuracy : ',result[0])