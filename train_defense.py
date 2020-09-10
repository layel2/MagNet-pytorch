from defensive_models import DenoisingAutoEncoder_1 as DAE_1
from defensive_models import DenoisingAutoEncoder_2 as DAE_2
import torchvision

data_train = torchvision.datasets.MNIST('./',train=True,download=True,transform=torchvision.transforms.ToTensor())
loder_train = torch.utils.data.DataLoader(data_train,batch_size=64)

ae_1 = DAE_1()
ae_1.train(loader_train,v_noise=0.1)

ae_2 = DAE_2()
ae_2.train(loader_train,v_noise=0.2)

print("train complete")

