# MagNet-pytorch

 Code for MagNet: a Two-Pronged Defense against Adversarial Examples in pytorch (paper : https://arxiv.org/abs/1705.09064)
 
 Base on : https://github.com/Trevillie/MagNet
 
 Train autoencoder
 

    python train_defense.py

Test

    python test_defense.py
You can test on another attack method by edit test_defense.py Line 12

    data_atk = getData.attackMnist(attack_model=clf,eps=0.3)
*use data from torch dataloader with data,labels or `getData.dataLoader(data,labels)`
