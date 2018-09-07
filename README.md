# LearnToPayAttention

PyTorch implementation of ICLR 2018 paper [Learn To Pay Attention](http://www.robots.ox.ac.uk/~tvg/publications/2018/LearnToPayAttention_v5.pdf)  

![](https://github.com/SaoYan/LearnToPayAttention/blob/master/learn_to_pay_attn.png) 

My implementation is based on "(VGG-att3)-concat-pc" in the paper. I implemented two version of the model, the only difference is whether to insert the attention module before or after the max-pooling layer.

# Dependences  
* PyTorch (>=0.4.1)
* OpenCV
* [tensorboardX](https://github.com/lanpa/tensorboardX)  
#NOTE# If you are using PyTorch 0.4, then replace *torch.nn.functional.interpolate* by *[torch.nn.Upsample]*(https://pytorch.org/docs/stable/nn.html#upsample). (Modify the code in utilities.py).  

# Training  
1. Pay attention before max-pooling layers  
```
python train.py --attn_mode before --outf logs_before
```

2. Pay attention after max-pooling layers  
```
python train.py --attn_mode after --outf logs_after
```

# Results
