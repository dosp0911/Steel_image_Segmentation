# Steel_image_Segmentation
 Image Binary segmentation on kaggle steel data set with U-net

 - Pytorch, Trained on CoLab with one GPU Tesla P100, 100 hours
 - Dice Loss, SGD, lr_rate=0.001, no weight decay, batch size=2   
 - original image size(H,W) : (256, 1600)
 - Input/Output size(H,W) : (430, 1780), (244, 1588)
 - Input images are resized as mirroring the boundary for decreasing size of the images over training(Overlap-tile strategy)
 - Preprocessing : RGB into gray, Normalize pixels, Overlap-tile strategy, No patch used

### Result
||Train|Val|
|---|---|---|
|Dice accuracy| 0.84| 0.63|
|Dice loss| 0.15| 0.36|

<div align="center">
<img src= "https://github.com/dosp0911/Steel_image_Segmentation/blob/master/result/train.PNG?raw=true" width="300px" height="150px"/>

<img src= "https://github.com/dosp0911/Steel_image_Segmentation/blob/master/result/val_loss.PNG?raw=true" width="300px" height="150px"/>

<img src= "https://github.com/dosp0911/Steel_image_Segmentation/blob/master/result/val_acc.PNG?raw=true" width="300px" height="150px"/>
</div>


<div align="center">
<kbd>
<img src= "https://github.com/dosp0911/Steel_image_Segmentation/blob/master/result/output.PNG?raw=true" width="500px" height="500px">
 </kbd>
 </div>




#### Several problems that I faced while training
 - First I used BinaryCrossEntropy Loss function. But imbalance between background and foreground pixels made the train unstable.
   - How to fix : DICELOSS used instead. It is robust over pixel imbalance


 - Weights are learned in opposite way that I unexpected. outputs should have positive values but negative. Loss values decreased in the way that input values have negatives. It ends up cancelling off the minus sign of nominator and denominator.

  - Tried but not Fixed : so I changed the denominator to be square sum so that nominator has nothing but positive values.   

  - Tried but not Fixed : I have got binary values[0,1] from input using threshold value( > 0.5). But as you know, Loss function must be differentiable. once I got new values with threshold, the loss function became _non differentiable_. It turned out that weights from models have never trained.

  - How to fix: Consequently, I added Sigmoid function frontside so the values from the prediction are 0 < 1. So training was stable.
