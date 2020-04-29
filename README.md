# Steel_image_Segmentation
 Image segmentation on kaggle steel data with Unet

## Several problems that I faced while training
 - Weights are learned opposite way that I expected. the output should have positive values but negative. My loss funtion is DiceLoss. Diceloss can be decreased in a way that input values have negatives. it ends up cancelling off the minus sign of nominator and denominator.

  - Tried but not Fixed : so I changed the denominator to be square sum so that nominator has nothing but positive values.   

  - Tried but not Fixed : I have got binary values[0,1] from input using threshold value( > 0.5). But as you know, Loss function must be differentiable. once I got new values with threshold, the loss function became _non differentiable_. It turned out that weights from models have never trained.

  - How to fix: Consequently, I added Sigmoid function in forward so the values from the prediction are 0 < 1. So training was stable. 
