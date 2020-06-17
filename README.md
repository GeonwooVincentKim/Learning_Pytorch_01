# Learning_Pytorch_01
Using Anaconda3 Version 1.9.12 and Pycharm 2020.01 to learn how to use Pytorch.


# What does Squeeze for? And How does it work?
- 1. Squeeze is one of command code that decresase or increase Matrix Dimension.
     And Squeeze has an attribute to increase or decrease Matrix Dimension.
     
     - UnSqueeze is increase Matrix Dimension data to Existing Matrix Data.
       For example, if x Variable has 2 Tensor(Two-Dimension Matrix), 
       and users trying to add another Matrix Dimension, 
       it shows Current Tensor and Add another Parameter where the user wants to put it. 
       For example, "x = torch.squeeze(x, 0)".
       And The Dimension Data became 3 Tensor(Three-Dimension Matrix).
       
     - Squeeze is decrease Matrix Dimension data to Existing Matrix Data.
       For example, if x Variable has 3 Tensor(Three-Dimension Matrix),
       and Users trying to release another Matrix Dimension,
       it shows Current Tensor and release it. It doesn't need to use Another Parameter such as 0 or 1 or 2.
       

 - 2. Dimension is one of the Mathematical Conception that is really needs in Pytorch Code. 

# Gradient Descent Methods
- 1. Gradient Descent is a method of grabbing a specific point X, calculating the slope of the cost function, and changing 
     the value of X according to the slope.
     
     - Before studying Gradient Descent Methods, you should know about what cost function meaning and what does cost function
       stands for. The meaning of Cost Function is a function that corresponds to the minimum production cost at a certain output.
       
     - The other meaning of Gradient Descent Methods is Linear Regression.
