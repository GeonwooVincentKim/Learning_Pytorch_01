# Learning Classification using Backpropagation_Algorithm

import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Number of Dimension
# Only Possible to input 2.

# n_dim = int(input("Please Input Dimension No.2. (Only Number)"))
n_dim = 2

# if n_dim is 2:
# Now you make Clusters to find out where cluster location
# which is included data .
# Make Training Value which is defined as X and Y.
# make_blobs() is a function which is imported from Scikit-Learn Library.
x_train, y_train = make_blobs(
    n_samples=80, n_features=n_dim,
    centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]],
    shuffle=True, cluster_std=0.3
)

# Make Testing Value which is defined as X and Y.
x_test, y_test = make_blobs(
    n_samples=20, n_features=n_dim,
    centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]],
    shuffle=True, cluster_std=0.3
)

# else:
#     try:
#         print(n_dim)
#
#     except:
#         TypeError("You might be not type No.2. "
#                   "Please reboot program and input No.2 again.")


# Implement label_map() function to change all of the Data
# which have No.0 or No.1 as a Label to No.0 Label.
def label_map(y_, from_, to_):
    y = numpy.copy(y_)
    for f in from_:
        y[y_ == f] = to_
    return y


y_train = label_map(y_train, [0, 1], 0)
y_train = label_map(y_train, [2, 3], 1)
y_test = label_map(y_test, [0, 1], 0)
y_test = label_map(y_test, [2, 3], 1)


# Visualization Data which is Labeled well or not
# by using Matplotlib Library.
def vis_data(x, y=None, c='r'):
    if y is None:
        y = [None] * len(x)
    for x_, y_ in zip(x, y):
        # If Label Number is 0, it shows '.',
        # else Label Number is 1, it shows '+'.
        if y_ is None:
            plt.plot(x_[0], x_[1], '*', markerfacecolor='none',
                     markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], c + 'o' if y_ == 0 else c + '+')


plt.figure()
vis_data(x_train, y_train, c='r')
plt.show()


# Convert the Numpy Vector format data you just created
# into Pytorch Tensor format.
x_train = torch.FloatTensor(x_train)
print(x_train.shape)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


# Define Neural Network Model, 'torch.nn.Module'.
class NeuralNet(torch.nn.Module):
    """
    Bring nn.Module Class Attribute when call super() function.

    :input_size
        Dimension of Data input in the Neural Network.
    """

    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        """
        1. Defines the operations that the input data 
        goes through the Neural Network.
        
        2. ReLU outputs 0 if the input value is less than 0, and outputs the input value 
        if it is greater than 0.
        
        3. Sigmoid returns between 0 and 1. 
        -> It is better to use ReLU to classify return 0 or 1. 
           correctly.
        -> Passing through the sigmoid() function, which is limited
           to any number between 0 and 1, you can see how close the result is to 0 or 1. 
        
        4. If you increase the Number of dimension, 
        the variable named linear will increase with the Number of Dimension.
        Which mean if Number of Dimension is 3,
        the the linear variable will also being 3.
        """
        self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    # Implement forward() function that executes movements
    # defined from init() function.
    def forward(self, input_tensor):
        linear1 = self.linear_1(input_tensor)
        relu = self.relu(linear1)

        linear2 = self.linear_2(relu)
        output = self.sigmoid(linear2)
        return output


# Generate the real Neural Network and define
# various variable and algorithm for Deep Learning.
model = NeuralNet(2, 5)
learning_rate = 0.03
criterion = torch.nn.BCELoss()

"""
  1. Use Binary Cross Entropy for set Learning Rate and
     prepare for Error Function.
  
  2. Now you decide how much Whole Study Data
     would you like to input into Model.
     
  3. Make sure you should set epoch not too small,
     but not too big.
     If you set epoch small, it doesn't studied well,
     otherwise if you set epoch big, it takes lots of times
     to execute Model training such a long time.
"""
epochs = 2000

"""
  1. It could be better to choose SGD(Stochastic gradient descent)
     Algorithm for Learning Data.
     
  2. SGD is not different between optimizing Method.
  
  3. The optimizer update the Weight by the Learning Rate each time
     the step() function is called.
     So we Input Weight and Learning Rate into inside of Model 
     which is already extracted.
"""
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""
  1. Test the Model Performance without any training.
  
  2. Get some Error after call squeeze() function to match 
     Model result and Label results.
"""
model.eval()
test_loss_before = criterion(model(x_test).squeeze(), y_test)
print('Before Training, test loss is {}'.format(test_loss_before.item()))

""" 
  Neural Network Learning
"""
# Create a 'For' Statement that repeats as many Epochs.
for epoch in range(epochs):
    # Change to Training mode by calling train() function.
    model.train()

    # Set Gradient Value as 0 by calling zero_grad() function
    # to calculate new Gradient values every Epochs.
    optimizer.zero_grad()

    # Calculate Result Value by input Training Output(Data)
    # that previously generated Model.
    train_output = model(x_train)

    # Calculate Error and Make Result Value dimension and
    # Label Dimension same.
    train_loss = criterion(train_output.squeeze(), y_train)

    # Check Training well by print out loss every 100 Epochs.
    if epoch % 100 == 0:
        print('Train loss at {} is {}'.format(epoch, train_loss.item()))

    """
      Differentiate the error function by weight to find the direction 
      in which the error is the minimum, and move the model in 
      that direction as much as the Learning rate.
    """
    train_loss.backward()
    optimizer.step()

    """
      Change Model as an Test Mode by using 'x_data' and 'y_data'
      to get Error.
    """
    model.eval()
    test_loss = criterion(torch.squeeze(model(x_test)), y_test)
    print('After Training, test loss is {}'.format(test_loss.item()))

    """
      A model.pt file containing the weights of 
      the learned Neural Network is created.
    """
    torch.save(model.state_dict(), './model.pt')
    print('state_dict format of the model : {}'.format(model.state_dict()))

    new_model = NeuralNet(2, 5)
    new_model.load_state_dict(torch.load('./model.pt'))

    """
      Load stored weights and apply them to new models 
      (Transfer Learning).
      
      You can find out the Probability of Label 1 is 94 %
      by input Vector[-1, 1] into the New Model. 
    """
    new_model.eval()
    print('The probability that Vector [-1, 1] will '
          'have Label 1 is {}'.
          format(new_model(torch.FloatTensor([-1, 1])).item()))
