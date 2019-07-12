# AML_2019_Group4

## Gradient Descent Implementation in Python
Gradient descent is one of the key tools in machine learning used in optimizing functions.
Gradient descent is employed to determine the best parameters to fit to a model to reduce the prediction error

## Gradient Descent: The Six-Hump Camel Function
The six-hump camel function is one of the optimization test problems used to demonstrate machine learning. To see other minimization problems, you can [click on this link](http://www.sfu.ca/~ssurjano/optimization.html).
The six hump loss function is given by:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function.png"> 
</p>
Its two partial derivatives are:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function_partial_x.png"> and  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function_partial_y.png">
</p>
The class ‘GradientDescent’ is used in performing the analysis and the notebook contains all the coding done.

The recommended evaluation range of the function is x ∈ [-3, 3] and y ∈ [-2, 2].
The function has six local minima of which two are the global minima.
The global minima of the function lie at (x, y) = (0.0898, -0.7126) and (-0.0898, 0.7126) 
The images below show the six-hump camel function:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/6_hump_plot.png">

### Note: Key Functions for Analysis:
</p> The functions used for our analysis are: the six hump camel function and its two partial derivatives with respect to both x and y:
$f(x,y)=x+y$

### Plain-Vanilla Gradient Descent
Plain vanilla gradient descent starts by working from the stated starting point in the opposite direction of the gradient at that point.
The movement of the loss function is as below:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path.gif" width="600" height="480"> 
As can be seen from the above chart, the loss function moves from an initial 162, and moves down till it gets to the local minimum at f(x,y) = 2.1043 where (x,y)=(1.7.0.896)
This is not the global minima and this is evidence of the inability of plain-vanilla gradient descent especially at very small step sizes.
Increasing the step sizes however gave different results as shown below:

  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss1.png" width="600" height="480">
The function attained the global minimum 50% of the time
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations1.png" width="600" height="480">
 <p align="center">Incresing the step size also led to a decline in the number of iterations as highlighted above
 
### Momentum Gradient Descent:
Momentum gradient descent addresses the key limitations of plain-vanilla: speed of convergence and inability to go past local minima to the global minima. 
As illustrated using the same step size of 0.001, the function converges faster in 278 steps compared to 1,599 under plain-vanilla.
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path2.png"> 
It however also has the challenge of not going past the local minima if the criteria is met.
Trying to do simulations over the same range of step sizes revealed the following:
The function exploded in many cases indicating that it went beyond the recommended evaluation rectangle.

### Nestrov’s Accelerated Gradient Descent:
This variant of plain vanilla was also meant to address its limitations.
As demonstrated below, with the same step size of 0.001, the function converges even faster and to the local minima.
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path3.png"> 
Experimenting over the same range of step sizes shows that the function explodes in very many circumstances
Comparing the three, despite its limitations, plain vanilla does converge albeit inefficiently:

<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path1.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path2.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path3.png" width="425">

<p align="center"> 
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations1.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations2.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations3.png" width="425">
  
<p align="center">  
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss1.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss2.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss3.png" width="425">

<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq1.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq2.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq3.png" width="425">
