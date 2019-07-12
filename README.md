# AML_2019_Group4

## Gradient Descent Implementation in Python
Gradient descent is one of the key tools in machine learning used in optimizing functions.
Gradient descent is employed to determine the best parameters to fit to a model to reduce the prediction error.
<br> The animation below illustrates the plain vanilla gradient descent. We have put other interactive modules within the notebook to show the path of gradient descent.
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path.gif" width="600" height="480"> 
</p>

## Gradient Descent: The Six-Hump Camel Function
The six-hump camel function is one of the optimization test problems used to demonstrate machine learning. To see other minimization problems, you can [click on this link](http://www.sfu.ca/~ssurjano/optimization.html).
<br> The six hump loss function is given by:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function.png"> 
</p>
Its two partial derivatives are:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function_partial_x.png"> and  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function_partial_y.png">
</p>
Our analysis was performed in a class called 'GradientDescent' which we imported as the 'gradient_descent'.

## Visualization of the Six-Hump Camel Function
The recommended evaluation range of the function is x∈[-3, 3] and y∈[-2, 2].
<br> The function has six local minima of which two are the global minima.
<br> The global minima of the function lie at (x, y) = (0.0898, -0.7126) and (-0.0898, 0.7126) 
<br> The images below show the six-hump camel function:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/6_hump_plot.png">
</p>

## Plain-Vanilla Gradient Descent
Plain vanilla gradient descent starts by working from the stated starting point in the opposite direction of the gradient at that point.
<br> The movement of the loss function is as highlighted in the animation below:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path1.png" width="600" height="480"> 
</p>
<br> As can be seen from the above chart, the loss function moves from an initial point at 162, and moves down till it gets to the local minimum at f(x,y) = 2.1043 where (x,y)=(1.7.0.896)
<br> This is not the global minima and this is evidence of the inability of plain-vanilla gradient descent especially at very small step sizes.

### Effect of changing the step size(Plian Vanilla)
Changing the step size produced very different results:
<br> 1. The function converged at the global minimum with an increase in the step size, i.e:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss1.png">
<br> 2. The number of times it converged at the global minimum was roughly 50% of the time(refer to the image below).
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq1.png">
<br> 3. The number of iterations need for the function to converge also decreased with an increase in the step size as highlighted below:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations1.png">

 ### Momentum Gradient Descent:
Momentum gradient descent addresses the key limitations of plain-vanilla:
<br>  1. Speed of convergence especially as it approaches a minimum point
<br>  2. Inability to go past other local minima(or saddle points) to the global minimum point. 
As illustrated using the same step size of 0.001, the function **converges faster than plain vanilla _(in 278 steps compared to 1,599 under plain-vanilla)._**
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path2.png"> 
</p>
It however also has the challenge of not going past the local minima if the criteria is met(gradient is zero at all minima).

### Effect of changing the step size(Momentum Gradient Descent)
<br> Doing simulations over the same range of step sizes used in plain vanilla revealed the following:
<br>  1. The function achieved the global minimum more over the step sizes at which it converged
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss2.png" width="425">
<br>  2. Of concern however is the fact that the function converged only 30% of the time implying that momentum gradient descent overshoots _(disregard the infinity)_. The function exploded in many cases indicating that it went beyond the recommended evaluation rectangle.
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations2.png" width="425">


### Nestrov’s Accelerated Gradient Descent:
This variant of plain vanilla was also meant to address its limitations.
As demonstrated below, with the same step size of 0.001, the function converges even faster and to the local minima.
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path3.png"> 
Experimenting over the same range of step sizes shows that the function explodes in very many circumstances
Comparing the three, despite its limitations, plain vanilla does converge albeit inefficiently:



<p align="center"> 
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations1.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations2.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations3.png" width="425">
  
<p align="center">  
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss1.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss2.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss3.png" width="425">

<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq1.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq2.png" width="425"><img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq3.png" width="425">
