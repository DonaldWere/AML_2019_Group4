# Gradient Descent Implementation in Python
<br> **The woork was done on jupyter notebook. To see the animations and interactive displays, please use jupyter.**
<br> Gradient descent is one of the key tools in machine learning used in optimizing functions.
<br> **The animations below illustrates a typical gradient descent. _(We have put other interactive modules within the notebook)_.**
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path.gif">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path_local.gif">
</p>

## Gradient Descent: The Six-Hump Camel Function
The six-hump camel function is one of the optimization test problems un machine learning. To see other minimization problems, you can [click on this link](http://www.sfu.ca/~ssurjano/optimization.html).
<br> The six hump loss function is given by:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function.png"> 
</p>
Its two partial derivatives are:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function_partial_x.png"> and  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function_partial_y.png">
</p>
Our analysis was performed in a class called `GradientDescent` which we imported as the module `gradient_descent`.

## Visualization of the Six-Hump Camel Function
<br> The images below show the six-hump camel function in its recommended input region:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/6_hump_plot.png">
</p>

## Plain-Vanilla Gradient Descent
<br> Plain vanilla gradient descent starts by working from the stated starting point in the opposite direction of the gradient at that point.
<br> The movement of the loss function is as below:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path1.png" width="600" height="480"> 
</p>
<br> The loss function moves from the initial point at 162 to the local minimum at 2.1043 where (x,y)=(1.7,0.896)
<br> This is not the global minima and this is evidence of the inability of plain-vanilla to locate the global minimum at very small step sizes.

### Effect of changing the step size (Plain Vanilla)
</p> 1. The function converged at the global minimum with an increase in the step size:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss1.png">
</p> 2. Tt converged at the global minimum roughly 50% of the time as illustrated below:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq1.png">
</p> 3. The number of iterations to convergence decreased with an increase in the step size as highlighted below:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations1.png">

 ## Momentum Gradient Descent:
<br> Momentum gradient descent addresses the key limitations of plain-vanilla:
* Speed of convergence especially as it approaches a minimum point
* Inability to go past other local minima (or saddle points). 

<br> Using the same step size of 0.001, the function **converges faster than plain vanilla _(in 278 steps compared to 1,599 under plain-vanilla)._**
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path2.png"> 
</p> It however also has the challenge of not going past the local minimum.

### Effect of changing the step size (Momentum Gradient Descent)
</p>1. The function achieved the global minimum more over the step sizes at which it converged
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss2.png">
</p> 2. The function converged only 30% of the time implying that momentum gradient descent overshoots (disregard the infinity).
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq2.png">
</p> 3. The number of iterations needed for convergence are roughly the same over the step sizes. This is a weakness of momentum (it does big, almost equal zig-zag movements when close to a minimum point).
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations2.png">

## Nestrov’s Accelerated Gradient Descent:
<br> This variant of plain vanilla was meant to address the limitations of momentum gradient descent by reducing the momentum when approaching a minimum point hence:
* Reduce the zig-zag movements around the minimum point and hence converge faster
* Reduce the overshooting hence converge more 

<br> With the same step size of 0.001, **the function converges even faster and at the global minima.**
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_path3.png"> 

### Effect of changing the step size (Nestrov's Accelerated Gradient Descent)
</p> 1. The function converges at a minimum over more points than momentum gradient descent.
<p align="center">  
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss3.png">

</p> 2. The number of times it converges at the global minima is roughly equivalent to that under momentum gradient descent
<p align="center">  
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/min_loss_freq3.png">
</p> 3. The number of iterations needed to converge decrease with step size indicating the improvement from momentum.
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/iterations3.png">
