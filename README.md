# AML_2019_Group4

## Gradient Descent Implementation in Python
</p>Gradient descent is one of the key useful tools in machine learning.
</p>In machine learning, gradient descent is very important when optimizing functions.
</p>It is employed to determine the best parameters to fit in a model to reduce the prediction error

## Gradient Descent: The Six-Hump Camel Function
</p>The six-hump camel function is one of the optimization test problems used to demonstrate machine learning.
</p>The recommended evaluation range of the function is x ∈ [-3, 3] and y ∈ [-2, 2].
</p>The function has six local minima of which two are the global minima.
</p>The global minima of the function lie at (x, y) = (0.0898, -0.7126) and (-0.0898, 0.7126) 
</p>The images below show the six-hump camel function:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/6_hump_plot.png" width="425"> <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/6_hump_closeup.png" width="425">

## Note: Key Functions for Analysis:
</p>The functions used for our analysis are: the six hump camel function and its two partial derivatives with respect to both x and y:
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/loss_function.PNG"> 
</p>
<p align="center">
  <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/partial_x.png"> <img src="https://github.com/DennisOndieki/AML_2019_Group4/blob/master/Images/partial_y.png">
</p>
</p>The class ‘GradientDescent’ is used in performing the analysis and the notebook contains all the coding done.

## Plain-Vanilla Gradient Descent
</p>Plain vanilla gradient descent works by working from the stated starting point in the opposite direction of the gradient at that point.
</p>As can be seen from the above chart, the loss function moves from an initial 150, and moves down till it gets to the local minima at f(x,y) = 2.1043 where (x,y)=(1.7.0.896)
</p>This is not the global minima and this is evidence of the inability of plain-vanilla gradient descent especially at very small step sizes.
</p>Increasing the step sizes however gave different results as shown below:

## Momentum Gradient Descent:
<br>Momentum gradient descent addresses the key limitations of plain-vanilla: speed of convergence and inability to go past local minima to the global minima. 
<br>As illustrated using the same step size of 0.001, the function converges faster in 278 steps compared to 1,599 under plain-vanilla.
<br>It however also has the challenge of not going past the local minima if the criteria is met.
<br>Trying to do simulations over the same range of step sizes revealed the following:
<br>The function exploded in many cases indicating that it went beyond the recommended evaluation rectangle.

##Nestrov’s Accelerated Gradient Descent:
This variant of plain vanilla was also meant to address its limitations.
As demonstrated below, with the same step size of 0.001, the function converges even faster and to the local minima.

Experimenting over the same range of step sizes shows that the function explodes in very many circumstances
Comparing the three, despite its limitations, plain vanilla does converge albeit inefficiently
