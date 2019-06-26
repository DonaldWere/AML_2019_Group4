import numpy as np

class gradient_descent:
    
    def __init__(self, fn_loss, fn_grad1, fn_grad2):
        self.fn_loss = fn_loss
        self.fn_grad1 = fn_grad1
        self.fn_grad2 = fn_grad2
        
    def find_min(self, x_init, y_init, max_iter, eta, tol):
       
        # initialise lists to score the path of {x,y} and the path of the loss functions
        x = x_init
        y = y_init
        num_iters = 0
        loss_path = []
        x_path = []
        y_path = []
        n_iter=[]
        #gradient_x_vector=[]
        #gradient_y_vector=[]
        x_path.append(x)
        y_path.append(y)
        n_iter.append(num_iters)
        loss_this = self.fn_loss(x, y)
        loss_path.append(loss_this)
        gradient_x = self.fn_grad1(x, y)
        gradient_y = self.fn_grad2(x, y)
        #gradient_x_vector.append(gradient_x)
        #gradient_y_vector.append(gradient_y)

        #while (gradient_x>tol or gradient_y>tol) and num_iters < max_iter:
        for i in range(max_iter):
            if abs(gradient_x) < tol and abs(gradient_y) < tol:
                break
            num_iters = num_iters + 1
            n_iter.append(num_iters)
            gradient_x = self.fn_grad1(x, y)
            x += -eta * gradient_x
            x_path.append(x)
            gradient_y = self.fn_grad2(x, y)
            y += -eta * gradient_y
            y_path.append(y)
            loss_this = self.fn_loss(x, y)
            loss_path.append(loss_this)
            #gradient_x_vector.append(gradient_x)
            #gradient_y_vector.append(gradient_y)
        
        self.loss_path = loss_path
        self.x_path = x_path
        self.y_path = y_path
        self.loss_fn_min = loss_this
        self.x_at_min = x
        self.y_at_min = y
        self.n_iter = n_iter
        self.num_iters = num_iters
        return(self.num_iters)
    