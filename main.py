from typing import Callable, Tuple
import numpy as np

student_id_str = '000000000'
student_email_address = r'000000000'

def tuple_print (t : tuple):
    """
    Print a tuple input
    """
    print ("{0}".format(t))
    return 0
       
def student_id():
    """
    Returns a tuple of student's ID and TAU email address.
    """
    return student_id_str, student_email_address

class QuadraticFunction:
    def __init__(   
        self, 
        Q: np.ndarray, 
        b: np.ndarray
    ) -> None:
        """
        Initializes a quadratic function with NumPy matrix 𝑄 and vector b.
        Attributes names should be ‘Q’ and ‘b’ respectively.
        Arguments:
            Q : matrix
            b : vector
        """
        self.Q = Q
        self.b = b

    def __call__(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates 𝑓(𝑥) at 𝑥
        Arguments:
            x : vector
        Returns:
            fx : scalar
        """
        fx = (0.5 * np.matmul(np.transpose(x),np.matmul(self.Q,x))) + \
             (np.matmul(np.transpose(self.b),x))
        return fx
    
    def grad(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates 𝑔(𝑥), the gradient of 𝑓(𝑥), at 𝑥
        Arguments:
            x : vector
        Returns:
            gx : vector
        """
        gx = 0.5 * np.matmul((self.Q + np.transpose(self.Q)),x) + \
             self.b
        return gx

    def hessian(
        self,
        x: np.ndarray
    ) -> np.ndarray :
        """
        Evaluates 𝐻(𝑥), the Hessian of 𝑓(𝑥), at 𝑥
        Arguments:
            x : vector
        Returns:
            hx : matrix
        """
        hx = 0.5 * (self.Q + np.transpose(self.Q))
        return hx
        
    def print_call(
        self,
        x: np.ndarray
    ) -> None:
        """
        prints 𝑓(𝑥) at 𝑥
        Arguments:
            x : vector
        """
        print ("f(x) at \nx = {0}\nf(x) =\n{1}".format(x, self.__call__(x)))
        
    def print_grad(
        self,
        x: np.ndarray
    ) -> None:
        """
        print 𝑔(𝑥), the gradient of 𝑓(𝑥), at 𝑥
        Arguments:
            x : vector
        """
        print ("grad of f(x) at \nx = {0}\ng(x) = {1}".format(x, self.grad(x)))
    
    def print_hessian(
        self,
        x: np.ndarray
    ) -> None:
        """
        print h(𝑥), the Hessian of 𝑓(𝑥), at 𝑥
        Arguments:
            x : vector
        """
        print ("hessian of f(x) at \nx = {0}\nh(x) =\n{1}".format(x, self.hessian(x)))

class NewtonOptimizer:

    def __init__(
        self,
        objective: Callable,
        x_0: np.ndarray,
        alpha: float,
        threshold: float,
        max_iters: int
    ) -> None:
        """
        Initializes a Newton’s method optimizer with an objective function.
        Arguments:
            objective : callable to an objective function, such as QuadraticFunction above
            x_0 : vector, initial guess for the minimizer
            alpha : (scalar) constant step size
            threshold : scalar, stopping criteria |𝑥_k+1 − 𝑥_k| < threshold, return 𝑥_k+1
            max_iters : scalar, maximal number of iterations (stopping criteria)
        """
        self.objective = objective
        self.x_0 = x_0
        self.x = x_0
        self.alpha = alpha
        self.threshold = threshold
        self.step_num = 0
        self.max_iters = max_iters
        
    def step(self) -> Tuple:
        """
        Executes a single step of newton’s method.
        Return: a tuple (next_x, gx, hx) as follow:
            next_x : vector, updated x.
            gx : vector, the gradient of 𝑓(𝑥) evaluated at the current 𝑥 (not next_x)
            hx : matrix, the Hessian of 𝑓(𝑥) evaluated at the current 𝑥 (not next_x)
        """
        gx = self.objective.grad(self.x)
        hx = self.objective.hessian(self.x)
        next_x = self.x - \
                 self.alpha * np.matmul(np.linalg.inv(hx), gx)
        self.x = next_x
        self.step_num += 1
        return next_x , gx, hx

    def optimize(self) -> Tuple:
        """
        Execution of optimization flow
        Return: a tuple as follow:
            fmin : scalar, objective function evaluated at x_opt
            minimizer : vector, the optimal 𝑥
            num_iters : scalar, number of iterations until convergence
        """       
        while (self.step_num < self.max_iters):
            previous_x = self.x
            self.step()
            if np.linalg.norm(self.x - previous_x , 2) < self.threshold:
                break
        return self.objective.__call__(self.x) , \
               self.x , \
               self.step_num
    
    def print_step(self) -> Tuple:
        """
        Executes a single step of newton’s method.
        Prints: a tuple (next_x, gx, hx) as follow:
            next_x : vector, updated x.
            gx : vector, the gradient of 𝑓(𝑥) evaluated at the current 𝑥 (not next_x)
            hx : matrix, the Hessian of 𝑓(𝑥) evaluated at the current 𝑥 (not next_x)
        """
        next_x, gx, hx = self.step()
        print ("next_step of Newton optimizer leads to \nx = {0}\ng(x) = {1}\nh(x) =\n{2}".format(next_x,
                                                                                                  gx,
                                                                                                  hx))
        
    def print_optimize(self) -> Tuple:
        """
        Execution of optimization flow
        Prints: a tuple as follow:
            fmin : scalar, objective function evaluated at x_opt
            minimizer : vector, the optimal 𝑥
            num_iters : scalar, number of iterations until convergence
        """
        f_min, minimizer, num_iters = self.optimize()
        print ("minimizer by Newton optimizer leads to \nx_min = {0}\ng_min = {1}\nf_min = {2}\nnum_iters = {3}".format(minimizer, self.objective.grad(self.x), f_min, num_iters))

class BFGSOptimizer:

    def __init__(
        self, 
        objective: Callable,
        x_0: np.ndarray,
        B_0: np.ndarray,
        alpha_0: float,
        beta: float,
        sigma: float,
        threshold: float,
        max_iters: int
    ) -> None:
        """
        Initializes a Newton’s method optimizer.
        Arguments:
            objective : callable to an objective function, such as QuadraticFunction above
            x_0 : vector, initial guess for the minimizer
            B_0 : matrix , initial guess of the inverse Hessian
            alpha_0 : scalar, initial step size for Armijo line search
            beta : scalar, beta parameter of Armijo line search, a float in range (0,1)
            sigma : scalar, sigma parameter of Armijo line search, a float in range (0,1)
            threshold : scalar, stopping criteria (𝑥!"# − 𝑥!( < threshold, return 𝑥!"#
            max_iters : scalar, maximal number of iterations (stopping criteria)
        """
        self.objective = objective
        self.x_0 = x_0
        self.x = x_0
        self.B_0 = B_0
        self.B = B_0
        self.alpha_0 = alpha_0
        self.beta = beta
        self.sigma = sigma
        self.threshold = threshold
        self.max_iters = max_iters
        self.step_num = 0
        self.xsi = 1 #BFGS method
        self.update_dir()
        self.update_step_size()

    def update_dir(self) -> np.ndarray:
        """
        Computes step direction.
        Return:
            next_d : vector, the new direction
        """
        self.dir = (-1) * np.matmul(self.B, self.objective.grad(self.x))
        return self.dir
    
    def step_call_diff(
            self,
            step: float
    ) -> np.ndarray:
        """
        Evaluates 𝑓(𝑥 + step) - f(x)
        Arguments:
            step : vector
        Returns:
            delta_f_x : scalar
        """
        return self.objective.__call__(self.x + step * self.dir) - self.objective.__call__(self.x)
    
    def update_step_size(self) -> np.ndarray:
        """
        Compute the new step size using Backtracking Line Search algorithm (Armijo rule).
        Follow the algorithm described in class (see recording).
        Return:
            step_size : scalar
        """
        alpha = self.alpha_0
        if self.step_call_diff(alpha) > (-1) * self.sigma * alpha * np.linalg.norm(self.dir):
            while self.step_call_diff(alpha) > (-1) * self.sigma * alpha * np.linalg.norm(self.dir):
                alpha = self.beta * alpha
        else:
            alpha = (1.0 / self.beta) * alpha
            while self.step_call_diff(alpha) <= (-1) * self.sigma * alpha * np.linalg.norm(self.dir):
                alpha = (1.0 / self.beta) * alpha
            alpha = self.beta * alpha
        self.alpha = alpha
        return self.alpha

    def update_x(self) -> np.ndarray:
        """
        Take a step in the descending direction.
        Return:
            next_x : vector, updated x
        """
        self.x = self.x + self.alpha * self.dir
        return self.x

    def update_inv_hessian(
        self,
        prev_x: np.ndarray
    ) -> np.ndarray:
        """
        Compute the approximator of the inverse Hessian using BFGS algorithm
        with rank-2 update.
        Arguments:
            prev_x : vector, previous point x
        Return:
            next_inv_hessian : matrix, approximator of the inverse Hessian matrix
        """
        p = self.x - prev_x
        q = self.objective.grad(self.x) - self.objective.grad(prev_x)
        s = np.matmul(self.B, q)
        tau = np.matmul(np.transpose(s), q)
        mu = np.matmul(np.transpose(p), q)
        v = (p / mu) - (s / tau)
        self.B = self.B + \
                 (np.matmul(p, np.transpose(p)) / mu) + \
                 (np.matmul(v, np.transpose(v)) * tau * self.xsi) - \
                 (np.matmul(s, np.transpose(s)) / tau)
        return self.B

    def step(self) -> Tuple:
        """
        Executes a single Quasi-Newton step.
        Return:
            a tuple (next_x, next_d, step_size, next_inv_hessian) as follows:
            next_x : vector, updated x
            next_d : vector, updated direction
            step_size : scalar
            next_inv_hessian : matrix, approximator of the inverse Hessian matrix
        """
        prev_x = self.x
        self.update_x()
        self.update_inv_hessian(prev_x)
        self.update_dir()
        self.update_step_size()
        self.step_num += 1
        return self.x, self.dir, self.alpha, self.B

    def optimize(self) -> Tuple:
        """
        Execution of optimization flow.
        Return:
            fmin : scalar, objective function evaluated at the minimum
            minimizer : vector, the optimal 𝑥
            num_iters : scalar, number of iterations until convergence
        """
        while (self.step_num < self.max_iters):
            previous_x = self.x
            self.step()
            if np.linalg.norm(self.x - previous_x , 2) < self.threshold:
                break
        return self.objective.__call__(self.x) , \
               self.x , \
               self.step_num

    def print_step(self) -> Tuple:
        """
        Executes a single Quasi-Newton step.
        Prints: 
            a tuple (next_x, next_d, step_size, next_inv_hessian) as follows:
            next_x : vector, updated x
            next_d : vector, updated direction
            step_size : scalar
            next_inv_hessian : matrix, approximator of the inverse Hessian matrix
        """
        next_x, next_d, step_size, next_inv_hessian = self.step()
        print ("next_step of BFGS optimizer leads to \nx = {0}\nnext_grad = {1}\nnext_d = {2}\nnext_step_size = {3}\nnext_inv_hessian =\n{4}".format(next_x,
                                                                                                                                                     self.objective.grad(self.x),
                                                                                                                                                     next_d,
                                                                                                                                                     step_size,
                                                                                                                                                     next_inv_hessian))
        
    def print_optimize(self) -> Tuple:
        """
        Execution of optimization flow
        Prints:
            fmin : scalar, objective function evaluated at the minimum
            minimizer : vector, the optimal 𝑥
            num_iters : scalar, number of iterations until convergence
        """
        f_min, minimizer, num_iters = self.optimize()
        print ("minimizer by BFGS optimizer leads to \nx_min = {0}\nf_min = {1}\nnum_iters = {2}".format(minimizer, f_min, num_iters))

class TotalVariationObjective:

    def __init__(
        self,
        src_img: np.ndarray,
        mu: float,
        eps: float
    ) -> None:
        """
        Initialize a total variation objective.
        Arguments:
            src_img : (n,m) matrix, input noisy image
            mu : regularization parameter, determines the weight of total
            variation term
            eps : small number for numerical stability
        """
        self.src_img = src_img
        self.mu = mu
        self.eps = eps

    def TV_part(
        self,
        img_reshaped: np.ndarray,
        row_index: int,
        col_index: int
    ) -> float:
        """
        Evaluate the objective for img.
        Arguments:
            img_reshaped : (n,m) matrix,
            indexes
        Return:
            TV_part : scalar, objective contribution to tv of [row_index, col_index]
        """
        part = self.eps
        if(row_index + 1 != img_reshaped.shape[0]):
            part += np.square(img_reshaped[row_index+1][col_index] - \
                              img_reshaped[row_index][col_index])
        if(col_index + 1 != img_reshaped.shape[1]):
            part += np.square(img_reshaped[row_index][col_index + 1] - \
                              img_reshaped[row_index][col_index])
        return np.sqrt(part)
    
    def TV(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the objective for img.
        Arguments:
            img : (n×m,) vector, denoised image
        Return:
            TV : scalar, objective's value
        """
        img_reshaped = np.reshape(img, (self.src_img.shape[0], -1))
        sum = 0
        for row_index in range(img_reshaped.shape[0]):
            for col_index in range(img_reshaped.shape[1]):
                sum += self.TV_part(img_reshaped, row_index, col_index)
        return sum

    def MSE(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the objective for img.
        Arguments:
            img : (n×m,) vector, denoised image
        Return:
            MSE : scalar, objective's value
        """
        return (np.square(img - self.src_img.flatten())).mean()
    
    def __call__(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the objective for img.
        Arguments:
            img : (n×m,) vector, denoised image
        Return:
            total_variation : scalar, objective's value
        """
        return self.MSE(img) + self.mu * self.TV(img)
    
    def print__call__(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Print the objective for img.
        Arguments:
            img : (n×m,) vector, denoised image
        Prints:
            total_variation : scalar, objective's value
        """
        print ("total_variation for image is {0}".format(self.__call__(img)))

    def calculate_grad_specific(
        self,
        img_reshaped: np.ndarray,
        row_index: int,
        col_index: int
    ) -> float:
        """
        Evaluate the gradient of the objective.
        Arguments:
            img_reshaped : (n,m) vector, denoised image,
            indexes
        Return:
            specific grad according to change in index [row_index, col index]
        """
        
        sum = (1.0 / img_reshaped.size) * (-2) * (self.src_img[row_index][col_index] - \
                                         img_reshaped[row_index][col_index])
        if (row_index != 0):
            sum += self.mu * \
                   (img_reshaped[row_index][col_index] - \
                    img_reshaped[row_index - 1][col_index]) / \
                   self.TV_part(img_reshaped, row_index - 1, col_index)
        if (col_index != 0):
            sum += self.mu * \
                   (img_reshaped[row_index][col_index] - \
                    img_reshaped[row_index][col_index - 1]) / \
                   self.TV_part(img_reshaped, row_index, col_index - 1)
        if(row_index + 1 != img_reshaped.shape[0]):
            sum -= self.mu * \
                   (img_reshaped[row_index + 1][col_index] - \
                    img_reshaped[row_index][col_index]) / \
                   self.TV_part(img_reshaped, row_index, col_index)
        if(col_index + 1 != img_reshaped.shape[1]):
            sum -= self.mu * \
                   (img_reshaped[row_index][col_index + 1] - \
                    img_reshaped[row_index][col_index]) / \
                   self.TV_part(img_reshaped, row_index, col_index)
        return sum
        
    def grad(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the gradient of the objective.
        Arguments:
            img : (n×m,) vector, denoised image
        Return:
            grad : (nxm,) vector, the objective's gradient
        """
        img_grad = np.empty_like(self.src_img)
        img_reshaped = np.reshape(img, (self.src_img.shape[0], -1))
        for row_index in range(self.src_img.shape[0]):
            for col_index in range(self.src_img.shape[1]):
                img_grad[row_index][col_index] = self.calculate_grad_specific(img_reshaped,
                                                                              row_index,
                                                                              col_index)
        return img_grad.flatten()
    
    def print_grad(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Print the gradient of the objective.
        Arguments:
            img : (n×m,) vector, denoised image
        Print:
            grad : (nxm,) vector, the objective's gradient
        """
        print ("grad for image is\n{0}".format(self.grad(img)))

def denoise_img(
    noisy_img: np.ndarray,
    B_0: np.ndarray,
    alpha_0: float,
    beta: float,
    sigma: float,
    threshold: float,
    max_iters: int,
    mu: float,
    eps: float
) -> Tuple:
    """
    Optimizes a Total Variantion objective using BFGS optimizer to
    denoise a noisy image.
    Arguments:
        noisy_img : (n,m) matrix, input noisy image
        B_0 : matrix , initial guess of the inverse Hessian
        alpha_0 : scalar, initial step size for Armijo line search
        beta : scalar, beta parameter of Armijo line search, a float in range (0,1)
        sigma : scalar, sigma parameter of Armijo line search, a float in range (0,1)
        threshold : scalar, stopping criteria (𝑥!"# − 𝑥!( < threshold, return 𝑥!"#
        max_iters : scalar, maximal number of iterations (stopping criteria)
        mu : regularization parameter, determines the weight of total variation term
        eps : small number for numerical stability
    Return:
        total_variation : loss at minimum
        img : (n,m) matrix, denoised image, values expected range is [0,1]
        num_iters : number of iterations until convergence
    """
    total_variation_objective_inst = TotalVariationObjective(noisy_img, mu, eps)
    bfgs_optimizer_inst = BFGSOptimizer(total_variation_objective_inst,
                                        noisy_img.flatten(),
                                        B_0,
                                        alpha_0,
                                        beta,
                                        sigma,
                                        threshold,
                                        max_iters)
    total_variation, img, num_iters = bfgs_optimizer_inst.optimize()
    img = img.reshape(noisy_img.shape[0],(-1))
    img = np.clip(img, 0, 1) # we might get out of bounds due to optimization
    return total_variation, img, num_iters

def show_grad_data(total_variation_objective_inst, pic_data, img_show_size):
    grad_data = total_variation_objective_inst.grad(pic_data.flatten())\
                                                  .reshape(pic_data.shape)
    grad_image_data = (grad_data - np.min(grad_data)) / (np.max(grad_data) - np.min(grad_data))
    Image.fromarray(grad_image_data * 255).resize((img_show_size,img_show_size)).show()

