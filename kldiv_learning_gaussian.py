import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def Gaussian1D(x, mu, sigma):
    """Implements 1D Gaussian using the following equation
     P(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\big[\frac{(x-\mu)^2}{\sigma^2}\big]},
    Args:
        x (np.array): x points to evaluate Gaussian on 
        mu (float): mean of the distribution
        sigma (float): standard deviation of the distribution

    Returns:
        np.array: distribution evaluated on x points
    """
    ATerm = 1/(sigma * np.sqrt(2 * np.pi))
    BTerm = np.exp(-0.5 * ((x-mu)/sigma) ** 2)
    return ATerm * BTerm

class KLDivUnimodalGaussianLoss:
    def forward(self, target, output):
        """
        Implements
        
        D_{KL}(P_{gt} \parallel P_{m}) = -\frac{1}{2} + \log \big(\frac{\sigma_{m}}{\sigma_{gt}}\big) + \frac{1}{2 \sigma_{m}^{2}} \big[ \sigma_{gt}^2 + (\mu_{gt}-\mu_{m})^2 \big]
        """

        return -0.5 + np.log(output[:, 1]/target[:, 1]) + (1/(2*(output[:, 1]**2))) * (target[:, 1]**2 + (target[:, 0] - output[:, 0])**2)
    

    def backward(self, target, output, grad_output):
        """
        Implements

        \partial D_KL/\partial \mu_m
        and
        \partial D_KL/\partial \sigma_m
        
        """
        deloutput = np.zeros_like(output)
        
        # \partial \mu
        deloutput[:, 0] = (output[:, 0] - target[:, 0])/(output[:, 1]**2)
        # \partial \sigma
        deloutput[:, 1] = 1/output[:, 1] - (1/output[:, 1]**3)*(target[:, 1]**2 + (target[:, 0]-output[:, 0])**2)

        return None, deloutput * grad_output

def optimise_gradient_descent(param_p, param_q, x_grid=np.linspace(-6, 6, 100), lr=0.01, save_interval=None, n_epochs=1000000):
    param_q_list = []
    loss_list = []

    # keep record of loss, to check termination
    loss_stack = deque(100*[0], 1000)

    kl = KLDivUnimodalGaussianLoss()

    for epoch in range(n_epochs):
        print('%d/%d' % (epoch, n_epochs))

        # do forward pass to calculate loss
        out = kl.forward(param_p, param_q)
        loss_stack.append(out[0])

        avg_loss = sum(loss_stack)/len(loss_stack)
        
        if avg_loss < 0.000000001:
            break
        
        # do backward pass to optimise params
        delout = kl.backward(param_p, param_q, 1)[1] 
        param_q = param_q - lr * delout
    
        if save_interval != None and epoch % save_interval == 0:
            param_q_list.append(param_q)
            loss_list.append(out[0])

    return param_q, param_q_list, loss_list

def optimise_gradient_descent_momentum(param_p, param_q, alpha=0.1, x_grid=np.linspace(-6, 6, 100), lr=0.01, save_interval=None, n_epochs=1000000):
    param_q_list = []
    loss_list = []

    # keep record of loss, to check termination
    loss_stack = deque(100*[0], 1000)

    kl = KLDivUnimodalGaussianLoss()

    prev_grad = 0

    for epoch in range(n_epochs):
        print('%d/%d' % (epoch, n_epochs))

        # do forward pass to calculate loss
        out = kl.forward(param_p, param_q)
        loss_stack.append(out[0])

        avg_loss = sum(loss_stack)/len(loss_stack)
        
        if avg_loss < 0.001:
            break
        
        # do backward pass to optimise params
        # implement momentum
        delout = alpha * kl.backward(param_p, param_q, 1)[1] + (1-alpha) * prev_grad
        param_q = param_q - lr * delout
        prev_grad = delout
    
        if save_interval != None and epoch % save_interval == 0:
            param_q_list.append(param_q)
            loss_list.append(out[0])

    return param_q, param_q_list, loss_list


def save_optimise_gif(param_p, param_q_list, loss_list, save_interval, x_grid, save_path):
    os.makedirs(save_path, exist_ok=True)

    # import packages for GIF generation
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    print('*' * 60)
    print('generating optimiser gif at: %s' % save_path)
    print('*' * 60)

    landscape = get_energy_landscape(param_p, x_grid)

    p = Gaussian1D(x_grid, param_p[0, 0], param_p[0, 1])

    fig = plt.figure(figsize=(8, 4))
    canvas = FigureCanvas(fig)
    
    fig_list = []
    for i, (param_q, loss) in enumerate(zip(param_q_list, loss_list)):

        q = Gaussian1D(x_grid, param_q[0, 0], param_q[0, 1])
        plt.subplot(1, 2, 1)
        plt.plot(x_grid, landscape)
        plt.plot(param_q[0, 0], loss, 'k.', markersize=12)
        plt.grid()
        canvas.draw()

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        plt.plot(x_grid, p, 'g')
        plt.plot(x_grid, q, 'r')
        plt.legend(['P(x)', 'Q(x)'])
        # plt.text(0.3, 0.9,'epoch: %d' % i * save_interval, ha='center', va='center', transform=ax.transAxes)


        plt.ylim([-0.01, 0.5])
        plt.grid()
        canvas.draw()

        plt.draw()
        
        # figure to image help from: https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        fig_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        fig_list.append(Image.fromarray(fig_image.reshape(canvas.get_width_height()[::-1] + (3,))).resize((600, 300), Image.BILINEAR))
        plt.clf()
        
    
    # gif help from: https://note.nkmk.me/en/python-pillow-gif/
    fig_list[0].save(os.path.join(save_path, 'optimiser.gif'),
               save_all=True, append_images=fig_list[1:], optimize=True, duration=80, loop=0)
    print('*' * 60)
    print(' ')

def get_energy_landscape(param_p, grid):
    kl = KLDivUnimodalGaussianLoss()

    param_q = param_p.copy()

    landscape = []
    for mu in grid:
        param_q[0, 0] = mu
        landscape.append(kl.forward(param_p, param_q))

    return landscape

if __name__ == '__main__':
    # saveGradientDescentOptim_Gif()
    
    param_p = np.zeros((1, 2))
    param_p[0, 0] = 0.0
    param_p[0, 1] = 1.0

    param_q = np.zeros((1, 2))
    param_q[0, 0] = 3#np.random.rand()*6-3
    param_q[0, 1] = 1.0
    
    x_grid = np.linspace(-6, 6, 100)

    save_interval=30

    param_q_opt, param_q_list, loss_list = optimise_gradient_descent(param_p, param_q, x_grid=x_grid, lr=0.01, save_interval=save_interval, n_epochs=100000)
    save_optimise_gif(param_p, param_q_list, loss_list, save_interval, x_grid, 'saved_figures/gradientdesc')
    
    param_q_opt, param_q_list, loss_list = optimise_gradient_descent_momentum(param_p, param_q, alpha=0.3, x_grid=x_grid, lr=0.01, save_interval=save_interval, n_epochs=100000)
    save_optimise_gif(param_p, param_q_list, loss_list, save_interval, x_grid, 'saved_figures/gradientdesc_mom')
