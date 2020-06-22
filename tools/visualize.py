# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt

viz = Visdom()


def init_visdom(title_a='Main Loss', title_b='Auxilliary Loss'):
    window = viz.line(X=np.arange(0, 1), Y=np.reshape(0, 1), opts=dict(title=title_a))
    windowb = viz.line(X=np.arange(0, 1), Y=np.reshape(0, 1), opts=dict(title=title_b))
    return window, windowb


def plot_grad_flow(named_parameters, id):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(id + '.png', dpi=300)

def plot_representation(coeff, y_label, freqs=[], id='tt.png'):
    plt.figure()
    plt.imshow(coeff, aspect='auto', origin='lower', vmin=0., vmax=1., extent=[0, 345, 0, 800])
    plt.xlabel('Time frames ' + r'$T$', fontsize=20)

    if freqs.any():
        locs = np.arange(0, 800, 75)
        plt.yticks(locs, (np.round(freqs[locs], 0).astype(int)), rotation=45, fontsize=16)

    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(rotation=45, fontsize=16)
    plt.ylabel(y_label, fontsize=20)
    plt.minorticks_on()
    cbar = plt.colorbar(extend='both')
    cbar.ax.tick_params(labelsize=16, rotation=45)
    cbar.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(id, dpi=350)
    plt.close()

    
# EOF
