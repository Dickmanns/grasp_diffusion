import numpy as np
import torch
import os, os.path as osp

import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3


class ApproximatedGrasp_AnnealedLD():
    def __init__(self, model, device='cpu', batch=10, dim =3,
                 T=200, T_fit=5, deterministic=False):

        self.model = model
        self.device = device
        self.dim = dim
        self.shape = [4,4]
        self.batch = batch

        ## Langevin Dynamics evolution ##
        self.T = T
        self.T_fit = T_fit
        self.deterministic = deterministic

    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def _step(self, H0, t, noise_off=True):

        ## Phase
        eps = 1e-3
        phase = ((self.T - t)/self.T) + eps
        sigma_T = self._marginal_prob_std(eps)

        ## Move points to axis-angle ##
        xw0 = SO3_R3(R=H0[..., :3, :3] , t=H0[..., :3, -1]).log_map()

        ## Annealed Langevin Dynamics ##
        alpha = 1e-3
        sigma_i = self._marginal_prob_std(phase)
        ratio = sigma_i/sigma_T
        c_lr = alpha*ratio**2

        ## 1. Add Noise
        if noise_off:
            noise = torch.zeros_like(xw0)
        else:
            noise = torch.randn_like(xw0)

        noise = np.sqrt(c_lr)*noise

        xw01 = xw0 + np.sqrt(alpha)*ratio*noise


        ## 2. Compute gradient ##
        t_in = phase*torch.ones_like(xw01[...,0])
        xw01 = xw01.detach().requires_grad_(True)
        H_in = SO3_R3().exp_map(xw01).to_matrix()
        energy = self.model(H_in, t_in)
        grad_energy = torch.autograd.grad(energy.sum(), xw01, only_inputs=True)[0]

        ## 3. Evolve gradient ##
        delta_x = -.5*c_lr*grad_energy
        xw1 = xw01 + delta_x

        ## Build H ##
        H1 = SO3_R3().exp_map(xw1)
        return H1.to_matrix()

    def sample(self, save_path=False, batch=None):

        ## 1.Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        H0 = SO3_R3().sample(batch).to(self.device, torch.float32)

        ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
        Ht = H0
        if save_path:
            trj_H = Ht[None,...]
        for t in range(self.T):
            Ht = self._step(Ht, t, noise_off=self.deterministic)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)
        for t in range(self.T_fit):
            Ht = self._step(Ht, self.T, noise_off=True)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)

        if save_path:
            return Ht, trj_H
        else:
            return Ht


class Grasp_AnnealedLD():
    def __init__(self, model, device='cpu', batch=10, dim =3, k_steps=1,
                 T=200, T_fit=5, deterministic=False):

        self.model = model
        self.device = device
        self.dim = dim
        self.shape = [4,4]
        self.batch = batch

        ## Langevin Dynamics evolution ##
        self.T = T
        self.T_fit = T_fit
        self.k_steps = k_steps
        self.deterministic = deterministic

    def _marginal_prob_std(self, t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def _step(self, H0, t, noise_off=True):

        ## Phase
        noise_std = .5
        eps = 1e-3
        phase = ((self.T - t) / (self.T)) + eps
        sigma_T = self._marginal_prob_std(eps)

        ## Annealed Langevin Dynamics ##
        alpha = 1e-3
        sigma_i = self._marginal_prob_std(phase)
        ratio = sigma_i ** 2 / sigma_T ** 2
        c_lr = alpha * ratio
        if noise_off:
            c_lr = 0.003

        H1 = H0
        for k in range(self.k_steps):
            H1, H2 = torch.tensor_split(H1,2,dim=1)
            H1=H1.reshape(-1,4,4)
            H2=H2.reshape(-1,4,4)

            ## 1.Set input variable to Theseus ##
            H0_in = SO3_R3(R=H1[:,:3,:3], t=H1[:,:3, -1])
            phi0 = H0_in.log_map()

            H1_in = SO3_R3(R=H2[:,:3,:3], t=H2[:,:3, -1])
            phi1 = H1_in.log_map()

            ## 2. Compute energy gradient ##
            phi0_in = phi0.detach().requires_grad_(True)
            H_in0 = SO3_R3().exp_map(phi0_in).to_matrix()
            t_in = phase*torch.ones_like(H_in0[:,0,0])

            phi1_in = phi1.detach().requires_grad_(True)
            H_in1 = SO3_R3().exp_map(phi1_in).to_matrix()

            H_in = torch.stack([H_in0, H_in1], dim=1)

            


            e = self.model(H_in, t_in)
            d_phi0 = torch.autograd.grad(e.sum(), phi0_in, retain_graph=True)[0]
            d_phi1 = torch.autograd.grad(e.sum(), phi1_in, retain_graph=True)[0]

            #only_inputs=True, retain_graph=True, create_graph=True)

            ## 3. Compute noise vector ##
            if noise_off:
                noise = torch.zeros_like(phi0_in)
            else:
                noise = torch.randn_like(phi0_in)*noise_std

            ## 4. Compute translation ##
            delta0 = -c_lr/2*d_phi0 + np.sqrt(c_lr)*noise
            w_Delta0 = SO3().exp_map(delta0[:, 3:])
            t_delta0 = delta0[:, :3]

            delta1 = -c_lr/2*d_phi1 + np.sqrt(c_lr)*noise
            w_Delta1 = SO3().exp_map(delta1[:, 3:])
            t_delta1 = delta1[:, :3]

            ## 5. Move the points ##
            R1_out = th.compose(w_Delta0, H0_in.R)
            t1_out = H0_in.t + t_delta0

            R2_out = th.compose(w_Delta1, H1_in.R)
            t2_out = H1_in.t + t_delta1

            H1 = SO3_R3(R=R1_out, t=t1_out).to_matrix()
            H2 = SO3_R3(R=R2_out, t=t2_out).to_matrix()
            H1 = torch.stack([H1, H2], dim=1)


        return H1

    def sample(self, save_path=False, batch=None):

        ## 1.Sample initial SE(3) ##
        if batch is None:
            batch = self.batch
        H0 = SO3_R3().sample(batch).to(self.device, torch.float32)
        H1 = SO3_R3().sample(batch).to(self.device, torch.float32)


        ## 2.Langevin Dynamics (We evolve the data as [R3, SO(3)] pose)##
        Ht = torch.stack([H0,H1], dim=1)
        print(self.T)
        if save_path:
            trj_H = Ht[None,...]
        for t in range(self.T):
            Ht = self._step(Ht, t, noise_off=self.deterministic)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)
        for t in range(self.T_fit):
            Ht = self._step(Ht, self.T, noise_off=True)
            if save_path:
                trj_H = torch.cat((trj_H, Ht[None,:]), 0)

        if save_path:
            return Ht, trj_H
        else:
            return Ht



if __name__ == '__main__':
    import torch.nn as nn

    class model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, H, k):
            H_th = SO3_R3(R=H[:, :3, :3], t=H[:, :3, -1])
            x = H_th.log_map()
            return x.pow(2).sum(-1)

    ## 1. Approximated Grasp_AnnealedLD
    generator = ApproximatedGrasp_AnnealedLD(model(), T=100, T_fit=500)
    H = generator.sample()
    print(H)

    ## 2. Grasp_AnnealedLD
    generator = Grasp_AnnealedLD(model(), T=100, T_fit=500, k_steps=1)
    H = generator.sample()
    print(H)




