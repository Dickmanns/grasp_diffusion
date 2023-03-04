import torch
import torch.nn as nn



class GraspDiffusionFields(nn.Module):
    ''' Grasp DiffusionFields. SE(3) diffusion model to learn 6D grasp distributions. See
        SE(3)-DiffusionFields: Learning cost functions for joint grasp and motion optimization through diffusion
    '''
    def __init__(self, vision_encoder, geometry_encoder, points, feature_encoder, decoder):
        super().__init__()
        ## Register points to map H to points ##
        self.register_buffer('points', points)
        ## Vision Encoder. Map observation to visual latent code ##
        self.vision_encoder = vision_encoder
        ## vision latent code
        self.z = None
        ## Geometry Encoder. Map H to points ##
        self.geometry_encoder = geometry_encoder
        ## Feature Encoder. Get SDF and latent features ##
        self.feature_encoder = feature_encoder
        ## Decoder ##
        self.decoder = decoder

    def set_latent(self, O, batch = 1):
        self.z = self.vision_encoder(O.squeeze(1))
        self.z = self.z.unsqueeze(1).repeat(1, batch, 1).reshape(-1, self.z.shape[-1])
        #--------------print(self.z.shape, 'z, latent_vecs')

    def forward(self, H, k):
        ## 1. Represent H with points
        p = self.geometry_encoder(H, self.points)
        point=p.reshape(p.shape[0], -1, 3)
        #---------------------print(k.shape, 'k')
        #---------------------print(p.shape[2], 'p.shape[1]')
        #---------------------print(k.unsqueeze(2).shape, 'k.unsqueeze(1)')
        '''import pdb
        pdb.set_trace()'''
        k_ext = k[:,None].repeat(1,point.shape[1]) 
        '''k_ext1, k_ext2 = torch.tensor_split(k.unsqueeze(2), 2 ,dim=1)
        k_ext1 = k_ext1.reshape(-1,1)
        k_ext2 = k_ext2.reshape(-1,1)
        k_ext1 = k_ext1.repeat(1, p.shape[2])
        k_ext2 = k_ext2.repeat(1, p.shape[2])

        k_ext = torch.stack([k_ext1, k_ext2], dim=1)'''
        #---------------------print(self.z.shape, 'self.z')
        z_ext = self.z.unsqueeze(1).repeat(1, point.shape[1], 1)
        '''
        z_ext = z_ext.repeat(1, p.shape[2], 1)
        z_ext = torch.stack([z_ext, z_ext], dim=1)'''

        #z_ext1, z_ext2 = torch.tensor_split(k.unsqueeze(2), 2 ,dim=0)
        ## 2. Get Features
        psi = self.feature_encoder(point, k_ext, z_ext)

        '''psi1, psi2 = torch.tensor_split(psi, 2, dim=1)
        psi1 = psi1.reshape(-1,30,7)
        psi2 = psi2.reshape(-1,30,7)'''
        psi_flatten = psi.reshape(psi.shape[0], -1)
        '''psi_flatten1 = psi1.reshape(psi1.shape[0], -1)
        psi_flatten2 = psi2.reshape(psi2.shape[0], -1)'''

        #psi_flatten = torch.stack([psi_flatten1, psi_flatten2], dim=1)

        
        ## 3. Flat and get energy
        #---------------------print(psi_flatten.shape, 'psi_flatten')
        e = self.decoder(psi_flatten)
        return e

    def compute_sdf(self, x):
        k = torch.rand_like(x[..., 0])
        '''
        k ist die erste spalte von x aber mit komplett random zahlen zwishcen 0 und 1
        '''
        #---------------------print(k.shape, 'k, timesteps')
        psi = self.feature_encoder(x, k, self.z)
        #---------------------print(psi.shape, 'psi')
        return psi[..., 0]
