import torch as tr
import torch.nn as nn
from torch import autograd


class mmd2_noise_injection(autograd.Function):

    @staticmethod
    def forward(ctx,true_feature,fake_feature,noisy_feature):
        b_size,d, n_particles = noisy_feature.shape
        with  tr.enable_grad():

            mmd2 = tr.mean((true_feature-fake_feature)**2)
            mean_noisy_feature = tr.mean(noisy_feature,dim = -1 )

            mmd2_for_grad = (n_particles/b_size)*(tr.einsum('nd,nd->',fake_feature,mean_noisy_feature) - tr.einsum('nd,nd->',true_feature,mean_noisy_feature))

        ctx.save_for_backward(mmd2_for_grad,noisy_feature)

        return mmd2

    @staticmethod
    def backward(ctx, grad_output):
        mmd2_for_grad, noisy_feature = ctx.saved_tensors
        with  tr.enable_grad():
            gradients = autograd.grad(outputs=mmd2_for_grad, inputs=noisy_feature,
                        grad_outputs=grad_output,
                        create_graph=True, only_inputs=True)[0] 
                
        return None, None, gradients


class mmd2_func(autograd.Function):

    @staticmethod
    def forward(ctx,true_feature,fake_feature):

        b_size,d, n_particles = fake_feature.shape

        with  tr.enable_grad():

            mmd2 = (n_particles/b_size)*tr.sum((true_feature-tr.mean(fake_feature,dim=-1))**2)

        ctx.save_for_backward(mmd2,fake_feature)

        return (1./n_particles)*mmd2

    @staticmethod
    def backward(ctx, grad_output):

        mmd2, fake_feature = ctx.saved_tensors
        with  tr.enable_grad():
            gradients = autograd.grad(outputs=mmd2, inputs=fake_feature,
                        grad_outputs=grad_output,
                        create_graph=True, only_inputs=True)[0] 
                
        return None, gradients


class chard_func(autograd.Function):

    @staticmethod
    def forward(ctx,true_feature,fake_feature, noisy_fake_feature, lmbda):
        b_size, d, n_particles = fake_feature.shape
        m_particles = 1
        true_feature = true_feature[:, :, None]
        fake_feature = fake_feature.clone().detach()
        with  tr.enable_grad():
            # This is MMD, for debugging purposes
            # K_XX = tr.einsum('ijk,ijl->jkl', true_feature, true_feature).sum(0) / b_size
            # K_YX = tr.einsum('ijk,ijl->jkl', fake_feature, true_feature).sum(0) / b_size
            # K_YY = tr.einsum('ijk,ijl->jkl', fake_feature, fake_feature).sum(0) / b_size

            # chard = 0.5 * (K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()) 

            # K_YY_ = tr.einsum('ijk,ijl->jkl', fake_feature, fake_feature.clone().detach()).sum(0) / b_size

            # chard_first_variation = (K_YY_.mean() - K_YX.mean()) * n_particles

            K_XX = tr.einsum('ijk,ijl->jkl', true_feature, true_feature).sum(0) / b_size
            K_YX = tr.einsum('ijk,ijl->jkl', fake_feature, true_feature).sum(0) / b_size
            K_YY = tr.einsum('ijk,ijl->jkl', fake_feature, fake_feature).sum(0) / b_size
            inv_K_XX = tr.linalg.inv(K_XX + m_particles * lmbda * tr.eye(K_XX.shape[0]).to(fake_feature.device))
            part1 = K_YY.mean() + K_XX.mean() - 2 * K_YX.mean()
            part2 = -(K_YX @ inv_K_XX @ K_YX.T).mean()
            part3 = (K_XX.T @ inv_K_XX @ K_YX.T).mean() * 2
            part4 = -(K_XX.T @ inv_K_XX @ K_XX).mean()
            chard = 0.5 * (part1 + part2 + part3 + part4) * (1 + lmbda) / lmbda

            K_noisyY_X = tr.einsum('ijk,ijl->jkl', noisy_fake_feature, true_feature).sum(0) / b_size
            K_noisyY_Y = tr.einsum('ijk,ijl->jkl', noisy_fake_feature, fake_feature).sum(0) / b_size

            part1 = K_noisyY_Y.mean() - K_noisyY_X.mean()
            part2 = - (K_noisyY_X @ inv_K_XX @ K_YX.T).mean()
            part3 = (K_noisyY_X @ inv_K_XX @ K_XX).mean()
            chard_first_variation = (part1 + part2 + part3) / lmbda * (1 + lmbda)
            chard_first_variation = chard_first_variation * n_particles

        ctx.save_for_backward(chard_first_variation, noisy_fake_feature)
        return chard

    @staticmethod
    def backward(ctx, grad_output):

        chard_first_variation, noisy_fake_feature = ctx.saved_tensors
        with  tr.enable_grad():
            gradients = autograd.grad(outputs=chard_first_variation, inputs=noisy_fake_feature,
                        grad_outputs=grad_output,
                        create_graph=True, only_inputs=True)[0] 
        return None, None, gradients, None


class sobolev(autograd.Function):
    @staticmethod
    def forward(ctx,true_feature,fake_feature,matrix):

        b_size,_, n_particles = fake_feature.shape

        m = tr.mean(fake_feature,dim=-1) -  true_feature

        alpha = tr.solve(m,matrix)[0].clone().detach()

        with  tr.enable_grad():

            mmd2 = (0.5*n_particles/b_size)*tr.sum((true_feature-tr.mean(fake_feature,dim=-1))**2)
            mmd2_for_grad = (1./b_size)*tr.einsum('id,idm->',alpha,fake_feature)
        
        ctx.save_for_backward(mmd2_for_grad,fake_feature)

        return (1./n_particles)*mmd2

    @staticmethod
    def backward(ctx, grad_output):
        mmd2, fake_feature = ctx.saved_tensors
        with  tr.enable_grad():
            gradients = autograd.grad(outputs=mmd2, inputs=fake_feature,
                        grad_outputs=grad_output,
                        create_graph=True, only_inputs=True)[0] 
                
        return None, gradients,None

class CHARD(nn.Module):
    def __init__(self,student,with_noise,lmbda):
        super(CHARD, self).__init__()
        self.student = student
        self.chard = chard_func.apply
        self.with_noise=with_noise
        self.lmbda = lmbda
    def forward(self,x,y):
        out = self.student(x)
        self.student.set_noisy_mode(self.with_noise)
        noisy_out = self.student(x)
        loss = self.chard(y, out, noisy_out, self.lmbda)
        return loss
    
class MMD(nn.Module):
    def __init__(self,student,with_noise):
        super(MMD, self).__init__()
        self.student = student
        self.mmd2 = mmd2_noise_injection.apply
        self.with_noise=with_noise
    def forward(self,x,y):
        if self.with_noise:
            out = tr.mean(self.student(x),dim = -1).clone().detach()
            self.student.set_noisy_mode(True)
            noisy_out = self.student(x)
            loss = 0.5*self.mmd2(y,out,noisy_out)
        else:
            out = tr.mean(self.student(x),dim = -1).clone().detach()
            self.student.set_noisy_mode(False)
            noisy_out = self.student(x)
            loss = 0.5*self.mmd2(y,out,noisy_out)
        return loss

class MMD_Diffusion(nn.Module):
    def __init__(self,student):
        super(MMD_Diffusion, self).__init__()
        self.student = student
        self.mmd2 = mmd2_func.apply
    def forward(self,x,y):
        self.student.add_noise()
        noisy_out = self.student(x)
        
        loss = 0.5*self.mmd2(y,noisy_out)
        return loss

class Sobolev(nn.Module):
    def __init__(self,student):
        super(Sobolev, self).__init__()
        self.student = student
        self.sobolev = sobolev.apply
        self.lmbda = 1e-6
    def forward(self,x,y):
        self.student.zero_grad()
        out = self.student(x)
        b_size,_,num_particles = out.shape
        grad_out = compute_grad(self.student,x)
        matrix = (1./(num_particles*b_size))*tr.einsum('im,jm->ij',grad_out,grad_out)+self.lmbda*tr.eye(b_size, dtype= x.dtype, device=x.device)
        matrix = matrix.clone().detach()
        loss = self.sobolev(y,out,matrix)
        return loss

def compute_grad(net,x):
    J = []
    F = net(x)
    F = tr.einsum('idm->i',F)
    b_size = F.shape[0]
    for i in range(b_size):
        if i==b_size-1:
            grads =  autograd.grad(F[i], net.parameters(),retain_graph=False)
        else:
            grads =  autograd.grad(F[i], net.parameters(),retain_graph=True)
        grads = [x.view(-1) for x in grads]
        grads = tr.cat(grads)
        J.append(grads)

    return tr.stack(J,dim=0)

