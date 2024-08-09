import torch
import torch.nn as nn

class SpacedRepetition(nn.Module):
    
    def __init__(self, input_dim, alpha, lambda_reg):

        super(SpacedRepetition, self).__init__()
        self.theta = nn.Linear(input_dim, 1, bias=False)
        self.alpha = alpha
        self.lambda_reg = lambda_reg

    def half_life(self, x):
        theta_x = self.theta(x)
        h_hat = torch.pow(2, theta_x)
        return h_hat

    def p_recall(self, h_hat, delta):
        p_hat = torch.pow(2, -1 * (delta / h_hat.squeeze()))
        return p_hat
    
    def predict(self, x, delta):
        h_hat = self.half_life(x)
        p_hat = self.p_recall(h_hat,delta)
        return(p_hat)


    def loss(self,p, p_hat, h, h_hat):
        loss_p = torch.sum((p_hat - p) ** 2)
        loss_h = torch.sum((h_hat - h) ** 2)
        reg_term = self.lambda_reg * torch.sum(self.theta.weight ** 2)
        total_loss = loss_p + self.alpha * loss_h + reg_term
        return total_loss
