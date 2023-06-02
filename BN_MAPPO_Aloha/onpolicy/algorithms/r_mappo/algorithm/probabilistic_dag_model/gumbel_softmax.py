import torch
import torch.nn.functional as F
from torch.autograd import Variable

class GumbleSoftmax(torch.nn.Module):
    def __init__(self, device = 'cpu', temp=1):
        super(GumbleSoftmax, self).__init__()
        self.device = device
        self.temp = temp
    
    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise).to(self.device)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor
    
    def gumbel_softmax_sample(self, logits, test):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        #dim = logits.size(-1)-1
        dim = 1
        if test == False:
            gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        else:
            gumble_samples_tensor = torch.zeros(logits.shape).cuda()
        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
        soft_samples = F.softmax(gumble_trick_log_prob_samples / self.temp, dim)
        return soft_samples
    
    def gumbel_softmax(self, logits, hard=False, test=False):
        """
        Sample from the Gumbel-Softmax distribution and optionally discretize.
            Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
            Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, test)
        if hard:
            _, max_value_indexes = y.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y
    
    def forward(self, logits, force_hard=False, test = False):
        
        if not force_hard:
            return self.gumbel_softmax(logits, hard=False, test = test)
        else:
            return self.gumbel_softmax(logits, hard=True, test = test)

