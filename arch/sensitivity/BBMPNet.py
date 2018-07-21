from torch.autograd import Variable

from .BDNet import GernarativeBDNet


class BBMP_SSR_Generative(GernarativeBDNet):
    def initialize(self, train_loader, epochs):
        super(BBMP_SSR_Generative, self).initialize(train_loader, epochs)
        self.logit_p.data.fill_(1.) # Initialize the mask as 1

    def sampled_from_logit_p(self, num_samples):
        assert num_samples == 1, 'Only use 1 sample of mask'
        expanded_logit_p = self.logit_p.unsqueeze(0)
        if self.upsample is not None:
            expanded_logit_p = self.upsample(expanded_logit_p)

        return expanded_logit_p

    def sampled_noisy_inputs(self, inputs):
        bern_val = self.sampled_from_logit_p(inputs.size(0))

        # Define mask as the non-1 part!
        the_mask = (bern_val.data >= 1.).float()

        background = self.generative_model.generate_background(inputs.data, the_mask)
        noised_input = inputs * bern_val + (1. - bern_val) * Variable(background)
        return noised_input

    def forward(self, inputs):
        # Clip the value between 0 or 1
        self.logit_p.data.clamp_(0, 1)

        noised_inputs = self.sampled_noisy_inputs(inputs)
        return self.trained_classifier(noised_inputs)

    def get_l2_loss(self):
        # Add L2 support loss here
        if self.tv_coef == 0:
            return 0.

        expand_p = self.sampled_from_logit_p(1)

        def square(x):
            return x * x

        l2_loss = square(expand_p[0, :, :-1, :] - expand_p[0, :, 1:, :]).sum()
        l2_loss += square(expand_p[0, :, :, :-1] - expand_p[0, :, :, 1:]).sum()
        return self.tv_coef * l2_loss

    def sgvloss(self, outputs, targets, rw=1.0):
        avg_pred_loss = self.loss_criteria(outputs, targets)
        reg_loss = self.reg_coef * (self.eval_reg().sum()) # l1 loss only

        reg_loss += self.get_l2_loss()
        return avg_pred_loss + rw * reg_loss, avg_pred_loss, rw * reg_loss

    def eval_reg(self):
        expand_p = self.sampled_from_logit_p(1)
        return expand_p

    def get_importance_vector(self):
        self.logit_p.data.clamp_(0, 1)

        expand_p = self.sampled_from_logit_p(1)
        return expand_p.data[0, 0, ...]


class BBMP_SDR_Generative(BBMP_SSR_Generative):
    def sgvloss(self, outputs, targets, rw=1.0):
        '''
        Change two things. First the pred loss. Second the sparsity.
        '''
        avg_pred_loss = -self.loss_criteria(outputs, targets)
        reg_loss = self.reg_coef * ((1. - self.sampled_from_logit_p(1)).sum())

        reg_loss += self.get_l2_loss()
        return avg_pred_loss + rw * reg_loss, avg_pred_loss, rw * reg_loss

    def get_importance_vector(self):
        return 1. - super(BBMP_SDR_Generative, self).get_importance_vector()
