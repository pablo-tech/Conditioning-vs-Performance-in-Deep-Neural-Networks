import torch.autograd

class FudgeGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # just a pass through bro
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors

        grad_input = None
        grad_input = grad_output
        noise = torch.autograd.Variable(grad_input.data.new(grad_input.size()).normal_(0, grad_input.std() * 2))
        # print(grad_input)
        # print(noise)
        return grad_input