function [weight_grad, bias_grad, out_sensitivity] = fullyconnect_backprop(in_sensitivity,  in, weight)
%The backpropagation process of fullyconnect
%   input parameter:
%       in_sensitivity  : the sensitivity from the upper layer, shape: 
%                       : [number of images, number of outputs in feedforward]
%       in              : the input in feedforward process, shape: 
%                       : [number of images, number of inputs in feedforward]
%       weight          : the weight matrix of this layer, shape: 
%                       : [number of inputs in feedforward, number of outputs in feedforward]
%
%   output parameter:
%       weight_grad     : the gradient of the weights, shape: 
%                       : [number of inputs in feedforward, number of outputs in feedforward]
%       out_sensitivity : the sensitivity to the lower layer, shape: 
%                       : [number of images, number of inputs in feedforward]
%
% Note : remember to divide by number of images in the calculation of gradients.

% TODO

[N_img, N_out] = size(in_sensitivity);
N_in = size(in,2);

weight_grad = zeros(N_in, N_out);
bias_grad = zeros(N_out, 1);
for i = 1:N_img
    weight_grad = weight_grad+in(i,:)'*in_sensitivity(i,:);
    bias_grad = bias_grad+in_sensitivity(i,:)'*1;
end
weight_grad = weight_grad/N_img;
bias_grad = bias_grad/N_img;

out_sensitivity = in_sensitivity*weight';

end

