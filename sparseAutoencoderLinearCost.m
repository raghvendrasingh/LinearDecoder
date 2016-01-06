function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------  
p=sparsityParam;
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

 %forward pass  
      a2=sigmoid(W1*data+repmat(b1,1,size(data,2)));
      p_hat=(1/size(data,2)).* sum(a2,2);
      a3=W2*a2+repmat(b2,1,size(data,2));
      
      cost=(1/(2*size(data,2)))*sum( sum ( (a3-data).*(a3-data) ) ) +... 
           (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2))) +... 
           beta * ( p * sum(log(p./p_hat)) + (1-p)* sum( log( (1-p)./(1-p_hat) ) ) );        


    %backward pass  
      delta_3= (-1.*(data-a3));
      a=(W2'*delta_3);
      b=((-1*p)./p_hat);
      c=( (1-p)./(1-p_hat) );
      d=(beta .* (b+c));
      e=repmat(d,1,size(data,2));
      delta_2= ( a + e).* ( a2 .* (1-a2));
      del_W2=delta_3 * a2';
      del_b2=sum(delta_3,2);
      del_W1=delta_2 * data';
      del_b1=sum(delta_2,2);
      

W1grad = (1/size(data,2)).* del_W1 + lambda .* W1;
W2grad = (1/size(data,2)).* del_W2 + lambda .* W2;
b1grad = (1/size(data,2)).* del_b1;
b2grad = (1/size(data,2)).* del_b2;

%cost=(1/(2*size(data,2)))*cost + (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2))) + beta * ( sparsityParam * sum(log(sparsityParam./p_hat)) + (1-sparsityParam)* sum( log( (1-sparsityParam)./(1-p_hat) ) )   );        


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
     sigm = 1 ./ (1 + exp(-x));
end
