function [X] = myspectral(W,k)

[V, D] = eigs(W,k,'lm');
Y = V(:,1:k);
Dy = D(1:k,1:k);
X = abs(Y*Dy*Y');


end

