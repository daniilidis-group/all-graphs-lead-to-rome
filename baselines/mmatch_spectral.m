function [X,Y,run_time] = mmatch_spectral(W,dimGroup,k)

t0 = tic;

% display('Spectral method begins....')

k = min(k,size(W,1));

[V,~] = eigs(W,k,'la');
% [V,~] = svd(W);
% Y = rounding(V(:,1:k),dimGroup,0.5);
Y = abs(V(:,1:k));
Y = Y(:,1:min(size(Y,2),k));


% csdimGroup = [0;cumsum(dimGroup(:))];
% for i=1:numel(dimGroup)
%     
%    idx = csdimGroup(i)+1: csdimGroup(i+1);
%    Y(idx,:)= matrix2perm(Y(idx,:));
%    
% end

% X = single(Y)*single(Y)'>0;
X = max(0,single(Y)*single(Y)');

run_time = toc(t0);

% [display(sprintf('Spectral method terminated in %0.2f seconds...',run_time))
                        
                        
                
