function [XfXft, Xf,run_time] = PGDDS(X_in,dimGroup,K,param)
% Projected Gradient Descent Doubly Stochastic PGSS

t0 = tic;
% display(sprintf('Distributed gradient descent begins.....'))

if nargin <3
    error('Not enough input arguments')  
end

param.nObj = numel(dimGroup);
param.K = K; 
param.dimGroup = dimGroup;



if ~isfield(param,'Adj')
    param.Adj = ones(param.nObj);
    param.Adj(1:size(param.Adj,1)+1:end) = 0;
end

if ~all(all( param.Adj'== param.Adj))
    error('Adjacency matrix not symmetric'); 
end
 
if ~isfield(param,'flagDebug')
    param.flagDebug=0; 
end

 
if  isfield(param,'maxiter')
    maxIter = param.maxiter; 
else
    maxIter = 200;
end

 
if any(any( triu(1-param.Adj,1)))
    param.flagfullGraph = 0;
else
     param.flagfullGraph = 1;
end

if ~isfield(param,'t')
    param.t= [1]; 
end

param.Adj = param.Adj>0;

%%  
%param.Akron = kron(param.Adj+speye(size(param.Adj)),speye(param.K)); % TODO: remove this one
param.csdimGroup = [0;cumsum(param.dimGroup(:))]; % is this necessary?

% random initialization
X0 = rand(param.csdimGroup(end),param.K);
X0 = X0 ./ sum(X0,2);

idx_diag = 1:size(X_in,1)+1:size(X_in,1)*size(X_in,2);
X_in(idx_diag) = 0;

idx_diag = 1:size(param.Adj,1)+1:size(param.Adj,1)*size(param.Adj,2);
param.Adj(idx_diag) = 0;
dmax  =  max( sum(param.Adj));
%% MAIN LOOP 
for il = 1:numel(param.t)
    
    X = X0;
    gamma_t = param.t(il)/(1+param.t(il));

    maxstep = 2/( 2*gamma_t + 3*(1-gamma_t)*dmax  );
    step = 0.99*maxstep;
    
   % max step size computation
   
   for iIter=1:maxIter
    
       % gradient computation
         gij = - X_in*X;

        XtXe2 = zeros( [param.K,param.K,param.nObj]);
        XXXtd = zeros(size(X));
    
        for i=1:param.nObj
            idxr =  param.csdimGroup(i)+1: param.csdimGroup(i+1);
            XtXe2(:,:,i) = X(idxr,:)'*X(idxr,:);
            XXXtd(idxr,:) = X(idxr,:)*XtXe2(:,:,i); 
        end
        gi =  XXXtd-X;
        
        XtXesum = zeros([param.K,param.K,param.nObj]);
        
        for i=1:param.nObj
            idxr =  param.csdimGroup(i)+1: param.csdimGroup(i+1);
            XtXesum(:,:,i) = sum(XtXe2(:,:,param.Adj(i,:)),3) + XtXe2(:,:,i);
            gij(idxr,:) = gij(idxr,:) +  X(idxr,:)*XtXesum(:,:,i);  
        end

        G =  gamma_t*gi  + (1- gamma_t)*gij;  % Euclidean gradient wrt to x parametrization
       
        
        
       % update step
       X = X - step*G;
       
   
       
       % projection step
        for iview =1:numel(param.dimGroup)
            idx = (param.csdimGroup(iview)+1):param.csdimGroup(iview+1);
                
            if size(X(idx,:),1) == size(X(idx,:),2)
                X(idx,:) = projectDADMM(X(idx,:));
            else
                X(idx,:) = projectPDADMM(X(idx,:));
            end
        end
        

         
   end
    
    
   
   
   
    % perturb a little bit to avoid undersirable stationary points
    for iview =1:numel(param.dimGroup)
        idx = (param.csdimGroup(iview)+1):param.csdimGroup(iview+1);
        X0(idx,:) = projectPDADMM(X(idx,:) +0.01*randn(size(X(idx,:))));
    end
    
end



%% truncation to partial permutation matrix
Xf = X;
 
%  for i=1:param.nObj
%     idxr =  param.csdimGroup(i)+1: param.csdimGroup(i+1);
%     [ass,~] = munkres(-Xf(idxr,:)); 
%     idx=1:param.dimGroup(i);
%     Xf(idxr,:) = sparse(idx(ass >0),ass(ass >0),1,param.dimGroup(i),param.K);  
%  end

 
XfXft = Xf*Xf';

run_time = toc(t0);

% display(sprintf('Distributed gradient descent terminated in %0.2f seconds...',run_time))
end

function X= projectDADMM(X0)


if size(X0,1)~= size(X0,2)
    error('Matrix not square')
end

k = size(X0,1);
rho = 1;
tol = k*10^(-4);

maxiter = 500;

U= zeros(size(X0));
Z= zeros(size(X0));

bf1= ones(k,1);


for i=1:maxiter

    % compute X
    B = (1/(1+rho))*X0 + (rho/(1+rho))*(Z-U);
    
    nu = -(1/k)*sum(B,1)';
    mu = (1/k)*(-sum(B,2) + bf1 -bf1*sum(nu)  );
    X = B+ mu*bf1' + bf1*nu';
  
    
    
    % update Z
    XpU  = X + U;
    
    Zprev = Z;
    Z = max( 0, XpU);
 
    % update X
    U = XpU-Z;
    
    % compute primal residual
    primal_res = X-Z;
    
    % compute dual residual
    dual_res = - rho*(Z-Zprev);
    
    % termination condition
     if norm(primal_res(:),2) < tol  &&   norm(dual_res(:),2) < tol
        break;    
     end

end


X = max(eps,X);
X = X ./ sum(X,2);

end

function X= projectPDADMM(X0)


if size(X0,1) > size(X0,2)
    error('Number of rows should not be greater than the number of columns')
end

 

k = size(X0,1); % # of rows 
m = size(X0,2); % # of columns


rho = 1;
tol = m*10^(-3);

maxiter = 200;

U= zeros(size(X0));
Z= zeros(size(X0));
t = zeros(m,1);
w = zeros(m,1);

 
lambda = k +((rho+1)/rho);
   
%pinvAA2 =[ (1/m)*( eye(k) + (rho/(1+rho))*ones(k))  -(1/m)*(rho/(1+rho))*ones(k,m); ...
%         -(1/m)*(rho/(1+rho))*ones(m,k)   (1/lambda)*eye(m)+(k*rho/(lambda*m*(1+rho)) )*ones(m)];
   
%AA = [ m*eye(k)         ones(k,m);...
% ones(m,k)       lambda*eye(m)] ;
%norm(inv(AA)-pinvAA2,'fro')
%pause(2)
     
b1k = ones(k,1);
b1m = ones(m,1);
  
for i=1:maxiter

    % compute X
    B =  X0 + rho*(Z-U);
    b = t-w; 
    
    if 0
    
       % bb = [ (1+rho)*b1k-sum(B,2); ... 
        %       (1+rho)*(b1m-b)-sum(B,1)'  ];


        %warning('This can be done more efficiently....')   
      %  yy = pinvAA2*bb;
      %  mu  = yy(1:k);
      %  nu = yy(k+1:end);
 
    else

        bb1 = (1+rho)*b1k-sum(B,2);
        bb2 = (1+rho)*(b1m-b)-sum(B,1)';
        sbb1 = sum(bb1);
        sbb2 = sum(bb2);
        
        mu = (1/m)*(  bb1 + (rho/(1+rho))*repmat(sbb1,[k 1])) ...
             -((1/m)*(rho/(1+rho)))*repmat(sbb2,[k 1]);

        nu =  -((1/m)*(rho/(1+rho)))*repmat(sbb1,[m 1]) + ...
             +(1/lambda)*bb2+(k*rho/(lambda*m*(1+rho)) )*repmat(sbb2,[m 1]);
    
    end
    
    % solve for X,S
    X = (1/(1+rho))*(B+ repmat(mu,[1 m]) + repmat(nu',[k 1]));
    s =(1/rho)*nu + b;
    
    
    % update Z
    XpU  = X + U;
    
    Zprev = Z;
    Z = max( 0, XpU);
    
    tprev = t;
    t = max(0,s+w);
    
    % update X
    U = XpU-Z;
    w = w+s-t;
    
    % compute primal residual
    primal_res = [X-Z;s'-t'] ;
    
    % compute dual residual
    dual_res = - rho*( [Z-Zprev;t'-tprev'] );
    
    % termination condition
     if norm(primal_res(:),2) < tol  &&   norm(dual_res(:),2) < tol
        break;    
     end

end


X = max(0,X);
X = X ./ sum(X,2);

end


