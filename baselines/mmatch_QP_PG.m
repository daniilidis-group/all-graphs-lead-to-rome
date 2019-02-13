function Y = mmatch_QP_PG(Q,alpha,beta,nP,Y)

tol = 1e-3;
n = size(Q,1);
Q = - Q + beta;
nP = cumsum(nP);
nP = [0,nP];
concavify = false;

if concavify
    if n < 1000
        d = eig(Q);
        w = max(d,0);
        d = d - w; % all <= 0
        Q = Q - diag(w);
        W = w/2*ones(1,size(Y,2));
        mu = 1.1*max(abs(d));
    else
        d = sum(abs(Q),2)-abs(diag(Q)); % off-diagnal sum
        w = d + diag(Q);
        Q = Q - diag(w);
        W = w/2*ones(1,size(Y,2));
        mu = 1.1*norm(Q);
    end
else
    mu = 1.1*norm(Q);
    W = 0;
end

for iter = 1:200
    
    Y0 = Y;
    Y = Y - (Q*Y+W+alpha*Y)/mu;
    
    for i = 1:length(nP)-1
        ind = nP(i)+1:nP(i+1);
        Y(ind,:) = proj2dpam(Y(ind,:),1e-2);
    end
    
    RelChg = norm(Y(:)-Y0(:))/n;
    fprintf('Iter = %d, Res = (%d), mu = %d\n',iter,RelChg,mu);
    
    if  RelChg < tol 
        break
    end
    
end

end






