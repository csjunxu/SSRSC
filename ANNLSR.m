function C = ANNLSR( X , Par )

% Input:
% X ... (L x N) data matrix, where L is the number of features, and
%           N is the number of samples.
% Par ...  regularization parameters

% Objective function:
%      min_{A}  ||X - X * A||_F^2 + lambda * ||A||_F^2 s.t.  A>=0, 1'*A=1'

% Output: 
% A ... (N x N) is a coefficient matrix 

[L, N] = size (X);

%% initialization

% A       = eye (N);
% A   = rand (N);
A       = zeros (N, N);
C       = A;
Delta = C - A;

%%
tol   = 1e-4;
iter    = 1;
% objErr = zeros(Par.maxIter, 1);
err1(1) = inf; err2(1) = inf;
terminate = false;
if N < L
    XTXinv = (X' * X + Par.rho/2 * eye(N))\eye(N);
else
    P = (2/Par.rho * eye(N) - (2/Par.rho)^2 * X' / (2/Par.rho * (X * X') + eye(L)) * X );
end
while  ( ~terminate )
    %% update A the coefficient matrix
    if N < L
        A = XTXinv * (X' * X + Par.rho/2 * C + 0.5 * Delta);
    else
        A =  P * (X' * X + Par.rho/2 * C + 0.5 * Delta);
    end
    
    %% update C the data term matrix
    Q = (Par.rho*A - Delta)/(2*Par.lambda+Par.rho);
    C  = solver_BCLS_closedForm(Q);
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( C - A);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
    
    %% computing errors
    err1(iter+1) = errorCoef(C, A);
    err2(iter+1) = errorLinSys(X, A);
    if (  (err1(iter+1) >= err1(iter) && err2(iter+1)<=tol) ||  iter >= Par.maxIter  )
        terminate = true;
%                 fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
    else
%                 if (mod(iter, Par.maxIter)==0)
%                     fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
%                 end
    end
    
    %         %% convergence conditions
    %     objErr(iter) = norm( X - X*C, 'fro' ) + Par.lambda * norm(C, Par.p);
    %     fprintf('[%d] : objErr: %f \n', iter, objErr(iter));
    %     if ( iter>=2 && mod(iter, 10) == 0 || stopCC < tol)
    %         stopCC = max(max(abs(objErr(iter) - objErr(iter-1))));
    %         disp(['iter ' num2str(iter) ',stopADMM=' num2str(stopCC,'%2.6e')]);
    %         if stopCC < tol
    %             break;
    %         end
    %     end
    %% next iteration number
    iter = iter + 1;
end
end
