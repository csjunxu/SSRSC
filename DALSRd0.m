function C = DALSRd0( X , Par )

% Input
% X          Data matrix, dim * num
% Par        parameters

% Objective function:
%      min_{A}  ||X - X * A||_{F}^{2} + lambda * ||A||_{F}^{2}
%      s.t.  1'*A = s*1', diag(A) = 0

% Notation: L
% X ... (D x N) data matrix, where L is the dimension of features, and
%           N is the number of samples.
% A ... (N x N) is a column vector used to select
%           the most representive and informative samples
% Par ...  structure of the regularization parameters

[D , N] = size (X);

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
if N < D
    XTXinv = (X' * X + Par.rho/2 * eye(N))\eye(N);
else
    P = (2/Par.rho * eye(N) - (2/Par.rho)^2 * X' / (2/Par.rho * (X * X') + eye(D)) * X );
end
while  ( ~terminate )
    %% update A the coefficient matrix
    if N < D
        A = XTXinv * (X' * X + Par.rho/2 * C + 0.5 * Delta);
    else
        A =  P * (X' * X + Par.rho/2 * C + 0.5 * Delta);
    end
    A = A - diag(diag(A));
    
    %% update C the data term matrix
    Q = (Par.rho*A - Delta)/( Par.s*(2*Par.lambda+Par.rho) );
    C = -Par.s*solver_BCLS_closedForm(-Q);
    C = C - diag(diag(C));
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( C - A);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
    
    %% computing errors
    err1(iter+1) = errorCoef(C, A);
    err2(iter+1) = errorLinSys(X, A);
    if (  (err1(iter+1) <=tol && err2(iter+1)<=tol) ||  iter >= Par.maxIter  )
        terminate = true;
        %         fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
        %     else
        %         if (mod(iter, Par.maxIter)==0)
        %             fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
        %         end
    end
    
    %% next iteration number
    iter = iter + 1;
end
end
