function A = LSR( X , Par )

% Input
% X             Data matrix, dim * num
% lambda        parameter, lambda>0

% Objective function:
%      min_{A}  ||X - X * A||_F + lambda * ||A||_F

% Notation: L
% X ... (L x N) data matrix, where L is the number of features, and
%           N is the number of samples.
% A ... (N x N) is a row structured sparse matrix used to select
%           the most representive and informative samples
% p ... (p=1, 2, inf) the norm of the regularization term
% lambda ... nonnegative regularization parameter

[L, N] = size (X);


if N < L
    XTXinv = (X' * X + Par.lambda * eye(N))\eye(N);
else
    P = (1/Par.lambda * eye(N) - (1/Par.lambda )^2 * X' / (1/Par.lambda * (X * X') + eye(L)) * X );
end

%% update C the coefficient matrix
if N < L
    A = XTXinv * (X' * X);
else
    A =  P * (X' * X);
end
