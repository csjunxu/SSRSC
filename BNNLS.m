function C = BNNLS( X , Par )
warning off;
% Input
% X           Data matrix, dim * num
% Par        parameters

% Objective function:
%      min_{A}  ||X - X * A||_F s.t.  0<=A<=Par.s

% Notation: L
% X ... (L x N) data matrix, where L is the number of features, and
%           N is the number of samples.
% A ... (N x N) is a row structured sparse matrix used to select
%           the most representive and informative samples
% Par ...  regularization parameters

[D, N] = size(X);
C = zeros(N, N);
LB = zeros(N, 1);
UB = Par.s*ones(N, 1);
for i = 1:N
    C(:,i) = lsqlin(X,X(:,i),[],[],[],[],LB,UB);
end