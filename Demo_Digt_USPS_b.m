
clear ;

% load 'C:\Users\csjunxu\Desktop\SC\Datasets\YaleB_Crop.mat'  % load YaleB dataset
load 'C:\Users\csjunxu\Desktop\SC\Datasets\USPS_Crop.mat'   % load USPS dataset
% load 'C:\Users\csjunxu\Desktop\SC\Datasets\MNIST_Crop.mat' % load MNIST dataset
dataset = 'USPS';
writefilepath = ['C:/Users/csjunxu/Desktop/SC/Results/' dataset '/'];

Repeat = 1; %number of repeations
DR = 1; % perform dimension reduction or not
if DR == 0
    dim = size(Y{1, 1}, 1);
elseif DR == 1
    dim = 12;
else
    DR = 1;
    dim = 12;
end
%% Subspace segmentation methods
% SegmentationMethod = 'SSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2013 PAMI SSC');
SegmentationMethod = 'LRR' ; addpath('C:\Users\csjunxu\Desktop\SC\LRR ICML2010 NIPS2011 PAMI2013\code\');
% SegmentationMethod = 'LRSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2011 CVPR LRSC\');
% SegmentationMethod = 'LSR1' ; % 4.8
% SegmentationMethod = 'LSR2' ; % 4.6
% SegmentationMethod = 'LSR' ; % the same with LSR2
% SegmentationMethod = 'LSRd0' ;
% SegmentationMethod = 'SMR' ; addpath('C:\Users\csjunxu\Desktop\SC\SMR_v1.0');
% SegmentationMethod = 'SSCOMP' ;

%     SegmentationMethod = 'NNLSR' ;
%     SegmentationMethod = 'NNLSRd0' ;
%     SegmentationMethod = 'NPLSR' ;
%     SegmentationMethod = 'NPLSRd0' ;

    SegmentationMethod = 'ANNLSR' ;
%     SegmentationMethod = 'ANNLSRd0' ;
%     SegmentationMethod = 'ANPLSR' ;
%     SegmentationMethod = 'ANPLSRd0' ;

%% Subspace segmentation
for maxIter = [5]
    Par.maxIter = maxIter;
    for rho = [0.001]
        Par.rho = rho;
        for lambda = [0]
            Par.lambda =lambda*10^(-4);
            for nSet = [2 3 5 8 10]
                n = nSet;
                index = Ind{n};
                for i = 1:size(index,1)
                    fea = [];
                    gnd = [];
                    for p = 1:n
                        fea = [fea Y{index(i, p), 1}];
                        gnd= [gnd p * ones(1, length(S{index(i, p)}))];
                    end
                    [D, N] = size(fea);
                    fprintf( '%d: %d\n', size(index, 1), i ) ;
                    redDim = size(fea, 1);
                    if DR == 1
                        %% PCA Projection
                        [ eigvector , eigvalue ] = PCA( fea ) ;
                        maxDim = length(eigvalue);
                        fea = eigvector' * fea ;
                        redDim = min(n*dim, size(fea, 1)) ;
                    end
                    %% normalize
                    for c = 1 : size(fea,2)
                        fea(:,c) = fea(:,c) /norm(fea(:,c)) ;
                    end
                    %% Subspace Clustering
                    missrate = zeros(size(index, 1), Repeat) ;
                    fprintf( 'dimension = %d \n', redDim ) ;
                    Yfea = fea(1:redDim, :) ;
                    for j = 1 : Repeat
                        switch SegmentationMethod
                            case 'SSC'
                            alpha = Par.lambda;
                            C = admmLasso_mat_func(Yfea, true, alpha);
                        case 'LRR'
                            C = solve_lrr(Yfea, Par.lambda); % without post processing
                        case 'SMR'
                            para.aff_type = 'J1'; % J1 is unrelated to gamma, which is used in J2 and J2_norm
                            para.gamma = 1;
                            para.alpha = 20;
                            para.knn = 4;
                            para.elpson =0.01;
                            Yfea = [Yfea ; ones(1,size(ProjX,2))] ;
                            C = smr(Yfea, para);
                        case 'SSCOMP' % add the path of the SSCOMP method
                            addpath('C:\Users\csjunxu\Desktop\SC\SSCOMP_Code');
                            C = OMP_mat_func(Yfea, 9, 1e-6);
                            case 'LSR1'
                                C = LSR1( Yfea , Par.lambda ) ; % proposed by Lu
                            case 'LSR2'
                                C = LSR2( Yfea , Par.lambda ) ; % proposed by Lu
                            case 'LSR'
                                C = LSR( Yfea , Par ) ;
                            case 'LSRd0'
                                C = LSRd0( Yfea , Par ) ; % solved by ADMM
                            case 'NNLSR'                   % non-negative
                                C = NNLSR( Yfea , Par ) ;
                            case 'NNLSRd0'               % non-negative, diagonal = 0
                                C = NNLSRd0( Yfea , Par ) ;
                            case 'NPLSR'                   % non-positive
                                C = NPLSR( Yfea , Par ) ;
                            case 'NPLSRd0'               % non-positive, diagonal = 0
                                C = NPLSRd0( Yfea , Par ) ;
                            case 'ANNLSR'                 % affine, non-negative
                                C = ANNLSR( Yfea , Par ) ;
                            case 'ANNLSRd0'             % affine, non-negative, diagonal = 0
                                C = ANNLSRd0( Yfea , Par ) ;
                            case 'ANPLSR'                 % affine, non-positive
                                C = ANPLSR( Yfea , Par ) ;
                            case 'ANPLSRd0'             % affine, non-positive, diagonal = 0
                                C = ANPLSRd0( Yfea , Par ) ;
                        end
                        %% this normalization can be ignored
                        for k = 1 : size(C,2)
                            C(:, k) = C(:, k) / max(abs(C(:, k))) ;
                        end
                        Z = ( abs(C) + abs(C') ) / 2 ; % abs is useless in our model
                        idx = clu_ncut(Z,n) ;
                        missrate(i, j) = 1 - compacc(idx,gnd);
                        fprintf('%.3f%% \n' , missrate(i, j)*100) ;
                    end
                    missrateTot{n}(i) = mean(missrate(i, :)*100);
                    fprintf('Mean missrate of %d/%d is %.3f%%.\n ' , i, size(index, 1), missrateTot{n}(i)) ;
                end
                %% output
                avgmissrate(n) = mean(missrateTot{n});
                medmissrate(n) = median(missrateTot{n});
                fprintf('Total mean missrate  is %.3f%%.\n ' , avgmissrate(n)) ;
                if strcmp(SegmentationMethod, 'LSR')==1 || strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1
                    matname = sprintf([writefilepath dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'missrateTot','avgmissrate','medmissrate');
                else
                    matname = sprintf([writefilepath dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'missrateTot','avgmissrate','medmissrate');
                end
            end
        end
    end
end
