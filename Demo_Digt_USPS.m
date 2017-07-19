clear ;
dataset = 'USPS';
writefilepath = ['C:/Users/csjunxu/Desktop/SC/Results/' dataset '/'];
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
% SegmentationMethod = 'ANNLSR' ;
%     SegmentationMethod = 'ANNLSRd0' ;
%     SegmentationMethod = 'ANPLSR' ;
%     SegmentationMethod = 'ANPLSRd0' ;
%% Settings
for nSample = [200] % number of images for each digit
    load 'C:\Users\csjunxu\Desktop\SC\Datasets\USPS_Crop.mat'   % load USPS dataset
    nExperiment = 20; % number of repeations
    DR = 1; % perform dimension reduction or not
    if DR == 0
        dim = size(Y{1, 1}, 1);
    elseif DR == 1
        dim = 10;
    else
        DR = 1;
        dim = 10;
    end
    %% Subspace segmentation
    for maxIter = [5]
        Par.maxIter = maxIter;
        for rho = [1]
            Par.rho = rho;
            for lambda = [.1:.1:.9 1:.5:9 10:1:20]
                Par.lambda =lambda*10^(-0);
                missrate = zeros(nExperiment, 1) ;
                for i = 1:nExperiment
                    nCluster = 10;
                    digit_set = 0:9; % set of digits to test on, e.g. [2, 0]. Pick randomly if empty.
                    % prepare data
                    if isempty(digit_set)
                        rng(i); Digits = randperm(10, nCluster) - 1;
                    else
                        Digits = digit_set;
                    end
                    if length(nSample) == 1
                        nSample = ones(1, nCluster) * nSample;
                    end
                    fea = zeros(size(Y{1}, 1), sum(nSample));
                    gnd = zeros(1, sum(nSample));
                    nSample_cum = [0, cumsum(nSample)];
                    for iK = 1:nCluster % randomly take data for each digit.
                        [d, N] = size(Y{iK});
                        rng( (i-1) * nCluster + iK );
                        Select = randperm(N, nSample(iK));
                        fea(:, nSample_cum(iK) + 1 : nSample_cum(iK+1)) = Y{iK, 1}(:, Select);
                        gnd( nSample_cum(iK) + 1 : nSample_cum(iK+1) ) = iK * ones(1, nSample(iK));
                    end
                    %                     [D, N] = size(fea);
                    redDim = size(fea, 1);
                    if DR == 1
                        %% PCA Projection
                        [ eigvector , eigvalue ] = PCA( fea ) ;
                        maxDim = length(eigvalue);
                        fea = eigvector' * fea ;
                        redDim = min(nCluster*dim, size(fea, 1)) ;
                    end
                    %% normalize
                    for c = 1 : size(fea,2)
                        fea(:,c) = fea(:,c) /norm(fea(:,c)) ;
                    end
                    %% Subspace Clustering
                    fprintf( 'dimension = %d \n', redDim ) ;
                    Yfea = fea(1:redDim, :) ;
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
                    %% generate affinity
                    for k = 1 : size(C, 2)
                        C(:, k) = C(:, k) / max(abs(C(:, k))) ;
                    end
                    Z = ( abs(C) + abs(C') ) / 2 ; % abs is useless in our model
                    %% generate label
                    idx = clu_ncut(Z, nCluster) ;
                    %% Evaluation
                    missrate(i) = 1 - compacc(idx, gnd);
                    fprintf('%d: %.3f%% \n' , i, missrate(i)*100) ;
                end
                %% output
                avgmissrate = mean(missrate*100);
                medmissrate = median(missrate*100);
                fprintf('Total mean missrate  is %.3f%%.\n' , avgmissrate) ;
                if strcmp(SegmentationMethod, 'SSC')==1 || strcmp(SegmentationMethod, 'LRR')==1 || strcmp(SegmentationMethod, 'LRSC')==1 || strcmp(SegmentationMethod, 'LSR')==1 || strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1 || strcmp(SegmentationMethod, 'SMR')==1 %|| strcmp(SegmentationMethod, 'SSCOMP')==1
                    matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'missrate','avgmissrate','medmissrate');
                elseif strcmp(SegmentationMethod, 'NNLSR') == 1 || strcmp(SegmentationMethod, 'NPLSR') == 1 || strcmp(SegmentationMethod, 'ANNLSR') == 1 || strcmp(SegmentationMethod, 'ANPLSR') == 1
                    matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'missrate','avgmissrate','medmissrate');
                elseif strcmp(SegmentationMethod, 'SSCOMP')==1
                    matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_K' num2str(Par.rho) '_thr' num2str(Par.lambda) '.mat']);
                    save(matname,'missrate','avgmissrate','medmissrate');
                end
            end
        end
    end
end
