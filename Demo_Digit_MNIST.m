clear;

addpath('MNISThelpcode');
addpath(genpath('C:\Users\csjunxu\Desktop\SC\SSCOMP_Code\scatnet-0.2'));
dataset = 'MNIST';
write_results_dir = ['C:/Users/csjunxu/Desktop/SC/Results/' dataset '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end

%% Subspace segmentation methods
% SegmentationMethod = 'SSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2013 PAMI SSC');
% SegmentationMethod = 'LRR' ; addpath('C:\Users\csjunxu\Desktop\SC\LRR ICML2010 NIPS2011 PAMI2013\code\');
% SegmentationMethod = 'LRSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2011 CVPR LRSC\');
% SegmentationMethod = 'LSR1' ;
% SegmentationMethod = 'LSR2' ;
% SegmentationMethod = 'LSR' ; % the same with LSR2
% SegmentationMethod = 'LSRd0' ; % the same with LSR1
% SegmentationMethod = 'SMR' ; addpath('C:\Users\csjunxu\Desktop\SC\SMR_v1.0');
% SegmentationMethod = 'S3C' ; addpath('C:\Users\csjunxu\Desktop\SC\2015 CVPR S3C');
% SegmentationMethod = 'RSIM' ; addpath('C:\Users\csjunxu\Desktop\SC\Ncut_9');addpath('C:\Users\csjunxu\Desktop\SC\2015 ICCV RSIM');
% SegmentationMethod = 'SSCOMP' ;

% SegmentationMethod = 'NNLSR' ;
% SegmentationMethod = 'NNLSRd0' ;
% SegmentationMethod = 'NPLSR' ;
% SegmentationMethod = 'NPLSRd0' ;

% SegmentationMethod = 'ANNLSR' ;
% SegmentationMethod = 'ANNLSRd0' ;
% SegmentationMethod = 'ANPLSR' ;
% SegmentationMethod = 'ANPLSRd0' ;

% SegmentationMethod = 'DANNLSR' ;
% SegmentationMethod = 'DANNLSRd0' ;
SegmentationMethod = 'DANPLSR' ;
% SegmentationMethod = 'DANPLSRd0' ;

%% Settings
for nSample = [100 200 400 600] % number of images for each digit
    %% Load data
    addpath('C:\Users\csjunxu\Desktop\SC\Datasets\MNIST\')
    if ~exist('MNIST_DATA', 'var')
        try
            % MNIST_SC_DATA is a D by N matrix. Each column contains a feature
            % vector of a digit image and N = 60,000.
            % MNIST_LABEL is a 1 by N vector. Each entry is the label for the
            % corresponding column in MNIST_SC_DATA.
            load MNIST_SC.mat MNIST_SC_DATA MNIST_LABEL;
        catch
            MNIST_DATA = loadMNISTImages('train-images.idx3-ubyte');
            MNIST_LABEL = loadMNISTLabels('train-labels.idx1-ubyte');
            MNIST_SC_DATA = SCofDigits(MNIST_DATA);
            save C:\Users\csjunxu\Desktop\SC\Datasets\MNIST_SC.mat MNIST_SC_DATA MNIST_LABEL;
        end
        MNIST_DATA = MNIST_SC_DATA;
    end
    
    nExperiment = 20; % number of repeations
    DR = 1; % perform dimension reduction or not
    if DR == 0
        dim = size(Y{1, 1}, 1);
    elseif DR == 1
        dim = 50;
    else
        DR = 1;
        dim = 50;
    end
    %% Subspace segmentation
    for s = [2]
        Par.s = s;
        for maxIter = 5
            Par.maxIter = maxIter;
            for rho = [10]
                Par.rho = rho;
                for lambda = [0]
                    Par.lambda = lambda*10^(-0);
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
                        mask = zeros(1, sum(nSample));
                        gnd = zeros(1, sum(nSample));
                        nSample_cum = [0, cumsum(nSample)];
                        for iK = 1:nCluster % randomly take data for each digit.
                            allpos = find( MNIST_LABEL == Digits(iK) );
                            rng( (i-1) * nCluster + iK );
                            selpos = allpos( randperm(length(allpos), nSample(iK)) );
                            mask( nSample_cum(iK) + 1 : nSample_cum(iK+1) ) = selpos;
                            gnd( nSample_cum(iK) + 1 : nSample_cum(iK+1) ) = iK * ones(1, nSample(iK));
                        end
                        fea = MNIST_DATA(:, mask);
                        N = length(gnd);
                        
                        %% PCA Projection
                        redDim = size(fea, 1);
                        if DR == 1
                            [ eigvector , eigvalue ] = PCA( fea ) ;
                            maxDim = length(eigvalue) ;
                            fea = eigvector' * fea ;
                            redDim = min(nCluster*dim, size(fea, 1)) ;
                        end
                        %% normalize
                        for c = 1 : size(fea,2)
                            fea(:,c) = fea(:,c) /norm(fea(:,c)) ;
                        end
                        %% Subspace Clustering
                        Yfea = fea(1:redDim, :) ;
                        if strcmp(SegmentationMethod, 'RSIM') == 1
                            [missrate(i), grp, bestRank, minNcutValue,W] = RSIM(Yfea, gnd);
                        elseif strcmp(SegmentationMethod, 'S3C') == 1
                            opt.affine =0;
                            opt.outliers =1;
                            opt.lambda = 20;
                            opt.r =0;  % the dimension of the target space when applying PCA or random projection
                            opt.SSCrho=1;
                            % paramters for StrSSC
                            opt.iter_max =10; %  iter_max is for loop in StrLRSCE
                            opt.nu =1;
                            opt.gamma0 = 0.1;% This is for reweighting the off-diagonal entries in Z
                            opt.maxIter =150;
                            missrate(i) = StrSSC(Yfea, gnd, opt);
                        else
                            switch SegmentationMethod
                                case 'SSC'
                                    alpha = Par.lambda;
                                    C = admmLasso_mat_func(Yfea, true, alpha);
                                case 'LRR'
                                    C = solve_lrr(Yfea, Par.lambda); % without post processing
                                case 'LRSC'
                                    C = lrsc_noiseless(Yfea, Par.lambda);
                                    % [~, C] = lrsc_noisy(Yfea, Par.lambda);
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
                                    C = LSR1( Yfea , Par.lambda ) ; % prop osed by Lu
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
                                case 'DANNLSR'                 % deformable, affine, non-negative
                                    C = DANNLSR( Yfea , Par ) ;
                                case 'DANNLSRd0'             % deformable, affine, non-negative, diagonal = 0
                                    C = DANNLSRd0( Yfea , Par ) ;
                                case 'DANPLSR'                 % deformable, affine, non-positive
                                    C = DANPLSR( Yfea , Par ) ;
                                case 'DANPLSRd0'             % deformable, affine, non-positive, diagonal = 0
                                    C = DANPLSRd0( Yfea , Par ) ;
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
                        end
                        fprintf('%d: %.3f%% \n' , i, missrate(i)*100) ;
                    end
                    %% output
                    avgmissrate = mean(missrate*100);
                    medmissrate = median(missrate*100);
                    fprintf('Total mean missrate  is %.3f%%.\n' , avgmissrate) ;
                    if strcmp(SegmentationMethod, 'SSC')==1 ...
                            || strcmp(SegmentationMethod, 'LRR')==1 ...
                            || strcmp(SegmentationMethod, 'LRSC')==1 ...
                            || strcmp(SegmentationMethod, 'LSR')==1 ...
                            || strcmp(SegmentationMethod, 'LSR1')==1 ...
                            || strcmp(SegmentationMethod, 'LSR2')==1 ...
                            || strcmp(SegmentationMethod, 'SMR')==1
                        matname = sprintf([write_results_dir dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrate','avgmissrate','medmissrate');
                    elseif strcmp(SegmentationMethod, 'SSCOMP')==1
                        matname = sprintf([write_results_dir dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_K' num2str(Par.rho) '_thr' num2str(Par.lambda) '.mat']);
                        save(matname,'missrate','avgmissrate','medmissrate');
                    elseif strcmp(SegmentationMethod, 'NNLSR') == 1 ...
                            || strcmp(SegmentationMethod, 'NPLSR') == 1 ...
                            || strcmp(SegmentationMethod, 'NNLSRd0') == 1 ...
                            || strcmp(SegmentationMethod, 'NPLSRd0')==1 ...
                            || strcmp(SegmentationMethod, 'ANNLSR') == 1 ...
                            || strcmp(SegmentationMethod, 'ANNLSRd0')==1 ...
                            || strcmp(SegmentationMethod, 'ANPLSR')==1 ...
                            || strcmp(SegmentationMethod, 'ANPLSRd0')==1
                        matname = sprintf([write_results_dir dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrate','avgmissrate','medmissrate');
                    elseif strcmp(SegmentationMethod, 'DANNLSR')==1 ...
                            || strcmp(SegmentationMethod, 'DANNLSRd0')==1 ...
                            || strcmp(SegmentationMethod, 'DANPLSR')==1 ...
                            || strcmp(SegmentationMethod, 'DANPLSRd0')==1
                        matname = sprintf([write_results_dir dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_s' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrate','avgmissrate','medmissrate');
                    elseif strcmp(SegmentationMethod, 'RSIM')==1
                        matname = sprintf([write_results_dir dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '.mat']);
                        save(matname,'missrate','avgmissrate','medmissrate');
                    end
                end
            end
        end
    end
end