clear;

addpath('MNISThelpcode');
dataset = 'MNIST';

%% Subspace segmentation methods
% SegmentationMethod = 'SSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2009 CVPR 2013 PAMI SSC');
% SegmentationMethod = 'LRR' ; addpath('C:\Users\csjunxu\Desktop\SC\2010 ICML 2013 PAMI LRR\code\');
% SegmentationMethod = 'LRSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2011 CVPR LRSC\');
% SegmentationMethod = 'LSR1' ;
% SegmentationMethod = 'LSR2' ;
% SegmentationMethod = 'SMR' ; addpath('C:\Users\csjunxu\Desktop\SC\2014 CVPR SMR');
% SegmentationMethod = 'S3C' ; addpath('C:\Users\csjunxu\Desktop\SC\2015 CVPR S3C');
% SegmentationMethod = 'RSIM' ; addpath('C:\Users\csjunxu\Desktop\SC\Ncut_9');addpath('C:\Users\csjunxu\Desktop\SC\2015 ICCV RSIM');
% SegmentationMethod = 'SSCOMP' ;

% SegmentationMethod = 'LSR' ; % the same with LSR2
% SegmentationMethod = 'NLSR';
% SegmentationMethod = 'SLSR';
SegmentationMethod = 'SRLSR';
alltime = [];
jj=0;
%% Settings
for nSample = [50 100 200 400 600] % number of images for each digit
    write_results_dir = ['/Users/xujun/Desktop/Results/' dataset '/' num2str(nSample) '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end
    jj=jj+1;
    %% Load data
    addpath('/Users/xujun/Downloads/MNIST/')
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
            save /Users/xujun/Downloads/MNIST/MNIST_SC.mat MNIST_SC_DATA MNIST_LABEL;
        end
        MNIST_DATA = MNIST_SC_DATA;
    end
    
    nExperiment = 20; % 20 number of repeations
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
    for s = [.5]
        Par.s = s;
        for maxIter =  1:1:5
            Par.maxIter = maxIter;
            for rho =  [.5 1]
                Par.rho = rho;
                for lambda = [0 .1 .001 .01]
                    Par.lambda = lambda;
                    missrate = zeros(nExperiment, 1) ;
                    ii=0;
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
                        
                        ii = ii+1;
                        t1=clock;
                        
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
                            opt.iter_max =5; %  iter_max is for loop in StrLRSCE
                            opt.nu =1;
                            opt.gamma0 = 0.1;% This is for reweighting the off-diagonal entries in Z
                            opt.maxIter =150;
                            [missrate(i), Theta, C, eval_iter] = StrSSC(Yfea, gnd, opt);
                        else
                            switch SegmentationMethod
                                case 'SSC'
                                    alpha = Par.lambda;
                                    C = admmLasso_mat_func(Yfea, true, alpha);
                                case 'LRR'
                                    C = solve_lrr(Yfea, Par.lambda); % without post processing
                                case 'LRSC'
                                    %                                     C = lrsc_noiseless(Yfea, Par.lambda);
                                    [~, C] = lrsc_noisy(Yfea, Par.lambda);
                                case 'SMR'
                                    para.aff_type = 'J1'; % J1 is unrelated to gamma, which is used in J2 and J2_norm
                                    para.gamma = 1;
                                    para.alpha = 2e-5;
                                    para.knn = 4;
                                    para.elpson =0.001;
                                    Yfea = [Yfea ; ones(1,size(Yfea,2))] ;
                                    C = smr(Yfea, para);
                                case 'SSCOMP' % add the path of the SSCOMP method
                                    addpath('C:\Users\csjunxu\Desktop\CVPR2018 SC\SSCOMP_Code');
                                    C = OMP_mat_func(Yfea, 9, 1e-6);
                                case 'LSR1'
                                    C = LSR1( Yfea , Par.lambda ) ; % proposed by Lu
                                case 'LSR2'
                                    C = LSR2( Yfea , Par.lambda ) ; % proposed by Lu
                                    %% our methods
                                case 'LSR'
                                    C = LSR( Yfea , Par ) ;
                                case 'NLSR'                   % non-negative
                                    C = NLSR( Yfea , Par ) ;
                                case 'SLSR'
                                    C = SLSR(Yfea, Par); % affine
                                case 'SRLSR'                 % deformable, affine, non-negative
                                    C = SRLSR( Yfea , Par ) ;
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
                        t2=clock;
                        alltime(jj,ii) = etime(t2,t1);
                        fprintf('%d: %.3f%% \n' , i, missrate(i)*100) ;
                    end
                    %% output
                    avgmissrate = mean(missrate*100);
                    medmissrate = median(missrate*100);
                    fprintf('Total mean missrate  is %.3f%%.\n' , avgmissrate) ;
                    if strcmp(SegmentationMethod, 'SSC')==1 ...
                            || strcmp(SegmentationMethod, 'LRR')==1 ...
                            || strcmp(SegmentationMethod, 'LRSC')==1 ...
                            || strcmp(SegmentationMethod, 'LSR1')==1 ...
                            || strcmp(SegmentationMethod, 'LSR2')==1 ...
                            || strcmp(SegmentationMethod, 'SMR')==1
                        matname = sprintf([write_results_dir dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrate','avgmissrate','medmissrate');
                    elseif strcmp(SegmentationMethod, 'SSCOMP')==1
                        matname = sprintf([write_results_dir dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_K' num2str(Par.rho) '_thr' num2str(Par.lambda) '.mat']);
                        save(matname,'missrate','avgmissrate','medmissrate');
                    elseif strcmp(SegmentationMethod, 'NLSR') == 1 ...
                            || strcmp(SegmentationMethod, 'LSR')==1 ...
                            || strcmp(SegmentationMethod, 'SLSR') == 1 ...
                            || strcmp(SegmentationMethod, 'SRLSR') == 1
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

