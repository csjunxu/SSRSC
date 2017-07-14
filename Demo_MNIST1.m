clear;

addpath('MNISThelpcode');
addpath('C:\Users\csjunxu\Desktop\SC\SSCOMP_Code\scatnet-0.2');
%% Settings
for nSample = [600] % number of images for each digit
    
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
    
    
    dataset = 'MNIST';
    writefilepath = ['C:/Users/csjunxu/Desktop/SC/Results/' dataset '/'];
    
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
    
    %% Subspace segmentation methods
    %     SegmentationMethod = 'LSR' ;
    %     SegmentationMethod = 'LSRd0' ;
    %     SegmentationMethod = 'LSR1' ;
    %     SegmentationMethod = 'LSR2' ;
    
    %     SegmentationMethod = 'NNLSR' ;
    %     SegmentationMethod = 'NNLSRd0' ;
    %     SegmentationMethod = 'NPLSR' ;
    %     SegmentationMethod = 'NPLSRd0' ;
    
    %     SegmentationMethod = 'ANNLSR' ;
    %     SegmentationMethod = 'ANNLSRd0' ;
        SegmentationMethod = 'ANPLSR' ;
%     SegmentationMethod = 'ANPLSRd0' ;
    %% Subspace segmentation
    for maxIter = [5]
        Par.maxIter = maxIter;
        for rho = [30]
            Par.rho = rho;
            for lambda = [1]
                Par.lambda = lambda*10^(-4);
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
                    %                     fprintf( 'dimension = %d \n', redDim ) ;
                    %% normalize
                    for c = 1 : size(fea,2)
                        fea(:,c) = fea(:,c) /norm(fea(:,c)) ;
                    end
                    %% Subspace Clustering
                    Yfea = fea(1:redDim, :) ;
                    switch SegmentationMethod
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
                if strcmp(SegmentationMethod, 'LSR')==1 || strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1
                    matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'missrate','avgmissrate','medmissrate');
                else
                    matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'missrate','avgmissrate','medmissrate');
                end
            end
        end
    end
end

