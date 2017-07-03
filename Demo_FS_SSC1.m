

clear ;
warning off;
load 'C:\Users\csjunxu\Desktop\SC\Datasets\YaleBCrop025.mat';
% load 'C:\Users\csjunxu\Desktop\SC\Datasets\USPS_Crop.mat'   % load USPS dataset
dataset = 'YaleB_SSC';
writefilepath = ['C:/Users/csjunxu/Desktop/SC/Results/' dataset '/'];

%% Subspace segmentation methods
% SegmentationMethod = 'LSR' ;
% SegmentationMethod = 'LSRd0' ;
SegmentationMethod = 'LSR1' ;
% SegmentationMethod = 'LSR2' ;

% SegmentationMethod = 'NNLSR' ;
% SegmentationMethod = 'NNLSRd0' ;
% SegmentationMethod = 'NPLSR' ;
% SegmentationMethod = 'NPLSRd0' ;

% SegmentationMethod = 'ANNLSR' ;
% SegmentationMethod = 'ANNLSRd0' ;
% SegmentationMethod = 'ANPLSR' ;
% SegmentationMethod = 'ANPLSRd0' ;

Repeat = 1; %number of repeations
DR = 1; % dimension reduction
if DR == 0
    dim = size(Y, 1);
elseif DR == 1
    dim = 6;
else
    DR = 1;
    dim = 6;
end
%% Subspace segmentation
for maxIter = [5]
    Par.maxIter = maxIter;
    for mu = [1]
        Par.mu = mu;
        for rho = [1]
            Par.rho = rho;
            for lambda = [0 1]
                Par.lambda = lambda*10^(-4);
                for nSet = [2 3 5 8 10]
                    n = nSet;
                    index = Ind{n};
                    for i = 1:size(index,1)
                        X = [];
                        for p = 1:n
                            X = [X Y(:,:,index(i,p))];
                        end
                        [D,N] = size(X);
                        
                        fea = X ;
                        gnd = s{n} ;
                        
                        redDim = size(fea, 1);
                        if DR == 1
                            %% PCA Projection
                            [ eigvector , eigvalue ] = PCA( fea ) ;
                            maxDim = length(eigvalue);
                            fea = eigvector' * fea ;
                            redDim = min(nSet*dim, size(fea, 1)) ;
                        end
                        
                        % normalize
                        for c = 1 : size(fea,2)
                            fea(:,c) = fea(:,c) /norm(fea(:,c)) ;
                        end
                        missrate = zeros(size(index, 1), Repeat) ;
                        fprintf( 'dimension = %d \n', redDim ) ;
                        Yfea = fea(1:redDim, :) ;
                        for j = 1 : Repeat
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
                            for k = 1 : size(C,2)
                                C(:, k) = C(:, k) / max(abs(C(:, k))) ;
                            end
                            nCluster = length( unique( gnd ) ) ;
                            Z = ( abs(C) + abs(C') ) / 2 ;
                            idx = clu_ncut(Z,nCluster) ;
                            missrate(i, j) = 1 - compacc(idx,gnd');
                            fprintf('%.3f%% \n' , missrate(i, j)*100) ;
                        end
                        missrateTot{n}(i) = mean(missrate(i, :)*100);
                        fprintf('Mean error of %d/%d is %.3f%%.\n ' , i, size(index, 1), missrateTot{n}(i)) ;
                    end
                    avgmissrate(n) = mean(missrateTot{n});
                    medmissrate(n) = median(missrateTot{n});
                    if strcmp(SegmentationMethod, 'LSR')==1 || strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1
                        matname = sprintf([writefilepath dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrateTot','avgmissrate','medmissrate');
                    else
                        matname = sprintf([writefilepath dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrateTot','avgmissrate','medmissrate');
                    end
                end
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

