
clear ;

load 'C:\Users\csjunxu\Desktop\CVPR2018 SC\Datasets\YaleBCrop025.mat';
dataset = 'YaleB_SSC';
write_results_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 SC/Results/' dataset '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end

%% Subspace segmentation methods
% SegmentationMethod = 'SSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2013 PAMI SSC');
% SegmentationMethod = 'LRR' ; addpath('C:\Users\csjunxu\Desktop\SC\LRR ICML2010 NIPS2011 PAMI2013\code\');
% SegmentationMethod = 'LRSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2011 CVPR LRSC\');
% SegmentationMethod = 'LSR1' ;
% SegmentationMethod = 'LSR2' ;
% SegmentationMethod = 'LSR' ;
% SegmentationMethod = 'LSRd0' ;
% SegmentationMethod = 'SMR' ; addpath('C:\Users\csjunxu\Desktop\SC\SMR_v1.0');
% SegmentationMethod = 'RSIM' ; ii = 0;addpath('C:\Users\csjunxu\Desktop\SC\Ncut_9'); addpath('C:\Users\csjunxu\Desktop\SC\2015 ICCV RSIM');
% SegmentationMethod = 'SSCOMP' ; addpath('C:\Users\csjunxu\Desktop\SC\SSCOMP_Code');

% SegmentationMethod = 'NNLSR' ;
% SegmentationMethod = 'NNLSRd0' ;
% SegmentationMethod = 'NPLSR' ;
% SegmentationMethod = 'NPLSRd0' ;

% SegmentationMethod = 'ANNLSR' ;
% SegmentationMethod = 'ANNLSRd0' ;
% SegmentationMethod = 'ANPLSR' ;
% SegmentationMethod = 'ANPLSRd0' ;

% SegmentationMethod = 'DANNLSR';
% SegmentationMethod = 'DANNLSRd0';
% SegmentationMethod = 'DALSR';
SegmentationMethod = 'DALSRd0';

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

alltime = [];
jj=0;
%% Subspace segmentation
for maxIter = [1] % unique([floor(10*scale), ceil(10*scale)])
    Par.maxIter = maxIter;
    for rho = [1]
        Par.rho = rho;
        for scale = [.1:.1:1.5]
            Par.s = scale;
            for lambda = [0]
                Par.lambda = lambda;
                for nSet = [2 3 5 8 10]
                    n = nSet;
                    index = Ind{n};
                    jj=jj+1;
                    ii=0;
                    for i = 1:size(index,1)
                        X = [];
                        for p = 1:n
                            X = [X Y(:,:,index(i,p))];
                        end
                        [D,N] = size(X);
                        ii = ii+1;
                        t1=clock;
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
                            if strcmp(SegmentationMethod, 'RSIM') == 1
                                [missrate(i, j), grp, bestRank, minNcutValue,W] = RSIM(Yfea, gnd);
                            else
                                switch SegmentationMethod
                                    case 'SSC'
                                        alpha = Par.lambda;
                                        CMat = admmOutlier_mat_func(Yfea, true, alpha);
                                        N = size(Yfea,2);
                                        C = CMat(1:N,:);
                                    case 'LRR'
                                        C = solve_lrr(Yfea, Par.lambda); % without post processing
                                    case 'LRSC'
                                        C = lrsc_noiseless(Yfea, Par.lambda);
                                        %  [~, C] = lrsc_noisy(ProjX, Par.lambda);
                                    case 'SMR'
                                        para.aff_type = 'J1';
                                        % J1 is unrelated to gamma, which is used in J2 and J2_norm
                                        para.gamma = 1;
                                        para.alpha = Par.lambda; % 20
                                        para.knn = 4;
                                        para.elpson =0.01;
                                        Yfea = [Yfea ; ones(1,size(Yfea,2))] ;
                                        C = smr(Yfea, para);
                                    case 'SSCOMP'
                                        C = OMP_mat_func(Yfea, Par.rho, Par.lambda);
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
                                    case 'DALSR'                 % affine
                                        C = DALSR( Yfea , Par ) ;
                                    case 'DALSRd0'             % affine, diagonal = 0
                                        C = DALSRd0( Yfea , Par ) ;
                                    case 'DANNLSR'                 % deformable, affine, non-negative
                                        C = DANNLSR( Yfea , Par ) ;
                                    case 'DANNLSRd0'             % deformable, affine, non-negative, diagonal = 0
                                        C = DANNLSRd0( Yfea , Par ) ;
                                end
                                for k = 1 : size(C,2)
                                    C(:, k) = C(:, k) / max(abs(C(:, k))) ;
                                end
                                nCluster = length( unique( gnd ) ) ;
                                Z = ( abs(C) + abs(C') ) / 2 ;
                                idx = clu_ncut(Z,nCluster) ;
                                missrate(i, j) = 1 - compacc(idx,gnd');
                            end
                            t2=clock;
                            alltime(jj,ii) = etime(t2,t1);
                            % save(['YaleBSSC_' SegmentationMethod '.mat'], 'alltime');
                            fprintf('%.3f%% \n' , missrate(i, j)*100) ;
                        end
                        missrateTot{n}(i) = mean(missrate(i, :)*100);
                        fprintf('Mean error of %d/%d is %.3f%%.\n ' , i, size(index, 1), missrateTot{n}(i)) ;
                    end
                    avgmissrate(n) = mean(missrateTot{n});
                    medmissrate(n) = median(missrateTot{n});
                    fprintf('Total mean error  is %.3f%%.\n ' , avgmissrate(n)) ;
                    allavgmissrate = mean(avgmissrate(avgmissrate~=0));
                      if strcmp(SegmentationMethod, 'SSC')==1 || strcmp(SegmentationMethod, 'S3C')==1
                        matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_alpha' num2str(Par.lambda) '.mat']);
                        save(matname,'missrateTot','avgmissrate','medmissrate','allavgmissrate');
                    elseif  strcmp(SegmentationMethod, 'LRR')==1 || strcmp(SegmentationMethod, 'LRSC')==1 || strcmp(SegmentationMethod, 'LSR')==1 || strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1
                        matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrateTot','avgmissrate','medmissrate','allavgmissrate');
                    elseif strcmp(SegmentationMethod, 'NNLSR')==1 || strcmp(SegmentationMethod, 'NPLSR')==1 || strcmp(SegmentationMethod, 'NNLSRd0')==1 || strcmp(SegmentationMethod, 'NPLSRd0')==1
                        matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrateTot','avgmissrate','medmissrate','allavgmissrate');
                    elseif strcmp(SegmentationMethod, 'ANNLSR')==1 || strcmp(SegmentationMethod, 'ANNLSRd0')==1 ...
                            || strcmp(SegmentationMethod, 'DANNLSR')==1 || strcmp(SegmentationMethod, 'DANNLSRd0')==1 ...
                            || strcmp(SegmentationMethod, 'DALSR')==1 || strcmp(SegmentationMethod, 'DALSRd0')==1
                        matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '_s' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'missrateTot','avgmissrate','medmissrate','allavgmissrate');
                    elseif strcmp(SegmentationMethod, 'RSIM')==1 || strcmp(SegmentationMethod, 'S3C') == 1
                        matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_DR' num2str(DR) '_dim' num2str(dim) '.mat']);
                        save(matname,'missrateTot','avgmissrate','medmissrate','allavgmissrate');
                    end
                end
            end
        end
    end
end



