clear;

load '/home/csjunxu/Github/Deep-subspace-clustering-networks/Data/ORL_32x32.mat';
dataset = 'ORL';
write_results_dir = ['/home/csjunxu/Github/SRLSR_Results/' dataset '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end

%% Subspace segmentation methods
% SegmentationMethod = 'SSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2013 PAMI SSC');
% SegmentationMethod = 'LRR' ; addpath('C:\Users\csjunxu\Desktop\SC\LRR ICML2010 NIPS2011 PAMI2013\code\');
% SegmentationMethod = 'LRSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2011 CVPR LRSC\');
% SegmentationMethod = 'LSR1' ;
% SegmentationMethod = 'LSR2' ;
% SegmentationMethod = 'SMR' ; addpath('C:\Users\csjunxu\Desktop\SC\SMR_v1.0');
% SegmentationMethod = 'RSIM' ; ii = 0;addpath('C:\Users\csjunxu\Desktop\SC\Ncut_9'); addpath('C:\Users\csjunxu\Desktop\SC\2015 ICCV RSIM');
% SegmentationMethod = 'SSCOMP' ; addpath('C:\Users\csjunxu\Desktop\SC\SSCOMP_Code');

% SegmentationMethod = 'LSR' ; % the same with LSR2
% SegmentationMethod = 'NLSR';
% SegmentationMethod = 'SLSR';
SegmentationMethod = 'SRLSR';

Ctime = [];
%% Subspace segmentation
for scale = [.1:.1:2]
    Par.s = scale;
    for maxIter = 1:1:20
        Par.maxIter = maxIter;
        for rho = [.1:.1:5]
            Par.rho = rho;
            for lambda = [.1:.1:5]
                Par.lambda = lambda;
                t1=clock;
                % normalize
                for c = 1 : size(fea,2)
                    fea(:,c) = fea(:,c) /norm(fea(:,c)) ;
                end
                % clustering
                Yfea = fea';
                if strcmp(SegmentationMethod, 'RSIM') == 1
                    [missrate, grp, bestRank, minNcutValue,W] = RSIM(Yfea, gnd);
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
                        case 'NLSR'                   % non-negative LSR
                            C = NLSR( Yfea , Par ) ;
                        case 'SLSR'                 % scaled affine LSR
                            C = SLSR( Yfea , Par ) ;
                        case 'SRLSR'                 % scaled affine, non-negative, LSR
                            C = SRLSR( Yfea , Par ) ;
                    end
                end
                for k = 1 : size(C,2)
                    C(:, k) = C(:, k) / max(abs(C(:, k))) ;
                end
                nCluster = length( unique( gnd ) ) ;
                Z = ( abs(C) + abs(C') ) / 2 ;
                idx = clu_ncut(Z,nCluster) ;
                missrate = 1 - compacc(idx,gnd');
                t2=clock;
                Ctime = etime(t2,t1);
                % save(['YaleBSSC_' SegmentationMethod '.mat'], 'alltime');
                fprintf('error is %.3f%% \n' , missrate*100) ;
                if strcmp(SegmentationMethod, 'SSC')==1 || strcmp(SegmentationMethod, 'S3C')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_alpha' num2str(Par.lambda) '.mat']);
                elseif strcmp(SegmentationMethod, 'LRR')==1 || strcmp(SegmentationMethod, 'LRSC')==1 || strcmp(SegmentationMethod, 'LSR')==1 || strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_lambda' num2str(Par.lambda) '.mat']);
                elseif strcmp(SegmentationMethod, 'NNLSR')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                elseif strcmp(SegmentationMethod, 'SRLSR')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_s' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                elseif strcmp(SegmentationMethod, 'RSIM')==1 || strcmp(SegmentationMethod, 'S3C') == 1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '.mat']);
                end
                if missrate<0.23
                    save(matname,'missrate');
                end
            end
        end
    end
end


