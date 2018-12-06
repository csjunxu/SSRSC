clear;

load '/home/csjunxu/Github/Deep-subspace-clustering-networks/Data/ORL_32x32.mat';
dataset = 'ORL';
write_results_dir = ['/home/csjunxu/Github/SRLSR_Results/' dataset '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end

%% Subspace segmentation methods
% SegmentationMethod = 'SSC' ; addpath('/home/csjunxu/Github/SubspaceClusteringMethods/2009 CVPR 2013 PAMI SSC');
% SegmentationMethod = 'LRR' ; addpath('/home/csjunxu/Github/SubspaceClusteringMethods/2010 ICML 2013 PAMI LRR');
% SegmentationMethod = 'LRSC' ; addpath('/home/csjunxu/Github/SubspaceClusteringMethods/2011 CVPR 2014 PRL LRSC');
% SegmentationMethod = 'LSR1' ;
% SegmentationMethod = 'LSR2' ;
% SegmentationMethod = 'SMR' ; addpath('/home/csjunxu/Github/SubspaceClusteringMethods/2014 CVPR SMR');
% SegmentationMethod = 'RSIM' ; addpath('/home/csjunxu/Github/SubspaceClusteringMethods/Ncut_9'); addpath('/home/csjunxu/Github/SubspaceClusteringMethods/2015 ICCV RSIM');
% SegmentationMethod = 'SSCOMP' ; addpath('/home/csjunxu/Github/SubspaceClusteringMethods/2016 CVPR SSCOMP');

Ctime = [];
%% Subspace segmentation
for rho = [1]
    Par.rho = rho;
    for lambda = [1]
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
        elseif strcmp(SegmentationMethod, 'SMR')==1 || strcmp(SegmentationMethod, 'LRR')==1 || strcmp(SegmentationMethod, 'LRSC')==1 || strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1
            matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_lambda' num2str(Par.lambda) '.mat']);
        elseif strcmp(SegmentationMethod, 'RSIM')==1 || strcmp(SegmentationMethod, 'S3C') == 1
            matname = sprintf([write_results_dir dataset '_' SegmentationMethod '.mat']);
        elseif strcmp(SegmentationMethod, 'SSCOMP')==1
            matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
        end
        save(matname,'missrate');
    end
end

