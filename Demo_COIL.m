clear;

load '/home/csjunxu/Github/Deep-subspace-clustering-networks/Data/COIL20.mat';
% COIL100
dataset = 'COIL20';
write_results_dir = ['/home/csjunxu/Github/SRLSR_Results/' dataset '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end

%% Subspace segmentation methods
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
                switch SegmentationMethod
                    case 'LSR'
                        C = LSR( Yfea , Par ) ;
                    case 'NLSR'                   % non-negative LSR
                        C = NLSR( Yfea , Par ) ;
                    case 'SLSR'                 % scaled affine LSR
                        C = SLSR( Yfea , Par ) ;
                    case 'SRLSR'                 % scaled affine, non-negative, LSR
                        C = SRLSR( Yfea , Par ) ;
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
                if strcmp(SegmentationMethod, 'LSR')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_lambda' num2str(Par.lambda) '.mat']);
                elseif strcmp(SegmentationMethod, 'NLSR')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                elseif strcmp(SegmentationMethod, 'SLSR')==1 || strcmp(SegmentationMethod, 'SRLSR')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_s' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                end
                if missrate<0.20
                    save(matname,'missrate');
                end
            end
        end
    end
end


clear;

load '/home/csjunxu/Github/Deep-subspace-clustering-networks/Data/COIL100.mat';
% COIL100
dataset = 'COIL100';
write_results_dir = ['/home/csjunxu/Github/SRLSR_Results/' dataset '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end

%% Subspace segmentation methods
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
                switch SegmentationMethod
                    case 'LSR'
                        C = LSR( Yfea , Par ) ;
                    case 'NLSR'                   % non-negative LSR
                        C = NLSR( Yfea , Par ) ;
                    case 'SLSR'                 % scaled affine LSR
                        C = SLSR( Yfea , Par ) ;
                    case 'SRLSR'                 % scaled affine, non-negative, LSR
                        C = SRLSR( Yfea , Par ) ;
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
                if strcmp(SegmentationMethod, 'LSR')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_lambda' num2str(Par.lambda) '.mat']);
                elseif strcmp(SegmentationMethod, 'NLSR')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                elseif strcmp(SegmentationMethod, 'SLSR')==1 || strcmp(SegmentationMethod, 'SRLSR')==1
                    matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_s' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                end
                if missrate<0.40
                    save(matname,'missrate');
                end
            end
        end
    end
end


