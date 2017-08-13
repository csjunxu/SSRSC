clear ;

%% reduced dimension
ProjRank = 12;
datadir = 'C:/Users/csjunxu/Desktop/SC/Datasets/Hopkins155/';
seqs = dir(datadir);
% Get rid of the two directories: "." and ".."
seq3 = seqs(3:end);
% Save the data loaded in struct "data "
data = struct('ProjX', {}, 'name',{}, 'ids',{});

dataset = 'Hopkins155';
writefilepath = ['C:/Users/csjunxu/Desktop/SC/Results/' dataset '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end

for i=1:length(seq3)
    fname = seq3(i).name;
    fdir = [datadir '/' fname];
    if isdir(fdir)
        datai = load([fdir '/' fname '_truth.mat']);
        id = length(data)+1;
        % the true group numbers
        data(id).ids = datai.s;
        % file name
        data(id).name = lower(fname);
        % X is the motion sequence
        X = reshape(permute(datai.x(1:2,:,:),[1 3 2]), 2*datai.frames, datai.points);
        % PCA projection
        [ eigvector , eigvalue ] = PCA( X ) ;
        ProjX = eigvector(:,1:ProjRank)' * X ;
        data(id).ProjX = [ProjX ; ones(1,size(ProjX,2)) ] ;
    end
end
clear seq3;


%% Subspace segmentation methods

% SegmentationMethod = 'SSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2009 CVPR 2013 PAMI SSC');
% SegmentationMethod = 'LRR' ; addpath('C:\Users\csjunxu\Desktop\SC\2010 ICML 2013 PAMI LRR\code\');
% SegmentationMethod = 'LRSC' ; addpath('C:\Users\csjunxu\Desktop\SC\2011 CVPR LRSC\');
% SegmentationMethod = 'SMR' ; addpath('C:\Users\csjunxu\Desktop\SC\SMR_v1.0');
% SegmentationMethod = 'S3C' ; addpath('C:\Users\csjunxu\Desktop\SC\2015 CVPR S3C');
% SegmentationMethod = 'RSIM' ; addpath('C:\Users\csjunxu\Desktop\SC\Ncut_9'); addpath('C:\Users\csjunxu\Desktop\SC\2015 ICCV RSIM');
% SegmentationMethod = 'SSCOMP' ; addpath('C:\Users\csjunxu\Desktop\SC\SSCOMP_Code');
% SegmentationMethod = 'LSR1' ;
% SegmentationMethod = 'LSR2' ;
% SegmentationMethod = 'LSR' ; % the same with LSR2
% SegmentationMethod = 'LSRd0' ; % the same with LSR1

% SegmentationMethod = 'NNLSR';
% SegmentationMethod = 'NNLSRd0';
% SegmentationMethod = 'NPLSR'; % SVD 的输入不能包含 NaN 或 Inf。
% SegmentationMethod = 'NPLSRd0'; % SVD 的输入不能包含 NaN 或 Inf。

% SegmentationMethod = 'ANNLSR';
% SegmentationMethod = 'ANNLSRd0';
% SegmentationMethod = 'ANPLSR';
% SegmentationMethod = 'ANPLSRd0';

% SegmentationMethod = 'DANNLSR';
% SegmentationMethod = 'DANNLSRd0';
% SegmentationMethod = 'DANPLSR';
SegmentationMethod = 'DANPLSRd0';
for s = [1:-.1:.1]
    Par.s = s;
    for maxIter = 1
        Par.maxIter = maxIter;
        for rho = [1]
            Par.rho = rho;
            for lambda = [0]
                Par.lambda = lambda*10^(-0);
                maxNumGroup = 5;
                for i = 1:maxNumGroup
                    num(i) = 0;
                end
                %%
                errs = zeros(length(data),1);
                for i = 1 : length(data)
                    ProjX = data(i).ProjX ;
                    gnd = data(i).ids' ;
                    K = length( unique( gnd ) ) ;
                    n = max(gnd);
                    if strcmp(SegmentationMethod, 'RSIM') == 1
                        [missrate, grp, bestRank, minNcutValue,W] = RSIM(ProjX, gnd);
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
                        missrate = StrSSC(ProjX, gnd, opt);
                    else
                        switch SegmentationMethod
                            case 'SSC'
                                alpha = 800;
                                C = admmLasso_mat_func(ProjX, true, alpha);
                            case 'LRR'
                                C = solve_lrr(ProjX, Par.lambda); % without post processing
                            case 'LRSC'
                                C = lrsc_noiseless(ProjX, 15);
                                %  [~, C] = lrsc_noisy(ProjX, Par.lambda);
                            case 'SMR'
                                para.aff_type = 'J1'; % J1 is unrelated to gamma, which is used in J2 and J2_norm
                                para.gamma = 1;
                                para.alpha = 20;
                                para.knn = 4;
                                para.elpson =0.01;
                                ProjX = [ProjX ; ones(1,size(ProjX,2))] ;
                                C = smr(ProjX, para);
                            case 'SSCOMP' % add the path of the SSCOMP method
                                C = OMP_mat_func(ProjX, 9, 1e-6);
                            case 'LSR'
                                C = LSR( ProjX , Par ) ;
                            case 'LSRd0'
                                C = LSRd0( ProjX , Par ) ; % solved by ADMM
                            case 'LSR1'
                                C = LSR1( ProjX , Par.lambda ) ; % proposed by Lu
                            case 'LSR2'
                                C = LSR2( ProjX , Par.lambda ) ; % proposed by Lu
                                %% our methods
                            case 'NNLSR'                   % non-negative
                                C = NNLSR( ProjX , Par ) ;
                            case 'NNLSRd0'               % non-negative, diagonal = 0
                                C = NNLSRd0( ProjX , Par ) ;
                            case 'NPLSR'                   % non-positive
                                C = NPLSR( ProjX , Par ) ;
                            case 'NPLSRd0'               % non-positive, diagonal = 0
                                C = NPLSRd0( ProjX , Par ) ;
                            case 'ANNLSR'                 % affine, non-negative
                                C = ANNLSR( ProjX , Par ) ;
                            case 'ANNLSRd0'             % affine, non-negative, diagonal = 0
                                C = ANNLSRd0( ProjX , Par ) ;
                            case 'ANPLSR'                 % affine, non-positive
                                C = ANPLSR( ProjX , Par ) ;
                            case 'ANPLSRd0'             % affine, non-positive, diagonal = 0
                                C = ANPLSRd0( ProjX , Par ) ;
                            case 'DANNLSR'                 % deformable, affine, non-negative
                                C = DANNLSR( ProjX , Par ) ;
                            case 'DANNLSRd0'             % deformable, affine, non-negative, diagonal = 0
                                C = DANNLSRd0( ProjX , Par ) ; 
                            case 'DANPLSR'                 % deformable, affine, non-positive
                                C = DANPLSR( ProjX , Par ) ;
                            case 'DANPLSRd0'             % deformable, affine, non-positive, diagonal = 0
                                C = DANPLSRd0( ProjX , Par ) ;
                        end
                        nCluster = length( unique( gnd ) ) ;
                        Z = ( abs(C) + abs(C') ) / 2 ;
                        idx = clu_ncut(Z,nCluster) ;
                        accuracy = compacc(idx,gnd) ;
                        missrate = 1-accuracy;
                    end
                    num(n) = num(n) + 1;
                    missrateTot{n}(num(n)) = missrate;
                    fprintf('seq %d\t %f\n', i , missrate ) ;
                end
                L = [2 3];
                allmissrate = [];
                for i = 1:length(L)
                    j = L(i);
                    avgmissrate(j) = mean(missrateTot{j});
                    medmissrate(j) = median(missrateTot{j});
                    allmissrate = [allmissrate missrateTot{j}];
                end
                avgallmissrate = sum(allmissrate)/length(allmissrate);
                medallmissrate = median(allmissrate);
                fprintf('Total mean error  is %.3f%%. \n' , avgallmissrate*100) ;
                if strcmp(SegmentationMethod, 'SSC')==1 || strcmp(SegmentationMethod, 'LRR')==1 || ...
                        strcmp(SegmentationMethod, 'LRSC')==1 || strcmp(SegmentationMethod, 'LSR')==1 || ...
                        strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1 || ...
                        strcmp(SegmentationMethod, 'SMR')==1
                    matname = sprintf([writefilepath dataset '_' SegmentationMethod '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'avgallmissrate','medallmissrate','missrateTot','avgmissrate','medmissrate');
                elseif strcmp(SegmentationMethod, 'NNLSR')==1 || strcmp(SegmentationMethod, 'NNLSRd0')==1 || ...
                        strcmp(SegmentationMethod, 'NPLSR')==1 || strcmp(SegmentationMethod, 'NPLSRd0')==1 || ...
                        strcmp(SegmentationMethod, 'ANNLSR')==1 || strcmp(SegmentationMethod, 'ANPLSR')==1 || ...
                        strcmp(SegmentationMethod, 'ANNLSRd0')==1 || strcmp(SegmentationMethod, 'ANPLSRd0')==1
                    matname = sprintf([writefilepath dataset '_' SegmentationMethod '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'avgallmissrate','medallmissrate','missrateTot','avgmissrate','medmissrate');
                elseif strcmp(SegmentationMethod, 'DANNLSR') == 1 || ...
                        strcmp(SegmentationMethod, 'DANNLSRd0') == 1 || ...
                        strcmp(SegmentationMethod, 'DANPLSR') == 1 || ...
                        strcmp(SegmentationMethod, 'DANPLSRd0') == 1
                    matname = sprintf([writefilepath dataset '_' SegmentationMethod '_s' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'avgallmissrate','medallmissrate','missrateTot','avgmissrate','medmissrate');
                elseif strcmp(SegmentationMethod, 'SSCOMP')==1
                    matname = sprintf([writefilepath dataset '_' SegmentationMethod '_K' num2str(Par.rho) '_thr' num2str(Par.lambda) '.mat']);
                    save(matname,'avgallmissrate','medallmissrate','missrateTot','avgmissrate','medmissrate');
                elseif strcmp(SegmentationMethod, 'RSIM')==1
                    matname = sprintf([writefilepath dataset '_' SegmentationMethod '.mat']);
                    save(matname,'avgallmissrate','medallmissrate','missrateTot','avgmissrate','medmissrate');
                end
            end
        end
    end
end




