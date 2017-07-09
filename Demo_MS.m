clear ;

%% reduced dimension
ProjRank = 12 ;
datadir = 'C:/Users/csjunxu/Desktop/SC/Datasets/Hopkins155/';
seqs = dir(datadir);
% Get rid of the two directories: "." and ".."
seq3 = seqs(3:end);
% Save the data loaded in struct "data "
data = struct('ProjX', {}, 'name',{}, 'ids',{});

dataset = 'Hopkins155';
writefilepath = ['C:/Users/csjunxu/Desktop/SC/Results/' dataset '/'];

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
% SegmentationMethod = 'LSR' ; % the same with LSR2
% SegmentationMethod = 'LSRd0' ;
% SegmentationMethod = 'LSR1' ; % 4.8
% SegmentationMethod = 'LSR2' ; % 4.6

% SegmentationMethod = 'NNLSR' ;
% SegmentationMethod = 'NNLSRd0' ;
% SegmentationMethod = 'NPLSR' ; % SVD 的输入不能包含 NaN 或 Inf。
% SegmentationMethod = 'NPLSRd0' ; % SVD 的输入不能包含 NaN 或 Inf。

% SegmentationMethod = 'ANNLSR' ;
% SegmentationMethod = 'ANNLSRd0' ;
% SegmentationMethod = 'ANPLSR' ;
% SegmentationMethod = 'ANPLSRd0' ;

SegmentationMethod = 'DANNLSR' ;
% SegmentationMethod = 'DANNLSRd0' ;
% SegmentationMethod = 'DANPLSR' ;
% SegmentationMethod = 'DANPLSRd0' ;

for mu = [1]
    Par.mu = mu;
    for maxIter = [5]
        Par.maxIter = maxIter;
        for s = [1 2]
            Par.s = s;
            for rho = [.1:.1:.9]
                Par.rho = rho;
                for lambda = [0]
                    Par.lambda = lambda*10^(-4);
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
                        switch SegmentationMethod
                            case 'LSR'
                                C = LSR( ProjX , Par ) ;
                            case 'LSRd0'
                                C = LSRd0( ProjX , Par ) ; % solved by ADMM
                            case 'LSR1'
                                C = LSR1( ProjX , Par.lambda ) ; % proposed by Lu
                            case 'LSR2'
                                C = LSR2( ProjX , Par.lambda ) ; % proposed by Lu
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
                        end
                        %% this step is useless for motion segmentation
                        %                             for k = 1 : size(C,2)
                        %                                 C(:, k) = C(:, k) / max(abs(C(:, k))) ;
                        %                             end
                        nCluster = length( unique( gnd ) ) ;
                        Z = ( abs(C) + abs(C') ) / 2 ;
                        idx = clu_ncut(Z,nCluster) ;
                        accuracy = compacc(idx,gnd) ;
                        missrate = 1-accuracy;
                        num(n) = num(n) + 1;
                        missrateTot{n}(num(n)) = missrate;
                        fprintf('seq %d\t %f\n', i , missrate ) ;
                    end
                    fprintf('\n') ;
                    
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
                    if strcmp(SegmentationMethod, 'LSR')==1 || strcmp(SegmentationMethod, 'LSR1')==1 || strcmp(SegmentationMethod, 'LSR2')==1
                        matname = sprintf([writefilepath dataset '_' SegmentationMethod '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'avgallmissrate','medallmissrate','missrateTot','avgmissrate','medmissrate');
                    else
                        matname = sprintf([writefilepath dataset '_' SegmentationMethod '_maxIter' num2str(Par.maxIter) '_scale' num2str(Par.s) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'avgallmissrate','medallmissrate','missrateTot','avgmissrate','medmissrate');
                    end
                end
            end
        end
    end
end




