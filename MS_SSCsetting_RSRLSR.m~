clear;

addpath '/Users/xujun/Documents/GitHub/SRLSR/';


dataset = 'Hopkins155';
cd '/Users/xujun/Desktop/SC/Datasets/Hopkins155/';
write_results_dir = ['/Users/xujun/Desktop/SC/Results/' dataset '/'];
if ~isdir(write_results_dir)
    mkdir(write_results_dir);
end
maxNumGroup = 5;
for i = 1:maxNumGroup
    num(i) = 0;
end


%% Subspace segmentation methods
SegmentationMethod = 'RSRLSR';

for maxIter = 1:1:5
    Par.maxIter = maxIter;
    for s = [1:-.1:.1]
        Par.s = s;
        for rho = [.1 .3 .5]
            Par.rho = rho;
            for lambda = [0 .001 .01 .1]
                Par.lambda = lambda;
                maxNumGroup = 5;
                for i = 1:maxNumGroup
                    num(i) = 0;
                end
                %%
                d = dir;
                ii=0;
                alltime = [];
                for i = 1:length(d)
                    if ( (d(i).isdir == 1) && ~strcmp(d(i).name,'.') && ~strcmp(d(i).name,'..') )
                        filepath = d(i).name;
                        eval(['cd ' filepath]);
                        
                        f = dir;
                        foundValidData = false;
                        for j = 1:length(f)
                            if ( ~isempty(strfind(f(j).name,'_truth.mat')) )
                                ind = j;
                                foundValidData = true;
                                break
                            end
                        end
                        eval(['load ' f(ind).name]);
                        cd ..
                        if (foundValidData)
                            gnd = s;
                            clear s;
                            n = max(gnd);
                            N = size(x,2);
                            F = size(x,3);
                            D = 2*F;
                            X = reshape(permute(x(1:2,:,:),[1 3 2]),D,N);
                            ii = ii+1;
                            r = 4*n;
                            t1=clock;
                            Xp = DataProjection(X,r);
                            
                            % Relaxed Simplex Representation
                            C = RSRLSR( Xp , Par ) ;
                            nCluster = length( unique( gnd ) ) ;
                            Z = (C+C')/2;
                            idx = clu_ncut(Z,nCluster) ;
                            accuracy = compacc(idx, gnd') ;
                            missrate = 1-accuracy;
                            fprintf('seq %d %f\n', ii , missrate ) ;
                            
                            t2=clock;
                            alltime(ii) = etime(t2,t1);
                            num(n) = num(n) + 1;
                            missrateTot{n}(num(n)) = missrate;
                        end
                    end
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
                fprintf('Total mean error  is %.3f%%.\n ' , avgallmissrate*100) ;
                matname = sprintf([write_results_dir dataset '_' SegmentationMethod '_s' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                save(matname,'avgallmissrate','medallmissrate','missrateTot','avgmissrate','medmissrate');
            end
        end
    end
end