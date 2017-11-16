clear; close all;
origimg = double(imread('../SmileyFace8bitGray.png'));


fix = [.2;.2];
sz = [100,100];
szn = sz(1)*sz(2);
img = lensdistortJD(origimg,100,fix');
img(img==255) = nan;
[a,b] = find(~isnan(img));
img = img(min(a):max(a),min(b):max(b));
img = inpaintn(img);
img = imresize(img,sz);
img = img-mean(img(:)); %zero-center
img = img/std(img(:)); %normalize
imagesc(img);
img = reshape(img,[szn,1]);


%% network architecture and random initializations
stepsize=1;
lambda = 1; %L2 regularization
iters = 10;
nfigs = 10;
nneurons = [szn+2 10000];
layers = length(nneurons);

% initialize
for L = 1:layers
    Lup = L+1;
    if Lup>layers
        Lup = Lup-layers;
    end
    NN{L} = zeros([nneurons(L),1]);
    WW{L} = (rand([nneurons(L),nneurons(Lup)])-0.5)/sqrt(nneurons(L)*nneurons(Lup)); %1/sqrt(n)
end

%% run
n=1; m=0;
figs = [1:nfigs]* iters/nfigs;

while n<=iters   
    NN{1} = NN{1}+[img;fix];
    
    %% forward prop
    for L = 1:layers
        % circular wrap
        Lup = L+1;
        if Lup>layers
            Lup = Lup-layers;
        end

        NN{Lup} = NN{Lup} + logsig(NN{L}'*WW{L})';
    end
    
    %% backward
    for L = layers:-1:1
        % circular wrap
        Lup = L+1;
        if Lup>layers
            Lup = Lup-layers;
        end
        
        
        if L==layers %last layer loops to original image
        EE{L} = NN{layers} - NN{1}(1:szn);
        else %otherwise just check previous
            EE{L} = WW{Lup}*(-EE{Lup});
        end
            EE{L} = EE{L} .*NN{Lup}.*(1-NN{Lup}); %still uncertain about this step..
        
        WW{L} = WW{L} - stepsize*(EE{L}*(NN{L}'))';
        WW{L} = WW{L} - 0.5*lambda*WW{L}; %L2 regularize
    end
    
    
    %% benchmark
    disp(n);
    for L=1:layers
        tmp(L)=sum(abs(EE{L}));
    end
    totEE(n) = sum(tmp);
        
    if any(figs==n)
        m=m+1;
%         figure; imagesc(reshape(in,[viewsz viewsz])); title('target');
        figure; imagesc(reshape(NN{1}(1:sz,1),[viewsz viewsz])); title('produced')
%         figure; imagesc(reshape(EE{layers}(1:sz,1),[viewsz viewsz])); title('error');
%         save(sprintf('ArdiNet.%d.mat',m));
    end
    n = n+1;
end


%% assess change in error over time
pl = totEE;
pl(1:5) = 0;
%pl(pl>mean(pl)+std(pl) | pl<mean(pl)-std(pl)) = nan;
figure;
plot(pl);
ylim([0 mean(pl)+std(pl)]);
