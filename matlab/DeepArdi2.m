clear; close all;

%% test input
img = double(imread('SmileyFace8bitGray.png'));
% img = imresize(img,1/2);
img = img/max(img(:));
% figure; imagesc(img);
origsz = size(img,1);

viewsz = 80;
sz = viewsz*viewsz;
fixation = [40 100];

%% network architecture and random initializations
stepsize=1;
lambda = 1; %L2 regularization
noise = 0.001; %noise to be added to each weight per iteration
iters = 100;
nfigs = 10;
nneurons = [sz+sz/2 1000 1000 1000];
motorlayer = 3;

layers = length(nneurons);
for L = 1:layers
    Lup = L+1;
    if Lup>layers
        Lup = Lup-layers;
    end
    NN{L} = rand([nneurons(L),1]);
    WW{L} = rand([nneurons(L),nneurons(Lup)])/sqrt(nneurons(L)*nneurons(Lup)); %1/sqrt(n)
end
EEWWsz = size(WW{motorlayer-1}(:,1:4));







%% run
n=1; m=0;
figs = [1:nfigs]* iters/nfigs;

in = img(fixation(1)-viewsz/2+1:fixation(1)+viewsz/2 , fixation(2)-viewsz/2+1:fixation(2)+viewsz/2);
in = reshape(in,[sz,1]);
while n<=iters
    %% in. sweep over image
    in = in-mean(in(:)); %zero-center
    in = in/std(in(:)); %normalize
    
    NN{1}(1:sz,1) = in;
    
    %% forward
    for L = 1:layers
        % circular wrap
        Lup = L+1;
        if Lup>layers
            Lup = Lup-layers;
        end

        NN{Lup} = logsig(NN{L}'*WW{L})';
    end
    
        %% motor
    motor(:,n) = softmax(NN{motorlayer}(1:5));
    [~,i(n)] = max(motor(:,n));
    if i(n) == 2
        fixation(1) = fixation(1)+5;
    elseif i(n) == 3
        fixation(1) = fixation(1)-5;
    elseif i(n) == 4
        fixation(2) = fixation(2)+5;
    elseif i(n) == 5
        fixation(2) = fixation(2)-5;
    end

    fixation(fixation < viewsz/2+1) = viewsz/2+1;
    fixation(fixation > origsz-viewsz/2) = origsz-viewsz/2;

    in = img(fixation(1)-viewsz/2+1:fixation(1)+viewsz/2 , fixation(2)-viewsz/2+1:fixation(2)+viewsz/2);
    in = reshape(in,[sz,1]);
    
    %% backward
    for L = layers:-1:1
        % circular wrap
        Lup = L+1;
        if Lup>layers
            Lup = Lup-layers;
        end
        
        if L==layers %last layer loops to original image
            NNi = NN{Lup}; NNi(1:sz,1) = in;
            EE{L} = NN{Lup}-NNi;
        else %otherwise just check previous
            EE{L} = WW{Lup}*(-EE{Lup});
        end
            EE{L} = EE{L} .*NN{Lup}.*(1-NN{Lup}); %still uncertain about this step..
        
        WW{L} = WW{L} - stepsize*(EE{L}*(NN{L}'))';
        WW{L} = WW{L} - 0.5*lambda*WW{L}; %L2 regularize
        WW{L} = WW{L} + noise*((0.5-rand([nneurons(L),nneurons(Lup)]))/sqrt(nneurons(L)*nneurons(Lup))); %add noise
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
