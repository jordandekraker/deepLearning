clear; close all;

%% get input & initial view
origimg = double(imread('../SmileyFace8bitGray.png'));
sz = [100,100]; szn = sz(1)*sz(2);
fix = [.5,.5];
img = saccade(origimg,fix);

testfixes = [.5,.5; .2,.2; .8,.8];
ntest = size(testfixes,1);
for n = 1:ntest
testimges(n,:) = saccade(origimg,testfixes(n,:));
end
%% network architecture and initialization
stepsize = 1;
lambda = .1; %L2 regularization
decay = 4000;
iters = 1000;
nneurons = [szn+2 1000 1000]; %first layer has size: image size + fixation coords + bias term
NNbias = 0;
nfigs = 10; %how many intermeidate figures to generate

% initialize
layers = length(nneurons);
for L = 1:layers
    NN{L} = zeros([1,nneurons(L)]);
    Lup = circup(L,layers);
    WW{L} = (rand([nneurons(L)+1,nneurons(Lup)])-0.5)/sqrt(nneurons(L)*nneurons(Lup)); %1/sqrt(n)
end

%% run
NNnew = [img,fix]; n=1; m=0; totEE=zeros([1,iters]); figs=[1:nfigs]*floor(iters/nfigs);
while n<=iters
    % add input
    NN{1} = NNnew;
    % forward prop
    for L = [2:layers]
        Ldn = circdn(L,layers);
        NN{L} = tanh([NNbias,NN{Ldn}]*WW{Ldn});
    end
    NN{1} = [NNbias,NN{layers}]*WW{layers}; % no tanh on recurrence
    
    % make a move
%     fix = testfixes(rem(n,ntest)+1,:); 
%     img = testimges(rem(n,ntest)+1,:);
%     if rand(1) < 0.1
    fix = round(rand([1,2]),2);
    img = saccade(origimg,fix);
%     end
    NNnew = [img,fix];
    
    % backprop based on the results of the move
    for L = layers:-1:1
        Lup = circup(L,layers);
        if L==layers
            EE{L} = NN{Lup} - NNnew; 
        else
            EE{L} = EE{Lup}*(WW{Lup}');
            EE{L}(end) = [];
            EE{L} = EE{L}.*[NN{Lup}].*[1-NN{Lup}];
        end
        
        WW{L} = WW{L}*(1-stepsize*lambda) - stepsize*[NNbias;NN{L}']*EE{L}; %L2 regularize
    end
    
    % benchmark
    for L=1:layers
        totEE(n) = totEE(n) + sum(abs(EE{L}));
    end
    
    % figures
    if any(figs==n)
        m=m+1;
        figure;
        subplot(2,2,1); imagesc(reshape(img,sz)); title(sprintf('target at iter %d',figs(m)));
        subplot(2,2,2); imagesc(reshape(NN{1}(1:szn),sz)); title(sprintf('produced at iter %d',figs(m)))
        subplot(2,2,3); imagesc(reshape(EE{layers}(1:szn),sz)); title(sprintf('error at iter %d',figs(m)));
        % save(sprintf('ArdiNet.%d.mat',m));
    end
    
    disp(n);
    n = n+1;
end


%% assess change in error over time
pl = totEE;
% pl(1:5) = 0;
%pl(pl>mean(pl)+std(pl) | pl<mean(pl)-std(pl)) = nan;
figure;
plot(pl);
% ylim([0 mean(pl)+std(pl)]);
