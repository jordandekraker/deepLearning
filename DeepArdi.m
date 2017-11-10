
layers = 5;
NN{1} = rand([500,2]);
NN{2} = rand([400,2]);
NN{3} = rand([300,2]);
NN{4} = rand([200,2]);
NN{5} = rand([100,2]);
WW{1} = rand([500,400])/500;
WW{2} = rand([400,300])/400;
WW{3} = rand([300,200])/300;
WW{4} = rand([200,100])/200;
WW{5} = rand([100,500])/100;

img = imread('SmileyFace8bitGray.png');
img = double(imresize(img,1/10));
img = img/max(img(:));
img = reshape(img,[400,1]);

stepsize=1;

n=0
while n<=10^3
    n = n+1
    %% in
    NN{1}(1:400,1) = img;
    
    %% forward
    for la = 1:layers
        % circular wrap
        laup = la+1;
        if laup>layers
            laup = laup-layers;
        end
        NN{laup}(:,2) = NN{laup}(:,1);
        NN{laup}(:,1) = t_func(NN{la}(:,1)'*WW{la})';
    end
    
    %% backward
    for la = 1:layers
        % circular wrap
        ladwn = la-1;
        if ladwn<1
            ladwn = ladwn+layers;
        end
        
        EE{n,ladwn} = NN{ladwn}(:,1)-NN{ladwn}(:,2);
        EE{n,ladwn} = EE{n,ladwn}.*(1-EE{n,ladwn}); %different for different t_func?
        WW{ladwn} = WW{ladwn} + stepsize*EE{n,ladwn}*NN{la}(:,1)';
    end
    
    %% push current value to past
%     for la = 1:layers
%         NN{la}(:,2) = NN{la}(:,1);
%     end
    %% out
    %NN{4} = out;
end
figure;
imagesc(reshape(NN{1}(1:400,1),[20,20]));
for n=1:1000
    for l=1:layers
        a(l)=sum(EE{n,l});
    end
    b(n) = sum(a);
end
figure;
plot(b);
%note: currently does not converge