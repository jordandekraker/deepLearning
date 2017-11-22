function img = saccade(origimg,fix)

sz = [100,100];
szn = sz(1)*sz(2);
img = lensdistortJD(origimg,100,fix);
img(img==255) = nan;
[a,b] = find(~isnan(img));
img = img(min(a):max(a),min(b):max(b));
img = inpaintn(img);
img = imresize(img,sz);
img = 2*img/max(img(:))-1; %normalize
% imagesc(img);
img = reshape(img,[1,szn]);
end