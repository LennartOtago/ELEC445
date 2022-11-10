d = double(imread('jupiter1.tif'));
figure(1); imagesc(d); colormap(gray); axis equal

xpos = 235; ypos = 86; % Pixel at centre of satellite
h = d(ypos+[-16:15],xpos+[-16:15]);

h = h./sum(sum(h));    % Normalize point-spread function