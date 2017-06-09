clear all;clc;
datamat = csvread('..\RULE\RULE7\datamat.csv',0,0);
optavg = zeros(2000,3);
global c
for i = 1:2000
    c = datamat(i,2:end);
    [y ,~]= fminbnd(('lossFun'),min(c),max(c));
    
    optavg(i,1) = datamat(i,1);
    optavg(i,2) = y;
    optavg(i,3) = mean(c);
end

filename_i=['optavg.csv'];
csvwrite(filename_i,optavg)

% 
% c = [1,1,30,40,50,1,1,32,43,52];
% for i = 1:length(c)
%     c(i) = c(i) + normrnd(0,c(i)*0.2);
% end
% c
% [y ,minloss]= fminbnd(('lossFun'),c(1),c(end))
