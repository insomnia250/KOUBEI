function loss = lossFun(pred)
global c
loss=0;
for t=1:length(c)
    loss = loss+abs(pred-c(t))/abs(pred+c(t));
end
loss = loss/length(c);