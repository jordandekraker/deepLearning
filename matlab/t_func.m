function x = t_func(x)
%ReLu
%x(x<0)=0;

%sigmoid
x = 1./(1+exp(-x));
end