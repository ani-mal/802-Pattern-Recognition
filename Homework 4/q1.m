
mu_1= 20;
mu_2=35;
sig_1=sqrt(5);
sig_2=sqrt(5);
rng('default')
rng(1);
N = 1000;
random1 = sig_1 .* randn(N, 1) + mu_1;
random2 = sig_2 .* randn(N, 1) + mu_2;

randomNumbs = [ random1' , random2' ]

result = zeros(1, 2000 )
randomNumbs = sort(randomNumbs)
h = [ 0.01 0.1 1 10]

densities = [0:1:55] 

for j=1:4
    for i = 1:55
       result(i) = parzen_window_estimation(h(j), i, randomNumbs, 2)
    end 
    figure(j) 
    plot( randomNumbs, result, 'g.', 'LineStyle', '-')
    title("h=" + h(j)) 
end 

function p = parzen_window_estimation(h, x, xi, d)
    n = length(xi) 
    hn = h^d 
    
    phi =  (1/sqrt(2*pi*hn))*exp(-0.5*((x*ones(1,n)-xi)/h).^2)
    p = sum(phi)/(hn*n)
end
