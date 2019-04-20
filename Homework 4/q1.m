dist1 = makedist('Normal', 20, 5) 
random1 = random(dist1, 1, 100)

dist2 = makedist('Normal', 35, 5) 
random2 = random(dist1, 1, 100)

randomNumbs = [ random1 , random2 ]

result = zeros(1, 200 )
randomNumbs = sort(randomNumbs)
h = [ 0.01 0.1 1 10 20 100 ] 

densities = [0:1:55] 



for j=1:6
    for i = 1:55
       pn = gauss_parzen_window_dens(h(j), randomNumbs, densities(i))
       result(i) = pn 
    end 
    figure(j) 
    plot( randomNumbs, result, 'r.', 'LineStyle', '-')
    title("h=" + h(j)) 
end 


function [y] = gaussian_kernel(u)
  [d,n] = size(u);
  y=zeros(1,n);
  for i=1:n
    y(i) = gaussian(u(:,i));
  end
end

 
function [y] = gaussian(u)
  u=u(:);
  d = length(u);
  y = exp(-(u'*u)/2)/((2*pi)^(d/2));
end

function [pn] = gauss_parzen_window_dens(h, u, v)
  [d, n] = size (u);
  hn=h/sqrt(n);
  phi = gaussian_kernel(( v*ones(1,n) - u)/hn)
  pn = sum(phi)/(hn);
end
 
