load data.txt

train_1 =  data(1:250,:)
test_1 = data(251:500, :)

train_2 = data(501:750, : )
test_2 = data(751:1000, : )

train_3 = data(1001:1250, :)
test_3 = data(1251:1500, :)


%% Part a) 
mu_1 = [0 0]
mu_2 = [10 0] 
mu_3 = [5 5]

identity = eye(2) 
var_1 = 4*identity 
var_2 = var_1 
var_3 = 5*identity

mu = [ mu_1; mu_2; mu_3 ]


% class 1 
[c1, c2, c3 ] = classify( test_1(:,1:2), mu, var_1, var_2, var_3)

first = [c1, c2, c3 ] 

% class 2
[c1, c2, c3 ] = classify( test_2(:,1:2), mu, var_1, var_2, var_3)

second = [ c1, c2, c3 ]

%class 3
[c1, c2, c3 ] = classify( test_3(:,1:2), mu, var_1, var_2, var_3)

third = [ c1, c2, c3 ]


ConfusionMatrix =  [ first; second; third ] 
figure(1)
confusionchart(ConfusionMatrix)
title("2a")
%% Estimating mu and variance 

estMu_1 = [ sum(train_1(:, 1)), sum(train_1(:,2)) ] / size(train_1,1) 
estMu_2 = [ sum(train_2(:, 1)), sum(train_2(:,2)) ] / size(train_2,1) 
estMu_3 = [ sum(train_3(:, 1)), sum(train_3(:,2)) ] / size(train_3,1) 

estSigma_1 =  estimate_sigma( train_1(:,1:2), estMu_1 ) 
estSigma_2 =  estimate_sigma( train_2(:,1:2), estMu_2 ) 
estSigma_3 =  estimate_sigma( train_3(:,1:2), estMu_3 ) 

estMu = [ estMu_1; estMu_2; estMu_3]
% class 1 
[c1, c2, c3 ] = classify( test_1(:,1:2), estMu, estSigma_1, estSigma_2, estSigma_3)

first = [c1, c2, c3 ] 

% class 2
[c1, c2, c3 ] = classify( test_2(:,1:2), estMu, estSigma_1, estSigma_2, estSigma_3)

second = [ c1, c2, c3 ]

%class 3
[c1, c2, c3 ] = classify( test_3(:,1:2), estMu, estSigma_1, estSigma_2, estSigma_3)

third = [ c1, c2, c3 ]


ConfusionMatrix2 =  [ first; second; third ] 
figure(2)
confusionchart(ConfusionMatrix2)
title("2b Estimated mu and Sigma ")
%% 2 c 

[c1, c2, c3 ] =  classifyParsenWindow(test_1(:,1:2), train_1(:,1:2), train_2(:,1:2), train_3(:,1:2), 1, 2) 

first =  [c1, c2, c3 ]

[c1, c2, c3 ] =  classifyParsenWindow(test_2(:,1:2), train_1(:,1:2), train_2(:,1:2), train_3(:,1:2), 1, 2) 

second =  [c1, c2, c3 ]

[c1, c2, c3 ] =  classifyParsenWindow(test_3(:,1:2), train_1(:,1:2), train_2(:,1:2), train_3(:,1:2), 1, 2) 

third =  [c1, c2, c3 ]


ConfusionMatrix3 =  [ first; second; third ] 
figure(3)
confusionchart(ConfusionMatrix3)

%% 2d k-nearest 


%%
function [c1, c2, c3 ] =  classifyParsenWindow(test, x1, x2, x3, h, d)
   c1 = 0
   c2 = 0
   c3 = 0
  for i =1:size(test,1)
      [y1, y2, y3 ] = parsenWindow(test(i,1:2), x1, x2, x3, h, d)  
      
      if y1 == max( [y1, y2, y3] )
          c1 = c1 +  1
      end
      
      if y2 == max( [y1, y2, y3] )
          c2 = c2 +  1
      end
      
      if y3 == max( [y1, y2, y3] )
          c3 = c3 + 1
      end
  end 
end

function [c1, c2, c3] = classify( data, mu, var_1, var_2, var_3 )
  
  y1 = mvnpdf(data,  mu(1,:), var_1)
  y2 = mvnpdf(data,  mu(2,:), var_2)
  y3 = mvnpdf(data,  mu(3,:), var_3)
 
  
  [m, n] = size(y1)
  
  c1 = 0
  c2 = 0
  c3 = 0
  
  for i = 1:m 
      if y1(i) == max( [y1(i), y2(i), y3(i)] )
          c1 = c1 +  1
      end
      
      if y2(i) == max( [y1(i), y2(i), y3(i)] )
          c2 = c2 +  1
      end
      
      if y3(i) == max( [y1(i), y2(i), y3(i)] )
          c3 = c3 + 1
      end
      
  end 
end 

function sigma =  estimate_sigma( data, mu )
    [m,n] = size(data) 
    sigma = zeros(n,n)
    mu_vec = ones(m,n).*mu 
    diff = ( data - mu_vec)
   
    trans = diff'
    sigma = trans*diff
    sigma = 1/m *sigma 
end


function [result1, reuslt2, result3 ] = parsenWindow(test, x1, x2, x3, h, d)
    
    result1 = 0 
    reuslt2 = 0 
    result3 = 0
    sigma = h^2*eye(d,d)
    for i=1:length(x1)
        result1 = result1 + gaussianKernel( sigma, test, x1(i,1:2),d) 
        reuslt2 = reuslt2 + gaussianKernel( sigma, test, x2(i,1:2),d) 
        result3 = result3 + gaussianKernel( sigma, test, x3(i,1:2),d) 
    end 
    
end

function gauss = gaussianKernel(Sigma, x, xi,d)
    SigmaDet = det(Sigma)
    SigmaInv = inv(Sigma)
    gauss =  (1/((2*pi)^(d/2)*sqrt(SigmaDet)))*exp(-0.5*(x-xi)*SigmaInv*(x-xi)')
end 