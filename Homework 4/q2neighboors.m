load data.txt

train_1 =  data(1:250,:)
test_1 = data(251:500, :)

train_2 = data(501:750, : )
test_2 = data(751:1000, : )

train_3 = data(1001:1250, :)
test_3 = data(1251:1500, :)


train = [ train_1; train_2; train_3 ]
test = [ test_1; test_2; test_3 ]

ypredict = zeros(1,750) 


for i=1:length(test) 
    dist = 10000000
    for j=1:length(train)
        d = pdist( [test(i,1:2); train(j,1:2) ] ,'euclidean')
        if d < dist 
            dist = d 
            ypredict(i) = train(j, 3)
        end 
    end 
end 

confusion_matrix = confusionmat( ypredict, test(:,3))

figure(1)
confusionchart(confusion_matrix)
