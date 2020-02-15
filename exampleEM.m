%exMax example script
%Creates 2 normal (gaussian) distributions. Then draw 400 points from either 
%distributions - the Expectation Maximization scipt exMax will
%estimates the parameters. Fascinating.


%first initialize some params:
numPoints=400; %number of samples to draw from one of the 2 gaussians


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create 2 normal distributions in R~2 parameterized by :
%means 3 & 7
Mu=      [1 2 ;5 6];
Sigma =[2 2; 3 3];
myPi=[.3 .7]; %aka mixing coefficients
samples=zeros(400,2); %will hold samples taken from one of the 2 gaussians
figure; hold on;
for i=1:numPoints, %sample from the 2 gaussians
        if rand>myPi(2),
            samples(i,:)=[normrnd(Mu(1,1),Sigma(1,1));normrnd(Mu(1,2),Sigma(1,2))];
            plot(samples(i,1),samples(i,2),'rx');
        else
            samples(i,:)=[normrnd(Mu(2,1),Sigma(2,1));normrnd(Mu(2,2),Sigma(2,2))];
            plot(samples(i,1),samples(i,2),'bx');
        end    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Guess some initial parameters to help EM estimate the hidden gaussians' parameters:
newMu= [1 2; 5 6];
newCov=[2 2; 3 3];
newPi=[.5 .5];              

%call EM function, using the correct # of components(2)
[bestU,bestCov,bestPi]=exMax(samples,newMu,newCov,newPi);
disp(strcat('estimated U=',num2str(bestU)));
disp(strcat('estimated Cov=',num2str(bestCov)));
disp(strcat('estimated Pi=',num2str(bestPi)));