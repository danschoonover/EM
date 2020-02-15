function[bestU,bestCov,bestPi]=exMax(data,Ui,Covi,Pii)
%uses the expectation maximization algorithm to estimate from data the means,
%covariances and mixing coefficients for 'numComps' number of latent
%variables.

% data=samples;
% Ui=Mu;
% Covi=Sigma;
% Pii=myPi;

%initialize params
% Ui=      [1 2 ;1 2];
% Covi =[4 3; 4 3];
% Pii=[.2 .8]; %aka mixing coefficient

change=.0001;    %threshold for log-likelihood change

meanEst=Ui;
CovEst=Covi;
PiEst=Pii;

%sizing the input space
datarows=size(data,1); %400 
datacols=size(data,2); %2

estrows=size(Ui,1); %dimension of input (should equal datacols)
estcols=size(Ui,2);   %number of gaussians to estimate

Nk=zeros(estrows,estcols); %number of members for each latent gaussian
responsibilities=zeros(datarows,datacols,estcols); % responsiblities of latent variable for being resp for data 'n'

likeVec=0; %will store the log likelihood history (for plotting later)
temp=0;




%step 1 %

%initialize means, Covariances and mixing coefficients
U=Ui;
Cov=Covi;
Pi=Pii;

%and evaluate log likelihood for current params
lnofpofx=0;
temp1=0;
temp2=0;

for i=1:datarows, %sum over N
    for j=1:datacols,
        
        for k=1:estcols,        %sum over k
            temp1=[temp1,Pi(k)*normpdf(data(i,j),meanEst(k,j),CovEst(k,j))]; %data ~ 400x2
        end
        temp2=[temp2,log(sum(temp1))];
    end
end

lnofpofx=(sum(temp2));

loglik=lnofpofx;  %initial log likelihood
likeVec=loglik; %loglikelihood history
    
while(1)          
    
    %%%%%%
    %E STEP    %               (step 2:evaluate responsibilities)
    %%%%%%

    for i=1:datarows
        for j=1:datacols,             %sum over N              
            
            for k=1:estcols,            %sum over K                
                num=Pi(k)*normpdf(data(i,j),meanEst(k,j),CovEst(k,j));  %numerator of responsibilities
                responsibilities(i,j,k)=num/sum(Pi'.*normpdf(data(i,j),meanEst(:,j),CovEst(:,j)));                    
            end                                               
        end            
    end

    %%%%%%
    %M STEP  %               (step 3: maximize (re-estimate) parameters using new responsibilities)
    %%%%%%    
    
    %update Nk (number of points in each class 'k')
    for k=1:estcols,
        for j=1:estrows
            Nk(j,k)=sum(sum(responsibilities(:,j,k)));
        end
    end
    
    %re-estimate means
    for k=1:estcols,
        for j=1:datacols,            
            meanEst(k,:)=sum((responsibilities(:,:,k).*data(:,:)))/Nk(k);                                   
        end
    end
    
    
    %re-estimate covariances            
    for k=1:estcols
        for j=1:datacols
        temp=sum(sum(responsibilities(:,j,k).*(data(:,j)-meanEst(k))*(data(:,j)-meanEst(k))'));                
        covEst(k)=temp/Nk(k);
        end
    end
    
    %re-estimate Pi's  (mixing coefficients)
    for k=1:estcols
        PiEst(k)=Nk(k)/numel(data);
    end
    
    %%%%%
    %step 4        
    %%%%%
    

%evaluate log likelihood for current params
lnofpofx=0;

temp1=0;
temp2=0;

for i=1:datarows, 
        for j=1:datacols,           %sum over N
            temp1=0;
           for k=1:estcols
                temp1=[temp1,log(sum(Pi(k).*normpdf(data(i,j),meanEst(k,j),CovEst(k,j))))];
           end 
           temp2=[temp2, temp1];
        end        
    end
            lnofpofx=log(sum(temp2));
      
    loglik=sum(lnofpofx);  %current log likelihood
    likeVec=[likeVec,loglik]; %loglikelihood history           

%     if change is less than threshold, were done
    if (likeVec(end)-likeVec(end-1)<change)        
        break %done with EM algorithm, break out of loop
    end        
    
end %otherwise, start over


%lets plot loglik of each iteration to see it converge:
figure; 
plot(likeVec,'b');

%return results
bestU=meanEst
bestCov=covEst
bestPi=PiEst

return