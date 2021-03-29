clear; clc; 

for x = [50, 100, 125, 150, 175, 200]

    sigma = 1; ell = 0.5;
    
    GP = generateGP(x,x,sigma,ell);
    
    disp(strcat("GP-", num2str(x), " is generated."))
    
    Z = reshape(sampleGP(GP), [x,x]);
    
    disp("Z is generated.")
    clearvars GP;
    disp(strcat("GP-", num2str(x), " is deleted."))
    
    writematrix(Z, strcat("env", num2str(x), ".csv"));
    
    clear;
    
end

