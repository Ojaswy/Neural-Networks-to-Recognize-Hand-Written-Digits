function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
for i = 1:m
    RX = repmat(X(i,:),num_labels,1);
    RX = RX .* all_theta;
    SX = sum(RX,2);
    [val, index] = max(SX);
    p(i) = index;
end

end
