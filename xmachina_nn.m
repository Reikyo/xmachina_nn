% xmachina_nn.m
% Darren Temple

% --------------------------------------------------------------------------------------------------------------------------
% Variables
% --------------------------------------------------------------------------------------------------------------------------

normalise                                        = true;
store_accuracy                                   = true;
verbose                                          = false;

if ~store_accuracy & verbose
  store_accuracy                                 = true;
end

N_epoch                                          = 200; % Default 100
epsilon                                          = 0.001; % Adversarial coefficient: 0.1 gave 0% training misclassification
eta                                              = 0.1; % Step size for gradient descent

% --------------------------------------------------------------------------------------------------------------------------

% Load the training data:
filename                                         = '../training_10000events.csv';
file                                             = fopen(filename);
file_full_linear                                 = textscan(file, '%s', 'Delimiter', ',');
fclose(file);

% Extract feature names and the rest of the data:
file_ncol                                        = 33;
file_full                                        = reshape(file_full_linear{1}, file_ncol, [])';
features                                         = file_full(1, 2:file_ncol-2)';
data                                             = str2double(file_full(2:end, 2:file_ncol-2));
w                                                = str2double(file_full(2:end, end-1));
t                                                = cell2mat(  file_full(2:end, end  ));
% Map from the given t = {'s', 'b'} to t = {1, 2}:
t                                                = (t == 's') + 2 * (t == 'b');
K                                                = 2; % Number of classes
[N, D]                                           = size(data);
clear file_full;

N_trn                                            = 5000;
N_tst                                            = N - N_trn;

data_trn                                         = data(      1:N_trn, :);
data_tst                                         = data(N_trn+1:end  , :);
clear data;

t_trn                                            = t(         1:N_trn);
t_tst                                            = t(   N_trn+1:end  );
clear t;

w_trn                                            = w(         1:N_trn);
w_tst                                            = w(   N_trn+1:end  );
clear w;

%for shuffle = 1:5
%  randperm_N_trn                                 = randperm(N_trn);
%  data_trn                                       = data_trn(randperm_N_trn, :);
%  t_trn                                          = t_trn(   randperm_N_trn);
%end

if normalise
  mean_data_trn                                  = mean(data_trn, 1);
  std_data_trn                                   = std( data_trn, 1, 1) + eps;
  data_trn                                       = data_trn  - repmat(mean_data_trn, [N_trn, 1]);
  data_trn                                       = data_trn ./ repmat( std_data_trn, [N_trn, 1]);
  data_tst                                       = data_tst  - repmat(mean_data_trn, [N_tst, 1]);
  data_tst                                       = data_tst ./ repmat( std_data_trn, [N_tst, 1]);
end

% --------------------------------------------------------------------------------------------------------------------------

% Create neural network data structure. Simple version, have weight vector per node, all nodes in a layer are same type.
% NN(i).weights is a matrix of weights, each row corresponds to the weights for a node at the next layer
% Note that bias term is added at the end
% i.e. a_k = NN(i).weights(:, k)' * z, where z is the vector of node outputs at the preceding layer

clear NN;
%clear NN_adv;
NN                                               = struct('type', '', 'weights', []);
%NN_adv                                           = struct('type', '', 'weights', []);
H                                                = 500; % Number of hidden nodes

NN(1).type                                       = 'sigmoid';
NN(1).weights                                    = randn(D+1, H);
NN(2).type                                       = 'softmax';
NN(2).weights                                    = randn(H+1, K) * 0.1;

%NN_adv(1).type                                   = 'sigmoid';
%NN_adv(1).weights                                = randn(D+1, H);
%NN_adv(2).type                                   = 'softmax';
%NN_adv(2).weights                                = randn(H+1, K) * 0.1;

% --------------------------------------------------------------------------------------------------------------------------

% AMS cost function variables:
%AMS_trn                                          = zeros(N_epoch, 1);
%AMS_tst                                          = zeros(N_epoch, 1);
%b_reg                                            = 10;

trn_accuracy                                     = [];
%trn_accuracy_adv                                 = [];
tst_accuracy                                     = [];
%tst_accuracy_adv                                 = [];

% --------------------------------------------------------------------------------------------------------------------------
% Calculation
% --------------------------------------------------------------------------------------------------------------------------

% Train neural network:

fprintf                 ('\nTraining ...\n\n');

for epoch = 1:N_epoch

  fprintf               ('epoch: %d/%d\n', epoch, N_epoch);

  for x_i = 1:N_trn

    [A, Z]                                       = feedforward(data_trn(x_i, :), NN);

% Output layer derivative:
% Assume classification with softmax
% Code for multiple hidden layers should use a for loop, but the first/last layers are special cases, which is all we have here
    dW2                                          = zeros(H+1, K);
    deltak                                       = Z{2};
    deltak(t_trn(x_i))                           = deltak(t_trn(x_i)) - 1;
    dW2                                          = repmat(deltak, H+1, 1) .* repmat([Z{1}, 1]', 1, K);

% Hidden layer derivative:
% Backpropagate error from output layer to hidden layer
    dW1                                          = zeros(D+1, H);
    deltaj                                       = Z{1} .* (1 - Z{1}) .* (NN(2).weights(1:end-1, :) * deltak')';
    dW1                                          = repmat(deltaj, D+1, 1) .* repmat([data_trn(x_i, :), 1]', 1, H);

% Adversarial training data:
    deltai                                       = data_trn(x_i, :) .* (1 - data_trn(x_i, :)) .* (NN(1).weights(1:end-1, :) * deltaj')';
    data_trn_adv                                 = data_trn(x_i, :) - epsilon * sign(deltai);
    [A_adv, Z_adv]                               = feedforward(data_trn_adv, NN);

% Adversarial output layer derivative:
    dW2_adv                                      = zeros(H+1, K);
    deltak_adv                                   = Z_adv{2};
    deltak_adv(t_trn(x_i))                       = deltak_adv(t_trn(x_i)) - 1;
    dW2_adv                                      = repmat(deltak_adv, H+1, 1) .* repmat([Z_adv{1}, 1]', 1, K);

% Adversarial hidden layer derivative:
    dW1_adv                                      = zeros(D+1, H);
    deltaj_adv                                   = Z_adv{1} .* (1 - Z_adv{1}) .* (NN(2).weights(1:end-1, :) * deltak_adv')';
    dW1_adv                                      = repmat(deltaj_adv, D+1, 1) .* repmat([data_trn_adv, 1]', 1, H); % Maybe data_trn(x_i, :) instead of data_trn_adv here???

% Apply the computed gradients in a stochastic gradient descent update (N.B. not actually stochastic right now as going through the dataset in the given order ...):
%    NN(2).weights                                = NN(2).weights - eta * dW2;
%    NN(1).weights                                = NN(1).weights - eta * dW1;
    NN(2).weights                                = NN(2).weights - eta * 0.5 * (dW2 + dW2_adv);
    NN(1).weights                                = NN(1).weights - eta * 0.5 * (dW1 + dW1_adv);

  end

  if store_accuracy
    [A_trn        , Z_trn    ]                   = feedforward(data_trn, NN);
%    [A_trn_adv    , Z_trn_adv]                   = feedforward(data_trn, NN_adv);
    [A_tst        , Z_tst    ]                   = feedforward(data_tst, NN);
%    [A_tst_adv    , Z_tst_adv]                   = feedforward(data_tst, NN_adv);
    [mvals_trn    , y_trn    ]                   = max(Z_trn{    end}, [], 2);
%    [mvals_trn_adv, y_trn_adv]                   = max(Z_trn_adv{end}, [], 2);
    [mvals_tst    , y_tst    ]                   = max(Z_tst{    end}, [], 2);
%    [mvals_tst_adv, y_tst_adv]                   = max(Z_tst_adv{end}, [], 2);
    N_incorrect_trn                              = sum(y_trn     ~= t_trn);
%    N_incorrect_trn_adv                          = sum(y_trn_adv ~= t_trn);
    N_incorrect_tst                              = sum(y_tst     ~= t_tst);
%    N_incorrect_tst_adv                          = sum(y_tst_adv ~= t_tst);
    trn_accuracy(    epoch)                      = length(find(y_trn     == t_trn)) / length(t_trn);
%    trn_accuracy_adv(epoch)                      = length(find(y_trn_adv == t_trn)) / length(t_trn);
    tst_accuracy(    epoch)                      = length(find(y_tst     == t_tst)) / length(t_tst);
%    tst_accuracy_adv(epoch)                      = length(find(y_tst_adv == t_tst)) / length(t_tst);
  end

  if verbose
    fprintf             ('  N_incorrect_trn    : %d\n'  , N_incorrect_trn);
%    fprintf             ('  N_incorrect_trn_adv: %d\n'  , N_incorrect_trn_adv);
    fprintf             ('  N_incorrect_tst    : %d\n'  , N_incorrect_tst);
%    fprintf             ('  N_incorrect_tst_adv: %d\n'  , N_incorrect_tst_adv);
    fprintf             ('  trn_accuracy       : %.4f\n', trn_accuracy(    epoch));
%    fprintf             ('  trn_accuracy_adv   : %.4f\n', trn_accuracy_adv(epoch));
    fprintf             ('  tst_accuracy       : %.4f\n', tst_accuracy(    epoch));
%    fprintf             ('  tst_accuracy_adv   : %.4f\n', tst_accuracy_adv(epoch));
  end

  %[A_trn    , Z_trn]                             = feedforward(data_trn, NN);
  %[A_tst    , Z_tst]                             = feedforward(data_tst, NN);
  %[mvals_trn, y_trn]                             = max(Z_trn{end}, [], 2);
  %[mvals_tst, y_tst]                             = max(Z_tst{end}, [], 2);
  %s                                              = sum(w_trn(find(y_trn == 1)));
  %b                                              = sum(w_trn(find(y_trn == 2)));
  %AMS_trn(epoch)                                 = sqrt(2 * ((s + b + b_reg) * log(1 + (s / (b + b_reg))) - s));
  %s                                              = sum(w_tst(find(y_tst == 1)));
  %b                                              = sum(w_tst(find(y_tst == 2)));
  %AMS_tst(epoch)                                 = sqrt(2 * ((s + b + b_reg) * log(1 + (s / (b + b_reg))) - s));

end

% --------------------------------------------------------------------------------------------------------------------------

% Test neural network:

fprintf                 ('\nTesting ...\n\n');

if ~store_accuracy
  [A_trn        , Z_trn    ]                     = feedforward(data_trn, NN);
%  [A_trn_adv    , Z_trn_adv]                     = feedforward(data_trn, NN_adv);
  [A_tst        , Z_tst    ]                     = feedforward(data_tst, NN);
%  [A_tst_adv    , Z_tst_adv]                     = feedforward(data_tst, NN_adv);
  [mvals_trn    , y_trn    ]                     = max(Z_trn{    end}, [], 2);
%  [mvals_trn_adv, y_trn_adv]                     = max(Z_trn_adv{end}, [], 2);
  [mvals_tst    , y_tst    ]                     = max(Z_tst{    end}, [], 2);
%  [mvals_tst_adv, y_tst_adv]                     = max(Z_tst_adv{end}, [], 2);
  N_incorrect_trn                                = sum(y_trn     ~= t_trn);
%  N_incorrect_trn_adv                            = sum(y_trn_adv ~= t_trn);
  N_incorrect_tst                                = sum(y_tst     ~= t_tst);
%  N_incorrect_tst_adv                            = sum(y_tst_adv ~= t_tst);
end

% Determine signal and background counts:

N_sig_t_trn                                      = sum(t_trn == 1);
N_bkg_t_trn                                      = sum(t_trn == 2);
N_sig_t_tst                                      = sum(t_tst == 1);
N_bkg_t_tst                                      = sum(t_tst == 2);

N_sig_y_trn                                      = sum(y_trn == 1);
N_bkg_y_trn                                      = sum(y_trn == 2);
N_sig_y_tst                                      = sum(y_tst == 1);
N_bkg_y_tst                                      = sum(y_tst == 2);

%N_sig_y_trn_adv                                  = sum(y_trn_adv == 1);
%N_bkg_y_trn_adv                                  = sum(y_trn_adv == 2);
%N_sig_y_tst_adv                                  = sum(y_tst_adv == 1);
%N_bkg_y_tst_adv                                  = sum(y_tst_adv == 2);

if (N_sig_y_trn + N_bkg_y_trn) ~= N_trn
  fprintf               ('ERROR: (N_sig_y_trn + N_bkg_y_trn) ~= N_trn\n');
end

if (N_sig_y_tst + N_bkg_y_tst) ~= N_tst
  fprintf               ('ERROR: (N_sig_y_tst + N_bkg_y_tst) ~= N_tst\n');
end

%if (N_sig_y_trn_adv + N_bkg_y_trn_adv) ~= N_trn
%  fprintf               ('ERROR: (N_sig_y_trn_adv + N_bkg_y_trn_adv) ~= N_trn\n');
%end

%if (N_sig_y_tst_adv + N_bkg_y_tst_adv) ~= N_tst
%  fprintf               ('ERROR: (N_sig_y_tst_adv + N_bkg_y_tst_adv) ~= N_tst\n');
%end

% --------------------------------------------------------------------------------------------------------------------------
% Output
% --------------------------------------------------------------------------------------------------------------------------

fprintf                 ('      N_epoch: %d\n', N_epoch)
fprintf                 ('      epsilon: %d\n', epsilon)
fprintf                 ('          eta: %f\n', eta)
fprintf                 ('\n')
fprintf                 ('     Training: Total: %4d\n', N_trn)
fprintf                 ('               N_sig: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
                         N_sig_y_trn    , (N_sig_y_trn     / N_trn) * 100, N_sig_t_trn, (N_sig_t_trn / N_trn) * 100);
%fprintf                 ('           N_sig_adv: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
%                         N_sig_y_trn_adv, (N_sig_y_trn_adv / N_trn) * 100, N_sig_t_trn, (N_sig_t_trn / N_trn) * 100);
fprintf                 ('               N_bkg: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
                         N_bkg_y_trn    , (N_bkg_y_trn     / N_trn) * 100, N_bkg_t_trn, (N_bkg_t_trn / N_trn) * 100);
%fprintf                 ('           N_bkg_adv: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
%                         N_bkg_y_trn_adv, (N_bkg_y_trn_adv / N_trn) * 100, N_bkg_t_trn, (N_bkg_t_trn / N_trn) * 100);
fprintf                 ('       Misclassified: %4d (%6.2f%%)\n', N_incorrect_trn    , (N_incorrect_trn     / N_trn) * 100);
%fprintf                 ('   Misclassified_adv: %4d (%6.2f%%)\n', N_incorrect_trn_adv, (N_incorrect_trn_adv / N_trn) * 100);
fprintf                 ('\n')
fprintf                 ('     Testing : Total: %4d\n', N_tst)
fprintf                 ('               N_sig: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
                         N_sig_y_tst    , (N_sig_y_tst     / N_tst) * 100, N_sig_t_tst, (N_sig_t_tst / N_tst) * 100);
%fprintf                 ('           N_sig_adv: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
%                         N_sig_y_tst_adv, (N_sig_y_tst_adv / N_tst) * 100, N_sig_t_tst, (N_sig_t_tst / N_tst) * 100);
fprintf                 ('               N_bkg: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
                         N_bkg_y_tst    , (N_bkg_y_tst     / N_tst) * 100, N_bkg_t_tst, (N_bkg_t_tst / N_tst) * 100);
%fprintf                 ('           N_bkg_adv: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
%                         N_bkg_y_tst_adv, (N_bkg_y_tst_adv / N_tst) * 100, N_bkg_t_tst, (N_bkg_t_tst / N_tst) * 100);
fprintf                 ('       Misclassified: %4d (%6.2f%%)\n', N_incorrect_tst    , (N_incorrect_tst     / N_tst) * 100);
%fprintf                 ('   Misclassified_adv: %4d (%6.2f%%)\n', N_incorrect_tst_adv, (N_incorrect_tst_adv / N_tst) * 100);
fprintf                 ('\n')

%close all;
%scrsz                   = get(groot,'ScreenSize');

%figure1                 = figure(1);
%figure1.Position        = [1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2];
%figure1.Name            = 'xmachina_nn';
%figure1.NumberTitle     = 'off';
%axes1                   = axes('Parent', figure1);
%hold off;
%plot1                   = plot(axes1, trn_accuracy, 'bo-');
%hold on;
%plot1                   = plot(axes1, tst_accuracy , 'ro-');
%hold off;
%axis                    (axes1, [1, N_epoch, 0, 1]);
%title                   (axes1, 'Training neural network with backpropagation');
%xlabel                  (axes1, 'Epoch');
%ylabel                  (axes1, 'Classification accuracy');
%legend                  (axes1, 'Training', 'Testing', 'Location', 'northwest');

%if savepdf
%  set                   (gcf, 'PaperPosition', [0, 0, 20, 10]);
%  set                   (gcf, 'PaperSize', [20, 10]);
%  pdfname               = 'q422';
%  if normalise
%    pdfname             = cat(2, pdfname, '_norm');
%  else
%    pdfname             = cat(2, pdfname, '_unnorm');
%  end
%  saveas                (gcf, pdfname, 'pdf');
%end

% Get predictions
%[A, Z]                                           = feedforward(data_tst, NN);
% Take max over output layer to get predictions
%[mvals, preds]                                   = max(Z{end}, [], 2);
% -1 to convert back to actual digits
%webpageDisplay          (X, TEST_INDS, preds-1, t_tst-1);
