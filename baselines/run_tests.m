function run_tests(npz_files, views, save_out)

if nargin < 2
  views = 3;
end
if nargin < 3
  save_out = false;
end

npymatlab_path = 'npy-matlab/npy-matlab';
pathCell = regexp(path, pathsep, 'split');
onPath = any(strcmp(npymatlab_path, pathCell));
if ~onPath
  addpath(npymatlab_path);
end
[~,~,~] = mkdir('/tmp/unzip');
if save_out
  [~,~,~] = mkdir('Adjmats');
end

v = views;
p = 80;
n = p*v;
dimGroups = ones(v,1)*p;
params015.maxiter = 15;
params025.maxiter = 25;
params050.maxiter = 50;
params100.maxiter = 100;
params200.maxiter = 200;

metric_info = { ...
      { 'l1',      'L1: %.03e, ',              @mean                 }, ...
      { 'l2',      'L2: %.03e, ',              @mean                 }, ...
      { 'bce',     'BCE: %.03e, ' ,            @mean                 }, ...
      { 'ssame_m', 'Same sim: %.03e ' ,        @mean                 }, ...
      { 'ssame_s', '+/- %.03e, ' ,             @(x) sqrt(mean(x.^2)) }, ...
      { 'sdiff_m', 'Diff sim: %.03e ' ,        @mean                 }, ...
      { 'sdiff_s', '+/- %.03e, ' ,             @(x) sqrt(mean(x.^2)) }, ...
      { 'roc',     'Area under ROC: %.03e, ' , @mean                 }, ...
      { 'pr',      'Area under P-R: %.03e, ' , @mean                 }, ...
};
metrics = cell(length(metric_info),1);
for i = 1:length(metric_info)
  metrics{i} = zeros(length(npz_files),1);
end
%  'MatchALS400Iter', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 400); ...
%  'PGDDS200Iter', @(W) PGDDS(W, dimGroups, p, params200); ...
%  'PGDDS100Iter', @(W) PGDDS(W, dimGroups, p, params100); ...
test_fns = { ...
 'Spectral', @(W)  myspectral(W, p); ...
 'MatchALS015Iter', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 15); ...
 'MatchALS025Iter', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 25); ...
 'MatchALS050Iter', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 50); ...
 'MatchALS100Iter', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 100); ...
 'PGDDS015Iter', @(W) PGDDS(W, dimGroups, p, params015); ...
 'PGDDS025Iter', @(W) PGDDS(W, dimGroups, p, params025); ...
 'PGDDS050Iter', @(W) PGDDS(W, dimGroups, p, params050); ...
};

saveout_str = '%sOutputs/%04d.npy';
for test_fn_index = 1:size(test_fns,1)
  test_fn_tic = tic;
  test_fn = test_fns{test_fn_index,2};
  fid = fopen(sprintf('%sTestErrors.log', test_fns{test_fn_index,1}), 'w');
  if save_out
    [~,~,~] = mkdir(sprintf('%sOutputs', test_fns{test_fn_index,1}))
  end
  fprintf('%s Method:\n', test_fns{test_fn_index,1})
  test_index = 0;
  for npz_index = 1:length(npz_files)
    fprintf('Matrix %03d of %03d\r', npz_index, length(npz_files))
    [ W, Agt ] = load_npz(npz_files{npz_index});
    tic;
    A_output = test_fn(W);
    Ah = max(0,min(1,A_output));
    run_time = toc;
    values = evaluate_tests(Ah, Agt);
    for metric_idx = 1:length(metrics)
      metrics{metric_idx}(npz_index) = values(metric_idx);
    end
    disp_values(metric_info, fid, npz_index, values, run_time);
    test_index = test_index + 1;
    if save_out
      output_name = sprintf(saveout_str, test_fns{test_fn_index,1}, npz_index);
      writeNPY(single(Ah), output_name);
      adjmat_name = sprintf('Adjmats/%04d.npy', npz_index);
      if ~exist(adjmat_name)
        writeNPY(Agt, adjmat_name)
      end
    end
  end
  fprintf('\n')
  fclose(fid);
  means = zeros(length(metrics),1);
  for metric_idx = 1:length(metrics)
    means(metric_idx) = metric_info{metric_idx}{3}(metrics{metric_idx});
  end
  disp_values(metric_info, 1, test_fn_index, means, run_time);
  fprintf(1, 'Total time: %.03f seconds\n', toc(test_fn_tic));
end

disp('Finished');

end

function disp_values(metric_info, fid, idx, values, time)
  fprintf(fid, '%06d Errors: ', idx);
  for i = 1:length(values)
    fprintf(fid, metric_info{i}{2}, values(i));
  end
  fprintf(fid, 'Time: %.03e\n', time);
end

function [ means ] = get_metric_means(metrics)
end

function [ values ] = evaluate_tests(Ah, Agt)
  [l1, l2, bce] = testOutput_soft(Ah,Agt);
  [ssame, ssame_std, sdiff, sdiff_std] = testOutputhist(Ah,Agt);
  % [roc, pr] = testOutput_roc_pr(Ah,Agt);
  pr = 0;
  roc = 0;
  values = [ l1, l2, bce, ssame, ssame_std, sdiff, sdiff_std, roc, pr ];
end


function [ssame, ssame_std, sdiff, sdiff_std] = testOutputhist(Ah,Agt)

N = sum(sum(Agt));
M = sum(sum(1-Agt));
ssame = sum(sum(Ah.*Agt)) / N;
ssame_std = sqrt(sum(sum((Ah.*Agt).^2)) / N - ssame^2);
sdiff = sum(sum(Ah.*(1-Agt))) / M;
sdiff_std = sqrt(sum(sum((Ah.*(1-Agt)).^2)) / M  - sdiff^2);

end

function [l1, l2, bce] = testOutput_soft(Ah,Agt)

l1  = mean2(abs(Ah-Agt));
l2  = mean2((Ah-Agt).^2);
bce = -mean2(Agt.*log2(eps+Ah) + (1-Agt).*log2(eps+1-Ah));

end

function [roc, pr] = testOutput_roc_pr(Ah,Agt)

[TP, TN, FP, FN] = compute_thresh_errs(Ah, Agt);
m = length(TP);
FPR = (FP ./ max(1e-8, FP + TN));
TPR = (TP ./ max(1e-8, TP + FN));
precision = (TP ./ max(1e-8, TP + FP));
recall    = (TP ./ max(1e-8, TP + FN));
% disp(size(FPR))
% disp(size(TPR))
% disp(class(FPR))
% disp(class(TPR))
roc = abs(trapz(FPR, TPR));
pr  = abs(trapz(precision, recall));

end

function [TP, TN, FP, FN] = compute_thresh_errs(output, adjmat, N_cutoffs)
  if nargin < 3
    N_cutoffs = 2048;
  end
  a = int32(adjmat);
  M_T = sum(a);
  M_F = numel(a) - M_T;

  TP = zeros(N_cutoffs, 1);
  TN = zeros(N_cutoffs, 1);
  FP = zeros(N_cutoffs, 1);
  FN = zeros(N_cutoffs, 1);
  for idx = 1:N_cutoffs
    i = N_cutoffs - idx;
    thresh = (1.0*i) / (N_cutoffs-1);
    o = int32(output > thresh);
    [ TP_, TN_, FP_, FN_ ] = calc_classifications(o,a);
    TP(idx) = double(TP_);
    TN(idx) = double(TN_);
    FP(idx) = double(FP_);
    FN(idx) = double(FN_);
  end
end

function [ TP, TN, FP, FN ] = calc_classifications(o, a)
  TP = sum(sum(o.*a));
  TN = sum(sum((1-o).*(1-a)));
  FP = sum(sum(o.*(1-a)));
  FN = sum(sum((1-o).*a));
end

function [W, Agt] = load_npz(npz_file)
  unzip(npz_file, '/tmp/unzip');
  AdjMat = readNPY('/tmp/unzip/AdjMat.npy');
  TrueEmbedding = readNPY('/tmp/unzip/TrueEmbedding.npy');
  W = squeeze(double(AdjMat)) + eye(size(AdjMat));
  Xgt = squeeze(double(TrueEmbedding));
  Agt = Xgt*Xgt';
end



