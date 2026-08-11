%% MARS FIRST-ORDER ZONAL COUPLING CHANNELS -- NATIVE-MATLAB ANCHOR (TASK-029)
% TASK-028 (Python) established a result now cited in the NASA proposal: for a
% degree-2 zonal tide, TWO even zonal rheology harmonics -- (2,0) AND (4,0),
% not (4,0) alone -- couple the forcing mode back to itself at FIRST order,
% with OPPOSITE signs, cancelling ~91%. That result had no independent anchor.
% This driver rebuilds it with the native MATLAB LOV3D solver, on the exact
% committed 4-layer Mars model, so the claim does not rest on the Python port
% plus a coupling-coefficient inspection alone.
%
% Three parts (mirroring pylov3d exactly):
%   PART 1  scaling exponents.  Isolate a single zonal rheology harmonic (n,0),
%           scale its amplitude by eps in {1e-3,1e-2}, solve forcing (2,0), and
%           fit exponent = log10( |shift(1e-2)| / |shift(1e-3)| ).  Python:
%             (2,0) -> ~1.00 (FIRST order -- the new claim)
%             (4,0) -> ~1.00 (control)
%             (3,0) -> ~2.00 (control)
%   PART 2  signed contributions on the DWAK InSight-Moho field, each channel
%           isolated then LINEARLY extrapolated to full physical amplitude
%           (shift(eps)/eps).  Python (method 'combination', Nrbase=30, order 2):
%             (2,0) alone   = -1.5901e-5
%             (4,0) alone   = +1.4528e-5
%             both together = -1.3738e-6   (91.4% cancellation; additive to 0.1%)
%           THE SIGNS ARE THE DECISIVE OUTCOME: if MATLAB flips a sign the
%           proposal's framing is wrong and must be pulled back the same day.
%   PART 3  coupling coefficients themselves, read straight out of the native
%           get_couplings() Coup array for rheology (1,0)..(6,0) against the
%           (2,0) forcing self-coupling.  Python coupling_coefficients gives
%           max|C| = 0.6389 (2,0), 0.8571 (4,0), identically 0 for (1,0),(3,0),
%           (5,0),(6,0).  Anchors the selection rule directly, not via response.
%
% The DWAK complex crust mu_variable is read VERBATIM from
% data/mars/mars_dwak_mu_variable.npz (written by
% scripts/export_mars_dwak_mu_variable.py) so MATLAB is handed the identical
% field Python used -- isolating the solver's sign convention from any
% spherical-harmonic re-derivation.  The 4-layer model arrays are copied
% VERBATIM from build_mars_model (same as scripts/mars_lateral_cross_check.m).
% Purely elastic: eta0 OMITTED on every layer (NaN poisons the MATLAB solve).
%
% Run headless from the repo root:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch \
%       "run('scripts/mars_first_order_channels.m')"

clearvars
clc

% --- put the LOV3D src on the path (repo root = parent of scripts/) --------
mfile_name = mfilename('fullpath');
if isempty(mfile_name) || contains(mfile_name, 'LiveEditorEvaluationHelper')
    mfile_name = matlab.desktop.editor.getActiveFilename;
end
[pathstr, ~, ~] = fileparts(mfile_name);
repo_root = fileparts(pathstr);   % scripts/ -> repo root
cd(repo_root);
addpath(genpath(repo_root));

coupling_dir = fullfile(repo_root, 'data', 'couplings');
if ~isfolder(coupling_dir); mkdir(coupling_dir); end

%% INTERIOR MODEL (verbatim from pylov3d.mars.build_mars_model)
Interior_Model(1).R0     = 1830.0;
Interior_Model(1).rho0   = 6128.075995512780;
Interior_Model(1).rho0_2 = 6128.075995512780;

Interior_Model(2).R0   = 2340.0;
Interior_Model(2).rho0 = 4136.503827041973;
Interior_Model(2).Ks0  = 160e9;
Interior_Model(2).mu0  = 96482476610.2174;

Interior_Model(3).R0   = 3339.5;
Interior_Model(3).rho0 = 3400.0;
Interior_Model(3).Ks0  = 115e9;
Interior_Model(3).mu0  = 67537733627.15218;

Interior_Model(4).R0   = 3389.5;
Interior_Model(4).rho0 = 2900.0;
Interior_Model(4).Ks0  = 70e9;
Interior_Model(4).mu0  = 30e9;

Interior_Model(1).Delta_rho0 = Interior_Model(1).rho0_2 - Interior_Model(2).rho0;

crust_matlab_idx = 4;   % python crust index 3 + 1

%% DWAK LATERAL RIGIDITY FIELD (verbatim from the committed npz)
npz_path = fullfile(repo_root, 'data', 'mars', 'mars_dwak_mu_variable.npz');
lv = read_mars_mu_variable_npz(npz_path);
assert(all(lv.layer_idx == 3), 'npz has non-crust layer_idx entries');

dwak_n   = double(lv.n(:));
dwak_m   = double(lv.m(:));
dwak_amp = lv.amp_real(:) + 1i*lv.amp_imag(:);
dwak_mu_variable = [dwak_n, dwak_m, dwak_amp];   % full field [n m amp]

%% NUMERICS (match pylov3d exactly: method 'combination', Nrbase 30, order 2)
Numerics.Nlayers = 4;
Numerics.method = 'combination';
Numerics.Nrbase = 30;
Numerics.perturbation_order = 2;
Numerics.solution_cutoff = 12;
Numerics.load_couplings = 2;
Numerics.Nenergy = 12;
Numerics.rheology_cutoff = 2;
Numerics.parallel_sol = 0;
Numerics.parallel_gen = 0;
Numerics.coupling_file_location = [coupling_dir filesep];

[Numerics, Interior_Model] = set_boundary_indices(Numerics, Interior_Model, 'verbose');
if ~(Numerics.Nlayers == length(Interior_Model))
    error('Numerics.Nlayers must equal length(Interior_Model)');
end

FORCING_TD = 44387.62;   % s (MARS_FORCING_TD); elastic => frequency-independent

%% OUTPUT / LOG
out_dir = fullfile(repo_root, 'data', 'tests', 'mars');
if ~isfolder(out_dir); mkdir(out_dir); end
log_path = fullfile(out_dir, 'mars_first_order_channels.log');
logf = fopen(log_path, 'w');
logp = @(varargin) both_print(logf, varargin{:});

logp('\n========== MARS FIRST-ORDER ZONAL CHANNELS (native MATLAB LOV3D) ==========\n');
logp('  TASK-029 anchor for the (2,0)/(4,0) first-order self-coupling result\n');
logp('  MATLAB %s\n', version);
logp('  method=%s  Nrbase=%d  perturbation_order=%d  (matching pylov3d)\n', ...
    Numerics.method, Numerics.Nrbase, Numerics.perturbation_order);
logp('  DWAK field: %d crust mu_variable entries from %s\n', ...
    size(dwak_mu_variable,1), 'data/mars/mars_dwak_mu_variable.npz');

%% Forcing struct helper (n=2 tidal, order m set per call)
mk_forcing = @(m) struct('Td', FORCING_TD, 'n', 2, 'm', m, 'F', 1);

%% ------------------------------------------------------------------------
%% Uniform (no-lateral) reference k2 for forcing (2,0)
%% ------------------------------------------------------------------------
Forcing20 = mk_forcing(0);
IM_uni = Interior_Model;
IM_uni(crust_matlab_idx).mu_variable = [];
IM_uni = get_rheology(IM_uni, Numerics, Forcing20);
[Love_Uni, ~] = get_Love(IM_uni, Forcing20, Numerics, 'verbose');
k2_uniform = get_mode_k(Love_Uni, 2, 0);
logp('\n  k2 uniform (no lateral), forcing (2,0) : %.12f\n', k2_uniform);

%% ========================================================================
%% PART 1: SCALING EXPONENTS (isolated single zonal rheology harmonic)
%% ========================================================================
logp('\n---- PART 1: scaling exponents (forcing (2,0)) --------------------------\n');
logp('    exponent = log10( |shift(1e-2)| / |shift(1e-3)| );  shift = k2 - k2_uniform\n');

exp_targets = struct('deg', {2, 4, 3}, 'py', {1.003, 1.001, 2.003}, ...
                     'order', {'1st', '1st', '2nd'});
part1 = struct([]);
for t = 1:numel(exp_targets)
    ndeg = exp_targets(t).deg;
    base = isolate_zonal(dwak_mu_variable, ndeg);   % [n 0 amp] rows, m==0 only
    if isempty(base)
        logp('    (%d,0): NO m==0 entry in DWAK field -- skipped\n', ndeg);
        continue
    end
    s_lo = channel_shift(Interior_Model, crust_matlab_idx, base, 1e-3, ...
                         Forcing20, Numerics, k2_uniform);
    s_hi = channel_shift(Interior_Model, crust_matlab_idx, base, 1e-2, ...
                         Forcing20, Numerics, k2_uniform);
    expo = log10(abs(s_hi) / abs(s_lo));
    logp('    (%d,0): shift(1e-3)=%+.4e  shift(1e-2)=%+.4e  exponent=%.4f  (py %.3f, %s order)\n', ...
        ndeg, s_lo, s_hi, expo, exp_targets(t).py, exp_targets(t).order);
    r.deg = ndeg; r.shift_lo = s_lo; r.shift_hi = s_hi; r.exponent = expo; r.py = exp_targets(t).py;
    if isempty(part1), part1 = r; else, part1(end+1) = r; end %#ok<SAGROW>
end

%% ========================================================================
%% PART 2: SIGNED CONTRIBUTIONS + CANCELLATION (DWAK, full amplitude)
%% ========================================================================
logp('\n---- PART 2: signed channel contributions on DWAK -----------------------\n');
logp('    each channel isolated, eps=1e-2, extrapolated to full amplitude (shift/eps)\n');

eps2 = 1e-2;
b20 = isolate_zonal(dwak_mu_variable, 2);
b40 = isolate_zonal(dwak_mu_variable, 4);
b_both = [b20; b40];

s20  = channel_shift(Interior_Model, crust_matlab_idx, b20,   eps2, Forcing20, Numerics, k2_uniform) / eps2;
s40  = channel_shift(Interior_Model, crust_matlab_idx, b40,   eps2, Forcing20, Numerics, k2_uniform) / eps2;
sbth = channel_shift(Interior_Model, crust_matlab_idx, b_both, eps2, Forcing20, Numerics, k2_uniform) / eps2;
ssum = s20 + s40;

py20 = -1.5901e-5; py40 = +1.4528e-5; pyboth = -1.3738e-6;
logp('    (2,0) alone   : %+.4e   (py %+.4e)\n', s20,  py20);
logp('    (4,0) alone   : %+.4e   (py %+.4e)\n', s40,  py40);
logp('    both together : %+.4e   (py %+.4e)\n', sbth, pyboth);
logp('    (2,0)+(4,0) linear sum : %+.4e   (superposition check vs both: rel %.2e)\n', ...
    ssum, abs(ssum - sbth)/abs(sbth));
if abs(s20) > 0
    logp('    cancellation : net/|(2,0)| = %.4f  => %.1f%% cancellation\n', ...
        abs(sbth)/abs(s20), 100*(1 - abs(sbth)/abs(s20)));
end
logp('    SIGN CHECK    : (2,0) %s , (4,0) %s , both %s  vs Python signs (-,+,-)\n', ...
    sign_str(s20), sign_str(s40), sign_str(sbth));
signs_match = (sign(s20)==sign(py20)) && (sign(s40)==sign(py40)) && (sign(sbth)==sign(pyboth));
if signs_match
    logp('    ==> SIGNS REPRODUCE: opposite-sign cancellation confirmed in native MATLAB.\n');
else
    logp('    ==> *** SIGN MISMATCH *** the TASK-028 cancellation does NOT reproduce.\n');
    logp('        This is the decisive negative outcome; the proposal must be revisited.\n');
end

%% ========================================================================
%% PART 3: COUPLING COEFFICIENTS (native get_couplings Coup array)
%% ========================================================================
logp('\n---- PART 3: coupling coefficients, forcing (2,0) self-coupling ----------\n');
logp('    max|C| over the 26 channels of Coup(i_f,i_f,:,ireo), rheology (nb,0)\n');
py_maxC = containers.Map({1,2,3,4,5,6}, {0.0, 0.6389, 0.0, 0.8571, 0.0, 0.0});
part3 = struct([]);
for nb = 1:6
    variations = [nb, 0];
    Couplings = get_couplings(variations, Forcing20, Numerics);
    n_s = Couplings.n_s(:); m_s = Couplings.m_s(:);
    i_f = find(n_s == 2 & m_s == 0, 1);
    if isempty(i_f)
        logp('    (%d,0): forcing mode (2,0) absent from active modes -- skipped\n', nb);
        continue
    end
    coup_self = squeeze(Couplings.Coup(i_f, i_f, 1:26, 1));
    maxC = max(abs(coup_self(:)));
    logp('    rheology (%d,0): max|C| = %.4f   (py %.4f)\n', nb, maxC, py_maxC(nb));
    r.nb = nb; r.maxC = maxC; r.py = py_maxC(nb);
    if isempty(part3), part3 = r; else, part3(end+1) = r; end %#ok<SAGROW>
end

logp('==========================================================================\n\n');
fclose(logf);
fprintf('  saved log: %s\n', log_path);

%% SAVE VERIFICATION ARTIFACT
foc.matlab_version     = version;
foc.method             = Numerics.method;
foc.Nrbase             = Numerics.Nrbase;
foc.perturbation_order = Numerics.perturbation_order;
foc.k2_uniform         = k2_uniform;
foc.part1_exponents    = part1;
foc.part2.s20          = s20;
foc.part2.s40          = s40;
foc.part2.both         = sbth;
foc.part2.linear_sum   = ssum;
foc.part2.signs_match  = signs_match;
foc.part2.py           = struct('s20', py20, 's40', py40, 'both', pyboth);
foc.part3_coupling     = part3;
save(fullfile(out_dir, 'mars_first_order_channels.mat'), '-struct', 'foc');
fprintf('  saved artifact: %s\n\n', fullfile(out_dir, 'mars_first_order_channels.mat'));


%% ========================================================================
%% local functions
%% ========================================================================
function k = get_mode_k(Love, n, m)
    n_s = Love.n(:); m_s = Love.m(:); k_s = Love.k(:);
    idx = find(n_s == n & m_s == m, 1);
    if isempty(idx); error('mode (%d,%d) not found in Love spectrum', n, m); end
    k = real(k_s(idx));
end

function rows = isolate_zonal(mu_variable, ndeg)
    % keep only [n 0 amp] rows of degree ndeg (m==0) -- the zonal channel
    mask = (mu_variable(:,1) == ndeg) & (mu_variable(:,2) == 0);
    rows = mu_variable(mask, :);
end

function shift = channel_shift(Interior_Model, crust_idx, base_rows, eps, ...
                               Forcing, Numerics, k2_uniform)
    % scale the isolated channel by eps, coupled solve, return k2 shift
    IM = Interior_Model;
    scaled = base_rows;
    scaled(:,3) = eps * base_rows(:,3);
    IM(crust_idx).mu_variable = scaled;
    IM = get_rheology(IM, Numerics, Forcing);
    [Love, ~] = get_Love(IM, Forcing, Numerics);
    k2 = get_mode_k(Love, Forcing.n, Forcing.m);
    shift = k2 - k2_uniform;
end

function s = sign_str(x)
    if x > 0; s = '(+)'; elseif x < 0; s = '(-)'; else; s = '(0)'; end
end

function both_print(logf, varargin)
    fprintf(varargin{:});
    if logf > 0; fprintf(logf, varargin{:}); end
end

%% minimal .npz reader (same as scripts/mars_lateral_cross_check.m) -----------
function s = read_mars_mu_variable_npz(npz_path)
    if ~isfile(npz_path); error('npz not found: %s', npz_path); end
    tmp = tempname; mkdir(tmp);
    cleanup = onCleanup(@() rmdir(tmp, 's')); %#ok<NASGU>
    unzip(npz_path, tmp);
    s.layer_idx = read_npy(fullfile(tmp, 'layer_idx.npy'));
    s.n         = read_npy(fullfile(tmp, 'n.npy'));
    s.m         = read_npy(fullfile(tmp, 'm.npy'));
    s.amp_real  = read_npy(fullfile(tmp, 'amp_real.npy'));
    s.amp_imag  = read_npy(fullfile(tmp, 'amp_imag.npy'));
end

function arr = read_npy(fname)
    fid = fopen(fname, 'r');
    if fid < 0; error('cannot open %s', fname); end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    magic = fread(fid, 6, '*char')';
    if ~strcmp(magic, sprintf('\x93NUMPY')); error('%s is not a .npy file', fname); end
    fread(fid, 2, 'uint8');
    header_len = fread(fid, 1, 'uint16');
    header = fread(fid, header_len, '*char')'; %#ok<NASGU>
    descr = regexp(header, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
    descr = descr{1};
    switch descr
        case {'<i8', '|i8', 'int64'}
            prec = 'int64=>double';
        case {'<f8', '|f8', 'float64'}
            prec = 'double';
        otherwise
            error('unsupported .npy dtype %s in %s', descr, fname);
    end
    arr = fread(fid, Inf, prec);
    arr = arr(:);
end
