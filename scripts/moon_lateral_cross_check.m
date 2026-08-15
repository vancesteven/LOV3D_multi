%% MOON LATERAL (COUPLED, OCEAN) CROSS-CHECK (TASK-035, re-anchored TASK-038)
% Runs the native LOV3D *coupled* (mode-coupling) solver on the as-built
% ten-layer Weber et al. (2011) Moon model -- INCLUDING ITS FLUID OUTER CORE,
% the part of this cross-check with no Mars precedent -- plus the crust-layer
% lateral rigidity field committed in data/moon/moon_mu_variable_lateral.npz,
% and prints the resulting Love-number spectrum for direct comparison against
% the Python port (pylov3d/moon_lateral.py / docs/MOON_MODEL.md, "Coupled
% spectrum"). This is the Moon analogue of scripts/mars_lateral_cross_check.m
% (TASK-014 pt 2): every load-bearing Mars result already has a native-MATLAB
% anchor (1D, lateral spectrum, diagonal k2m, first-order channels); this
% closes the same gap for the Moon lateral stage.
%
% TASK-038 (2026-08-14 PI decision): the field this script targets now
% RETAINS the degree-1 nearside-farside dichotomy (pylov3d.moon_lateral
% default include_degree1=True), so the committed npz carries 23 complex
% amplitudes (was 20; the three new entries are (1,0), (1,1), (1,-1)). The
% reference constants below are read at FULL precision (17 significant
% digits, repr-level) straight from the committed spectrum npz
% (docs/figures/proposal/moon_lateral_spectrum.npz) -- the prior 6-significant-
% figure constants made the anchor look ~60x looser than it is (true
% forcing-mode agreement is 9.6e-14 relative, not the ~1e-12 the rounded
% constants implied). The superseded degree-1-removed anchor (this same
% script, pinned against the pre-dichotomy field) is preserved in git history
% at 47b5377 (log/mat artifacts at dd86cdb); this file overwrites those
% artifacts in place per TASK-038.
%
% Python reference values (docs/figures/proposal/moon_lateral_spectrum.npz,
% full precision; lmax=4, method='variable', Nrbase=30, perturbation_order=2,
% unit (2,0) monthly forcing):
%   uniform Weber k2  = 0.02315914222851756
%   lateral k20       = 0.023161283468225102
%   Delta k20         = +2.1412397075426526e-06
%   N modes           = 115
%   |k(3,+/-1)|       = 6.372785331949207e-06   (new dominant off-forcing pair)
%   |k(2,+/-2)|       = 3.0301228895909613e-06
%   |k(2,+/-1)|       = 2.6352536444942246e-06
%   |k(3,+/-3)|       = 2.0205378604363597e-06
%
% Model arrays are copied VERBATIM from pylov3d.moon.build_moon_model()'s
% output (pylov3d/moon.py LAYER_RADII_KM / LAYER_RHO / LAYER_MU / LAYER_KS /
% LAYER_OCEAN, printed directly from the module, not re-derived); the lateral
% mu_variable entries are read VERBATIM from the committed npz
% (scripts/export_moon_mu_variable.py) so the cross-check exercises the
% coupled *ocean* solver, not the Airy/topography fit.
%
% npz -> MATLAB mapping (see the npz readme, and
% scripts/export_moon_mu_variable.py's embedded README string):
%   - complex amp = amp_real + 1i*amp_imag, SAME complex-SH convention as the
%     Mars lateral/DWAK exports (both +m and -m rows are present and already
%     form a real field, so we hand them to mu_variable directly and do NOT
%     use the mu_variable_p2p percent path).
%   - layer_idx is 0-based Python (9 = crust). MATLAB Interior_Model is
%     1-based: MATLAB layer index = python index + 1, so the Moon crust is
%     Interior_Model(10) and the sub-crust mantle shell is Interior_Model(9).
%   - purely elastic: eta0 = [] (EMPTY) on every layer; eta0 = NaN would wrongly
%     enter the viscoelastic branch and poison the solve (TASK-014 pt-1
%     gotcha).
%
% Run headless from the repo root:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch \
%       "run('scripts/moon_lateral_cross_check.m')"

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

% couplings are cached to disk; make sure the directory exists
coupling_dir = fullfile(repo_root, 'data', 'couplings');
if ~isfolder(coupling_dir)
    mkdir(coupling_dir);
end

%% INTERIOR MODEL (verbatim from pylov3d.moon.build_moon_model(); see
% pylov3d/moon.py LAYER_RADII_KM / LAYER_RHO / LAYER_MU / LAYER_KS /
% LAYER_OCEAN / LAYER_NAMES for the exact source arrays).
% Interior_Model goes from core (1) to surface (10). This is the Weber et al.
% (2011) 10-layer profile with an artificial 50 km/8000 kg/m^3 numerically-
% inert core prepended, the solid inner core rigidified (mu, Ks x1000), and
% the physical fluid outer core (Vs=0) tagged ocean=1 at layer 3.
% Purely elastic: eta0 = [] on every layer (see header note).

% Layer 1: artificial numerically-inert core (LAYER_NAMES[0]).
Interior_Model(1).R0     = 50.0;             % km
Interior_Model(1).rho0   = 8000.0;           % kg/m^3
Interior_Model(1).rho0_2 = 8000.0;           % matches rho0 (no icy-moon 2nd density)
Interior_Model(1).eta0   = [];
Interior_Model(1).ocean  = 0;

% Layer 2: solid inner core, rigidified (LAYER_NAMES[1]).
Interior_Model(2).R0   = 240.0;
Interior_Model(2).rho0 = 8000.0;
Interior_Model(2).mu0  = 4.2320000000000000e+13;   % rho*Vs^2, x1000 rigidified
Interior_Model(2).Ks0  = 9.1493333333333344e+13;   % rho*(Vp^2-4/3 Vs^2), x1000
Interior_Model(2).eta0 = [];
Interior_Model(2).ocean = 0;

% Layer 3: fluid outer core (LAYER_NAMES[2]) -- ocean=1. THE PART OF THIS
% CROSS-CHECK WITH NO MARS PRECEDENT: Mars's liquid core sits at the model
% centre (no mu0/Ks0 fields at all); the Weber Moon's fluid core is a
% *subsurface* ocean layer that exercises assemble_bc_ocean/get_solution's
% 24x24 ocean boundary-condition path.
Interior_Model(3).R0     = 330.0;
Interior_Model(3).rho0   = 5100.0;
Interior_Model(3).mu0    = 0.0;                    % Vs = 0 (fluid)
Interior_Model(3).Ks0    = 8.5731000000000000e+10;
Interior_Model(3).ocean  = 1;
Interior_Model(3).eta0   = [];

% Layer 4: top of the partial-melt zone (LAYER_NAMES[3]).
Interior_Model(4).R0   = 480.0;
Interior_Model(4).rho0 = 3400.0;
Interior_Model(4).mu0  = 3.4816000000000000e+10;
Interior_Model(4).Ks0  = 1.4482866666666669e+11;
Interior_Model(4).eta0 = [];
Interior_Model(4).ocean = 0;

% Layer 5: mantle_2 (LAYER_NAMES[4]).
Interior_Model(5).R0   = 999.1;
Interior_Model(5).rho0 = 3400.0;
Interior_Model(5).mu0  = 6.8850000000000000e+10;
Interior_Model(5).Ks0  = 1.5385000000000000e+11;
Interior_Model(5).eta0 = [];
Interior_Model(5).ocean = 0;

% Layer 6: mantle_3 (LAYER_NAMES[5]).
Interior_Model(6).R0   = 1249.1;
Interior_Model(6).rho0 = 3400.0;
Interior_Model(6).mu0  = 6.5824000000000000e+10;
Interior_Model(6).Ks0  = 1.0861866666666667e+11;
Interior_Model(6).eta0 = [];
Interior_Model(6).ocean = 0;

% Layer 7: mantle_4 (LAYER_NAMES[6]).
Interior_Model(7).R0   = 1499.1;
Interior_Model(7).rho0 = 3400.0;
Interior_Model(7).mu0  = 6.4929460000000000e+10;
Interior_Model(7).Ks0  = 1.2028338666666669e+11;
Interior_Model(7).eta0 = [];
Interior_Model(7).ocean = 0;

% Layer 8: mantle_5 (LAYER_NAMES[7]).
Interior_Model(8).R0   = 1675.1;
Interior_Model(8).rho0 = 3220.0;
Interior_Model(8).mu0  = 6.3477792000000000e+10;
Interior_Model(8).Ks0  = 1.0627674400000000e+11;
Interior_Model(8).eta0 = [];
Interior_Model(8).ocean = 0;

% Layer 9: mantle_6 (LAYER_NAMES[8]) -- MANTLE_LAYER_INDEX (python 8), the
% sub-crust layer referenced in the npz README.
Interior_Model(9).R0   = 1703.1;
Interior_Model(9).rho0 = 3220.0;
Interior_Model(9).mu0  = 6.3477792000000000e+10;
Interior_Model(9).Ks0  = 1.0627674400000000e+11;
Interior_Model(9).eta0 = [];
Interior_Model(9).ocean = 0;

% Layer 10: crust (LAYER_NAMES[9]) -- CRUST_LAYER_INDEX (python 9); lateral
% rigidity variation lives here.
Interior_Model(10).R0   = 1737.1;
Interior_Model(10).rho0 = 2800.0;
Interior_Model(10).mu0  = 2.8672000000000000e+10;
Interior_Model(10).Ks0  = 4.6470666666666672e+10;
Interior_Model(10).eta0 = [];
Interior_Model(10).ocean = 0;

% Core density contrast (matches the Python default: rho0_2 - rho0(layer 2);
% here rho0_2 == rho0(layer 1) == rho0(layer 2) == 8000, so this is 0 --
% same value pylov3d.types.make_interior_model's auto-fill computes).
Interior_Model(1).Delta_rho0 = Interior_Model(1).rho0_2 - Interior_Model(2).rho0;

%% LATERAL RIGIDITY FIELD (verbatim from data/moon/moon_mu_variable_lateral.npz)
% amp = amp_real + 1i*amp_imag; layer_idx (0-based) 9 = crust = Interior_Model(10).
% Read the npz without needing a Python bridge: unzip -> parse the .npy arrays.
npz_path = fullfile(repo_root, 'data', 'moon', 'moon_mu_variable_lateral.npz');
lv = read_moon_mu_variable_npz(npz_path);   % local function, see bottom of file

% All committed entries are in the crust layer (layer_idx == 9 == crust).
assert(all(lv.layer_idx == 9), 'npz has non-crust layer_idx entries');
crust_matlab_idx = 10;   % python 9 + 1

% Build the complex-SH mu_variable matrix [n, m, amp] for the crust layer.
Interior_Model(crust_matlab_idx).mu_variable = ...
    [double(lv.n(:)), double(lv.m(:)), (lv.amp_real(:) + 1i*lv.amp_imag(:))];

%% NUMERICS (matches pylov3d.moon_lateral.moon_lateral_love_spectrum defaults:
% lmax=4, perturbation_order=2, Nrbase=30, method='variable')
Numerics.Nlayers = 10;
Numerics.method = 'variable';
Numerics.Nrbase = 30;                    % Python production default for the spectrum
Numerics.perturbation_order = 2;         % matches moon_lateral_love_spectrum default
Numerics.solution_cutoff = 12;
Numerics.load_couplings = 2;             % search/generate a generic coupling file
Numerics.Nenergy = 12;
Numerics.rheology_cutoff = 2;
Numerics.parallel_sol = 0;
Numerics.parallel_gen = 0;
Numerics.coupling_file_location = [coupling_dir filesep];

[Numerics, Interior_Model] = set_boundary_indices(Numerics, Interior_Model, 'verbose');

if ~(Numerics.Nlayers == length(Interior_Model))
    error('Numerics.Nlayers must equal length(Interior_Model)');
end

%% FORCING (n=2, m=0 monthly tide; elastic model => frequency-independent,
% Td is nonetheless set to the Moon's monthly period, matching
% pylov3d.moon.MOON_FORCING_TD, for forward compatibility with an eventual
% anelastic extension.)
FORCING_TD = 2360591.6;   % s (MOON_FORCING_TD)
Forcing(1).Td = FORCING_TD;
Forcing(1).n  = 2;
Forcing(1).m  = 0;
Forcing(1).F  = 1;

% --- Python reference values (docs/figures/proposal/moon_lateral_spectrum.npz,
%     full precision -- TASK-038 re-anchor, dichotomy-retaining field) -------
py_k2_uniform  = 0.02315914222851756;
py_k20_lateral = 0.023161283468225102;
py_delta_k20   = 2.1412397075426526e-06;
py_N_modes     = 115;   % unchanged from TASK-035's total -- but this script
                         % does not break N down by perturbation order (no
                         % hardcoded 1/42/73-style split lives here). B: don't
                         % assume the *composition* of the 115 modes carried
                         % over unmodified -- the field gained three degree-1
                         % amplitudes that couple into the spectrum even
                         % though the total count landed back on 115; if a
                         % per-order breakdown is needed, derive it from this
                         % run's Love_Spectra output, not from TASK-035's.
py_k31_abs     = 6.372785331949207e-06;   % |k(3,+/-1)| -- new dominant off-forcing pair
py_k22_abs     = 3.0301228895909613e-06;  % |k(2,+/-2)|
py_k21_abs     = 2.6352536444942246e-06;  % |k(2,+/-1)|
py_k33_abs     = 2.0205378604363597e-06;  % |k(3,+/-3)|

out_dir = fullfile(repo_root, 'data', 'tests', 'moon');
if ~isfolder(out_dir)
    mkdir(out_dir);
end

% open a combined log (TASK-035 deliverable): tee fprintf to console+file.
log_path = fullfile(out_dir, 'moon_lateral_cross_check.log');
logf = fopen(log_path, 'w');
logp = @(varargin) both_print(logf, varargin{:});

logp('\n============ MOON LATERAL COUPLED (OCEAN) CROSS-CHECK (MATLAB LOV3D) ============\n');
logp('  TASK-035: native-MATLAB anchor for the Moon lateral spectrum (pylov3d.moon_lateral)\n');
logp('  MATLAB version: %s\n', version);
logp('  Nrbase=%d  perturbation_order=%d  method=%s  Nlayers=%d\n', ...
    Numerics.Nrbase, Numerics.perturbation_order, Numerics.method, Numerics.Nlayers);

%% --- uniform (no-lateral) reference solve ---------------------------------
IM_uni = Interior_Model;
IM_uni(crust_matlab_idx).mu_variable = [];   % strip lateral field
IM_uni = get_rheology(IM_uni, Numerics, Forcing);
[Love_Uni, ~] = get_Love(IM_uni, Forcing, Numerics, 'verbose');
k2_uniform = real(Love_Uni.k(1));

%% --- coupled (lateral) solve -----------------------------------------------
IM_lat = get_rheology(Interior_Model, Numerics, Forcing);
[Love_Spectra, ~] = get_Love(IM_lat, Forcing, Numerics, 'verbose');

n_s = Love_Spectra.n(:);
m_s = Love_Spectra.m(:);
k_s = Love_Spectra.k(:);
N_modes = length(n_s);
iforcing = find(n_s == Forcing.n & m_s == Forcing.m, 1);
k2_forcing = real(k_s(iforcing));
k2_shift   = k2_forcing - k2_uniform;

logp('\n------------------------------------------------------------------------\n');
logp('  FORCING (2,0)\n');
logp('    N coupled solution modes : %d   (Python: %d)\n', N_modes, py_N_modes);
logp('    k2 uniform (no lateral)  : %.12f   Python: %.12f   rel err = %.3e\n', ...
    k2_uniform, py_k2_uniform, abs(k2_uniform - py_k2_uniform) / abs(py_k2_uniform));
logp('    k2 forcing (2,0) lateral : %.12f   Python: %.12f   rel err = %.3e\n', ...
    k2_forcing, py_k20_lateral, abs(k2_forcing - py_k20_lateral) / abs(py_k20_lateral));
logp('    k2 lateral shift         : %+.6e   Python: %+.6e   rel err = %.3e\n', ...
    k2_shift, py_delta_k20, abs(k2_shift - py_delta_k20) / abs(py_delta_k20));

%% --- per-mode comparison against the four named off-forcing pairs ---------
% For each named (n,|m|) pair, report the MATLAB |k| for both signed m and
% compare against the Python |k| given in docs/figures/proposal/moon_lateral_spectrum.npz.
% (3,+/-1) is listed first: it is the new dominant off-forcing pair once the
% degree-1 dichotomy is retained (TASK-038), ahead of the (2,+/-2) pair that
% was dominant under the superseded degree-1-removed field.
named_pairs = [3, 1, py_k31_abs; 2, 2, py_k22_abs; 2, 1, py_k21_abs; 3, 3, py_k33_abs];
logp('\n  Named off-forcing pairs (|k|, both signs of m):\n');
for irow = 1:size(named_pairs, 1)
    n_named = named_pairs(irow, 1);
    m_named = named_pairs(irow, 2);
    py_abs  = named_pairs(irow, 3);
    for sgn = [1, -1]
        m_this = sgn * m_named;
        idx = find(n_s == n_named & m_s == m_this, 1);
        if isempty(idx)
            logp('    (n=%d, m=%+d)  NOT FOUND in coupled spectrum\n', n_named, m_this);
            continue
        end
        k_abs = abs(k_s(idx));
        rel_err = abs(k_abs - py_abs) / py_abs;
        logp('    (n=%d, m=%+d)  |k| = %.6e   Python |k| = %.6e   rel err = %.3e\n', ...
            n_named, m_this, k_abs, py_abs, rel_err);
    end
end

logp('\n  Top 12 lateral-response modes by |k| (forcing mode shown as deviation):\n');
kdev = k_s;
kdev(iforcing) = k_s(iforcing) - k2_uniform;
[~, ord] = sort(abs(kdev), 'descend');
nshow = min(12, length(ord));
for j = 1:nshow
    i = ord(j);
    tag = '';
    if i == iforcing, tag = '  <- forcing (deviation)'; end
    logp('      (n=%2d, m=%3d)  k = %+.6e %+.6ei%s\n', ...
        n_s(i), m_s(i), real(kdev(i)), imag(kdev(i)), tag);
end
logp('========================================================================\n\n');
fclose(logf);
fprintf('  saved log: %s\n', log_path);

%% SAVE VERIFICATION ARTIFACT (TASK-035)
% Persist the computed coupled spectrum to a small .mat so a MATLAB-less
% reader (or a future regression check) can verify the (2,0) forcing Love
% number and the mode list without re-running MATLAB.
moon_lat.N_modes            = N_modes;
moon_lat.k2_uniform          = k2_uniform;
moon_lat.k2_forcing          = k2_forcing;
moon_lat.k2_shift            = k2_shift;
moon_lat.n                   = n_s;
moon_lat.m                   = m_s;
moon_lat.k                   = k_s;
moon_lat.forcing_n           = 2;
moon_lat.forcing_m           = 0;
moon_lat.py_k2_uniform       = py_k2_uniform;
moon_lat.py_k20_lateral      = py_k20_lateral;
moon_lat.py_delta_k20        = py_delta_k20;
moon_lat.py_N_modes          = py_N_modes;
moon_lat.py_k31_abs          = py_k31_abs;
moon_lat.py_k22_abs          = py_k22_abs;
moon_lat.py_k21_abs          = py_k21_abs;
moon_lat.py_k33_abs          = py_k33_abs;
moon_lat.Nrbase              = Numerics.Nrbase;
moon_lat.perturbation_order  = Numerics.perturbation_order;
moon_lat.method              = Numerics.method;
moon_lat.matlab_version      = version;
save(fullfile(out_dir, 'moon_lateral_cross_check.mat'), '-struct', 'moon_lat');
fprintf('  saved artifact: %s\n\n', fullfile(out_dir, 'moon_lateral_cross_check.mat'));


%% ------------------------------------------------------------------------
%% local function: tee a printf to both the console and the open log file
%% ------------------------------------------------------------------------
function both_print(logf, varargin)
    fprintf(varargin{:});
    if logf > 0
        fprintf(logf, varargin{:});
    end
end

%% ------------------------------------------------------------------------
%% local function: minimal .npz reader for the committed lateral field
%% ------------------------------------------------------------------------
function s = read_moon_mu_variable_npz(npz_path)
    % Extracts the arrays this script needs (layer_idx, n, m, amp_real,
    % amp_imag) from a NumPy .npz (a plain zip of .npy files). Only the
    % 1-D numeric arrays used here are parsed; scalar/string provenance
    % fields in the npz are ignored.
    if ~isfile(npz_path)
        error('npz not found: %s', npz_path);
    end
    tmp = tempname;
    mkdir(tmp);
    cleanup = onCleanup(@() rmdir(tmp, 's'));
    unzip(npz_path, tmp);
    s.layer_idx = read_npy(fullfile(tmp, 'layer_idx.npy'));
    s.n         = read_npy(fullfile(tmp, 'n.npy'));
    s.m         = read_npy(fullfile(tmp, 'm.npy'));
    s.amp_real  = read_npy(fullfile(tmp, 'amp_real.npy'));
    s.amp_imag  = read_npy(fullfile(tmp, 'amp_imag.npy'));
end

function arr = read_npy(fname)
    % Minimal reader for 1-D little-endian NumPy .npy v1.0 arrays
    % (dtype '<i8' or '<f8', C-order). Sufficient for the committed field.
    fid = fopen(fname, 'r');
    if fid < 0, error('cannot open %s', fname); end
    cleanup = onCleanup(@() fclose(fid));
    magic = fread(fid, 6, '*char')';
    if ~strcmp(magic, sprintf('\x93NUMPY'))
        error('%s is not a .npy file', fname);
    end
    fread(fid, 2, 'uint8');                 % version major/minor
    header_len = fread(fid, 1, 'uint16');   % v1.0: 2-byte little-endian
    header = fread(fid, header_len, '*char')';

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
