%% MARS LATERAL (COUPLED) CROSS-CHECK (TASK-014 part 2)
% Runs the native LOV3D *coupled* (mode-coupling) solver on the exact 4-layer
% Mars reference model plus the crust-layer lateral rigidity field committed in
% data/mars/mars_mu_variable_lateral.npz, and prints the resulting Love-number
% spectrum for direct comparison against the Python port (pylov3d/mars_lateral.py
% / docs/MARS_MODEL.md section "Lateral variations").
%
% This is the coupled analogue of scripts/mars_1d_cross_check.m (part 1). It
% gives the Mars *lateral* model the same native-MATLAB anchor the 1D model
% already has. The single headline number to reproduce is the forcing-mode
% (2,0) k2 lateral shift:
%   Python:  k2 = 0.169000  ->  0.1690552   (shift +5.517e-5 at lmax=4)
%
% Model arrays are copied VERBATIM from build_mars_model() (see part-1 script);
% the lateral mu_variable entries are read VERBATIM from the committed npz so
% the cross-check exercises the coupled *solver*, not the Airy/topography fit.
%
% npz -> MATLAB mapping (see the npz readme):
%   - complex amp = amp_real + 1i*amp_imag, SAME complex-SH convention as the
%     already-validated Weber-Moon harness (both +m and -m rows are present and
%     already form a real field, so we hand them to mu_variable directly and do
%     NOT use the mu_variable_p2p percent path).
%   - layer_idx is 0-based (3 = crust). MATLAB Interior_Model is 1-based, so the
%     crust is Interior_Model(4): MATLAB layer index = python index + 1.
%   - purely elastic: eta0 OMITTED on every layer (eta0=NaN poisons the MATLAB
%     solve; part-1 gotcha).
%
% Run headless from the repo root:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch \
%       "run('scripts/mars_lateral_cross_check.m')"

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

%% INTERIOR MODEL (verbatim from pylov3d.mars.build_mars_model; see part-1 script)
% Interior_Model goes from core (1) to surface (n). Purely elastic (eta0 omitted).
Interior_Model(1).R0     = 1830.0;                % CMB radius [km]
Interior_Model(1).rho0   = 6128.075995512780;     % fitted core density [kg/m^3]
Interior_Model(1).rho0_2 = 6128.075995512780;

% Lower mantle
Interior_Model(2).R0   = 2340.0;
Interior_Model(2).rho0 = 4136.503827041973;
Interior_Model(2).Ks0  = 160e9;
Interior_Model(2).mu0  = 96482476610.2174;        % 100e9 * MARS_MU_SCALE

% Upper mantle
Interior_Model(3).R0   = 3339.5;
Interior_Model(3).rho0 = 3400.0;
Interior_Model(3).Ks0  = 115e9;
Interior_Model(3).mu0  = 67537733627.15218;       % 70e9 * MARS_MU_SCALE

% Crust (surface layer; lateral rigidity variation lives here)
Interior_Model(4).R0   = 3389.5;
Interior_Model(4).rho0 = 2900.0;
Interior_Model(4).Ks0  = 70e9;
Interior_Model(4).mu0  = 30e9;

% Core density contrast (matches the Python default: rho_core - rho_lm).
Interior_Model(1).Delta_rho0 = Interior_Model(1).rho0_2 - Interior_Model(2).rho0;

%% LATERAL RIGIDITY FIELD (verbatim from data/mars/mars_mu_variable_lateral.npz)
% amp = amp_real + 1i*amp_imag; layer_idx (0-based) 3 = crust = Interior_Model(4).
% Read the npz without needing a Python bridge: unzip -> parse the .npy arrays.
npz_path = fullfile(repo_root, 'data', 'mars', 'mars_mu_variable_lateral.npz');
lv = read_mars_mu_variable_npz(npz_path);   % local function, see bottom of file

% All committed entries are in the crust layer (layer_idx == 3 == crust).
assert(all(lv.layer_idx == 3), 'npz has non-crust layer_idx entries');
crust_matlab_idx = 4;   % python 3 + 1

% Build the complex-SH mu_variable matrix [n, m, amp] for the crust layer.
Interior_Model(crust_matlab_idx).mu_variable = ...
    [double(lv.n(:)), double(lv.m(:)), (lv.amp_real(:) + 1i*lv.amp_imag(:))];

%% NUMERICS (matches mars_lateral_love_spectrum: lmax=4, perturbation_order=2)
% Notebook-style coupled setup (Test_Moon_MultiLayered_Lateral_Variations.mlx):
% variable grid, load_couplings=2 (generic cached file), Nrbase per Python.
Numerics.Nlayers = 4;
Numerics.method = 'variable';
Numerics.Nrbase = 30;                    % Python production default for the spectrum
Numerics.perturbation_order = 2;         % matches mars_lateral_love_spectrum default
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

%% FORCING (n=2, m=0 tidal; elastic model => frequency-independent, Td arbitrary)
Forcing(1).Td = 44387.62;   % s (MARS_FORCING_TD)
Forcing(1).n = 2;
Forcing(1).m = 0;
Forcing(1).F = 1;

%% UNIFORM (no-lateral) reference solve, for the forcing-mode k2 shift
Interior_Model_Uni = Interior_Model;
Interior_Model_Uni(crust_matlab_idx).mu_variable = [];   % strip lateral field
Interior_Model_Uni = get_rheology(Interior_Model_Uni, Numerics, Forcing);
[Love_Uni, ~] = get_Love(Interior_Model_Uni, Forcing, Numerics, 'verbose');
k2_uniform = real(Love_Uni.k(1));

%% COUPLED (lateral) solve
Interior_Model = get_rheology(Interior_Model, Numerics, Forcing);
[Love_Spectra, ~] = get_Love(Interior_Model, Forcing, Numerics, 'verbose');

%% REPORT
n_s = Love_Spectra.n(:);
m_s = Love_Spectra.m(:);
k_s = Love_Spectra.k(:);

iforcing = find(n_s == Forcing.n & m_s == Forcing.m, 1);
k2_forcing = real(k_s(iforcing));
k2_shift = k2_forcing - k2_uniform;

fprintf('\n============ MARS LATERAL COUPLED CROSS-CHECK (MATLAB LOV3D) ============\n');
fprintf('  N coupled solution modes : %d\n', length(n_s));
fprintf('  k2 uniform (no lateral)  : %.12f\n', k2_uniform);
fprintf('  k2 forcing (2,0) lateral : %.12f\n', k2_forcing);
fprintf('  k2 lateral shift         : %.6e\n', k2_shift);
fprintf('------------------------------------------------------------------------\n');
fprintf('  Python (pylov3d) reference: k2_uniform=0.169000  k2=0.1690552  shift=+5.517e-5\n');
fprintf('  rel err  k2_uniform=%.3e   shift=%.3e\n', ...
    abs(k2_uniform - 0.169) / 0.169, ...
    abs(k2_shift - 5.517e-5) / 5.517e-5);
fprintf('------------------------------------------------------------------------\n');
fprintf('  Top lateral-response modes by |k| (forcing mode shown as deviation):\n');
kdev = k_s;
kdev(iforcing) = k_s(iforcing) - k2_uniform;
[~, ord] = sort(abs(kdev), 'descend');
nshow = min(12, length(ord));
for j = 1:nshow
    i = ord(j);
    tag = '';
    if i == iforcing, tag = '  <- forcing (deviation)'; end
    fprintf('    (n=%2d, m=%3d)  k = %+.6e %+.6ei%s\n', ...
        n_s(i), m_s(i), real(kdev(i)), imag(kdev(i)), tag);
end
fprintf('========================================================================\n\n');


%% ------------------------------------------------------------------------
%% local function: minimal .npz reader for the committed lateral field
%% ------------------------------------------------------------------------
function s = read_mars_mu_variable_npz(npz_path)
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
