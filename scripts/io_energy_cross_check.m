%% IO VISCOELASTIC LATERAL ENERGY CROSS-CHECK (TASK-046, Gate C)
% Native-MATLAB anchor for pylov3d's Io viscoelastic + lateral-rheology +
% energy validation (TASK-046 Gates A/B, scripts/io_energy_consistency.py).
% Reproduces the upstream tests/Consistency_test_Energy.m Io four-layer
% configuration at a SINGLE tractable radial resolution (Nrbase=50, the
% spec's chosen rung -- not the full [5,10,20,50,100,200,500,1000] ladder
% Consistency_test_Energy.m itself sweeps), solves the three (2,0)/(2,-2)/
% (2,2) eccentricity-tide forcings for both the uniform and laterally
% mu+eta-varying asthenosphere, and writes a compact .mat reference with
% the Gate C quantities the spec lists: forcing-mode complex k for all
% three forcings; selected first-/second-order coupled complex k
% coefficients; direct integrated energy; Love-number-derived energy;
% radial resolution and all model/forcing parameters.
%
% This machine has no MATLAB; TASK-046 is split A (pylov3d Gates A/B, this
% .m script, and the shared-field export) / B (actually running this
% script). DO NOT RUN from the Python side -- prepared only.
%
% Header comment (per spec): for machine B, TASK-046 Gate C.
%
% Run headless from the repo root, once B has this checked out:
%   /Applications/MATLAB_R2025x.app/bin/matlab -batch \
%       "run('scripts/io_energy_cross_check.m')"
%
% Model / forcing parameters are copied VERBATIM from the values pylov3d
% actually used (pylov3d/io_lateral.py IO_R0_KM / IO_RHO0 / IO_MU0 /
% IO_KS0 / IO_ETA0 / IO_FORCING_COMPONENTS / IO_OMEGA0), which are
% themselves copied verbatim from tests/Consistency_test_Energy.m lines
% 143-209 (including Ks0 = 200e12 Pa, NOT the pylov3d/tests/test_energy.py
% fixture's 200e16 -- see scripts/io_energy_consistency.py's module
% docstring for that audit).
%
% =========================================================================
% BASIS-CONVENTION CAVEAT -- READ BEFORE TRUSTING A NUMERIC DISAGREEMENT
% =========================================================================
% The mu_variable / eta_variable matrices this script builds below are
% read from data/io/io_mu_eta_variable.npz, which pylov3d's driver
% (scripts/io_energy_consistency.py) populates from
% pylov3d.rheology._sh_analysis's basis (scipy.special.sph_harm_y:
% orthonormal, Condon-Shortley phase) -- because that is the basis
% pylov3d.rheology.process_lateral_variations's VISCOELASTIC branch
% actually consumes internally (via _sh_synthesis), which is verified
% numerically to be DIFFERENT from the real-4pi-normalized
% (fully_normalized_legendre) basis the Mars/Moon ELASTIC-crust
% mu_variable exports use (see pylov3d/io_lateral.py's module docstring
% and pylov3d/tests/test_io_lateral.py::TestBasisConvention).
%
% Whether MATLAB's own get_rheology.m interprets an Interior_Model(...).
% mu_variable / eta_variable matrix for a VISCOELASTIC layer in that SAME
% sph_harm_y basis has NOT been checked (checking would require reading
% the MATLAB get_rheology.m source, out of scope for the Python-side half
% of this task). Notably, Consistency_test_Energy.m itself does NOT feed
% the asthenosphere a mu_variable/eta_variable matrix at all -- it feeds
% RAW lat/lon/z GRID structs (Interior_Model(3).mu_latlon /
% Interior_Model(3).eta_latlon, lines 116-125), which get_rheology.m
% presumably re-analyzes into SH coefficients using its own (likely
% real-4pi, no-Condon-Shortley, SPH_Tools/Legendre.m-based) convention --
% i.e. the ELASTIC-branch (mars_lateral) convention, not the sph_harm_y
% one this export uses.
%
% If B has time: the more defensible native check is to REBUILD the
% lat/lon/z grid directly from pylov3d.io_lateral's closed-form pattern
% formulas (io_heating_grid / _io_z_pattern / _io_dmu_deta -- all short,
% closed-form; see that module) and feed it through Interior_Model(3).
% mu_latlon / eta_latlon exactly as Consistency_test_Energy.m does, rather
% than trusting this script's mu_variable/eta_variable path. This script
% still runs the mu_variable/eta_variable path (per the TASK-046 spec,
% which explicitly asks for the shared-coefficient-export route) so B has
% *something* runnable immediately; if the resulting k/energy values
% disagree from the Python side by an implausible amount, this basis
% mismatch -- not a solver bug -- is the first thing to rule out.
%
% npz -> MATLAB mapping (same npy-parser pattern as
% scripts/moon_lateral_cross_check.m's read_moon_mu_variable_npz):
%   mu_n, mu_m, mu_amp_real, mu_amp_imag : asthenosphere mu_variable
%     entries, complex amp = amp_real + 1i*amp_imag.
%   eta_n, eta_m, eta_amp_real, eta_amp_imag : same for eta_variable.
%   (Provenance scalars in the npz -- R0_km, rho0, mu0, Ks0, eta0,
%   forcing_*, omega0, Td, lmax_sh -- are NOT re-read here; this script
%   hardcodes the identical values directly, matching the moon/mars
%   cross-check scripts' "copied verbatim" convention, so a stray npz
%   provenance-array read failure cannot silently corrupt the model.)

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
if ~isfolder(coupling_dir)
    mkdir(coupling_dir);
end

%% INTERIOR MODEL (verbatim from pylov3d.io_lateral.IO_R0_KM / IO_RHO0 /
% IO_MU0 / IO_KS0 / IO_ETA0, i.e. tests/Consistency_test_Energy.m lines
% 143-168 with Ks0 = 200e12 Pa).
Interior_Model(1).R0     = 965.0;
Interior_Model(1).rho0   = 5150.0;
Interior_Model(1).rho0_2 = 5150.0;
Interior_Model(1).eta0   = [];

Interior_Model(2).R0   = 1591.6;
Interior_Model(2).rho0 = 3244.0;
Interior_Model(2).Ks0  = 200e12;
Interior_Model(2).mu0  = 6e10;
Interior_Model(2).eta0 = 1e20;

Interior_Model(3).R0   = 1791.6;
Interior_Model(3).rho0 = 3244.0;
Interior_Model(3).Ks0  = 200e12;
Interior_Model(3).mu0  = 7.8e5;
Interior_Model(3).eta0 = 1e11;

Interior_Model(4).R0   = 1821.6;
Interior_Model(4).rho0 = 3244.0;
Interior_Model(4).Ks0  = 200e12;
Interior_Model(4).mu0  = 6.5e10;
Interior_Model(4).eta0 = 1e23;

Interior_Model(1).Delta_rho0 = Interior_Model(1).rho0_2 - Interior_Model(2).rho0;

%% LATERAL RIGIDITY/VISCOSITY FIELDS (asthenosphere = Interior_Model(3))
% Read from data/io/io_mu_eta_variable.npz -- see the basis-convention
% caveat above before trusting a Gate C disagreement.
npz_path = fullfile(repo_root, 'data', 'io', 'io_mu_eta_variable.npz');
lv = read_io_mu_eta_npz(npz_path);   % local function, see bottom of file

Interior_Model(3).mu_variable  = [double(lv.mu_n(:)),  double(lv.mu_m(:)),  (lv.mu_amp_real(:)  + 1i*lv.mu_amp_imag(:))];
Interior_Model(3).eta_variable = [double(lv.eta_n(:)), double(lv.eta_m(:)), (lv.eta_amp_real(:) + 1i*lv.eta_amp_imag(:))];

%% NUMERICS (single tractable rung -- the spec's Nrbase=50; matches
% Consistency_test_Energy.m's Numerics block otherwise, including
% Numerics.Nenergy = 12).
Numerics.Nlayers = 4;
Numerics.method = 'combination';
Numerics.Nrbase = 50;
Numerics.perturbation_order = 2;
Numerics.solution_cutoff = 12;
Numerics.load_couplings = 2;             % search/generate a generic coupling file
Numerics.Nenergy = 12;
Numerics.rheology_cutoff = 2;
Numerics.minimum_rheology_value = -13;
Numerics.parallel_sol = 0;
Numerics.parallel_gen = 0;
Numerics.coupling_file_location = [coupling_dir filesep];

[Numerics, Interior_Model] = set_boundary_indices(Numerics, Interior_Model, 'verbose');
if ~(Numerics.Nlayers == length(Interior_Model))
    error('Numerics.Nlayers must equal length(Interior_Model)');
end

%% FORCING (verbatim from tests/Consistency_test_Energy.m lines 197-209 /
% pylov3d.io_lateral.IO_FORCING_COMPONENTS).
omega0 = 4.1086E-05;
Forcing(1).Td = 2*pi/omega0;
Forcing(1).n  = 2;
Forcing(1).m  = 0;
Forcing(1).F  = 3/4*sqrt(1/5);
Forcing(2).Td = 2*pi/omega0;
Forcing(2).n  = 2;
Forcing(2).m  = -2;
Forcing(2).F  = -7/8*sqrt(6/5);
Forcing(3).Td = 2*pi/omega0;
Forcing(3).n  = 2;
Forcing(3).m  = 2;
Forcing(3).F  = 1/8*sqrt(6/5);

out_dir = fullfile(repo_root, 'data', 'tests', 'io');
if ~isfolder(out_dir)
    mkdir(out_dir);
end
log_path = fullfile(out_dir, 'io_energy_cross_check.log');
logf = fopen(log_path, 'w');
logp = @(varargin) both_print(logf, varargin{:});

logp('\n============ IO LATERAL VISCOELASTIC ENERGY CROSS-CHECK (MATLAB LOV3D) ============\n');
logp('  TASK-046 Gate C anchor: pylov3d.io_lateral / scripts/io_energy_consistency.py\n');
logp('  MATLAB version: %s\n', version);
logp('  Nrbase=%d  perturbation_order=%d  method=%s  Nlayers=%d  Nenergy=%d\n', ...
    Numerics.Nrbase, Numerics.perturbation_order, Numerics.method, Numerics.Nlayers, Numerics.Nenergy);

%% --- uniform (no-lateral) reference solve, all three forcings -------------
IM_uni = Interior_Model;
IM_uni(3).mu_variable  = [];
IM_uni(3).eta_variable = [];
IM_uni = get_rheology(IM_uni, Numerics, Forcing);

Love_Uni = repmat(struct(), 1, length(Forcing));
y_Uni = repmat(struct(), 1, length(Forcing));
for i = 1:length(Forcing)
    [Love_Uni(i), y_Uni(i)] = get_Love(IM_uni, Forcing(i), Numerics, 'verbose');
end

%% --- lateral (coupled) solve, all three forcings ---------------------------
IM_lat = get_rheology(Interior_Model, Numerics, Forcing);

Love_Lat = repmat(struct(), 1, length(Forcing));
y_Lat = repmat(struct(), 1, length(Forcing));
for i = 1:length(Forcing)
    [Love_Lat(i), y_Lat(i)] = get_Love(IM_lat, Forcing(i), Numerics, 'verbose');
end

%% --- direct integrated energy (get_energy.m general multi-forcing form) ---
[Energy_Uni] = get_energy(y_Uni, Numerics, Forcing, IM_uni, 'verbose', 1, 'calc_E_contribution', 0);
[Energy_Lat] = get_energy(y_Lat, Numerics, Forcing, IM_lat, 'verbose', 1, 'calc_E_contribution', 0);
e_direct_uni = Energy_Uni.energy_integral(1);
e_direct_lat = Energy_Lat.energy_integral(1);

%% --- Love-number-derived energy estimate (same cross-forcing double sum
%% as tests/Consistency_test_Energy.m lines 246-261, and
%% scripts/io_energy_consistency.py's love_energy_estimate) ------------------
E_k_uni = 0;
E_k_lat = 0;
for i = 1:length(Forcing)
    n_f = Forcing(i).n;
    m_f = Forcing(i).m;
    for j = 1:length(Forcing)
        ind1 = find(Love_Uni(j).n == n_f & Love_Uni(j).m == m_f);
        ind2 = find(Love_Lat(j).n == n_f & Love_Lat(j).m == m_f);
        if ~isempty(ind1)
            E_k_uni = E_k_uni - Forcing(i).F*Forcing(j).F*imag(Love_Uni(j).k(ind1));
        end
        if ~isempty(ind2)
            E_k_lat = E_k_lat - Forcing(i).F*Forcing(j).F*imag(Love_Lat(j).k(ind2));
        end
    end
end
% MATLAB's own normalization (see tests/Consistency_test_Energy.m lines
% 263-266); Python derives its own C = gs_norm(surface)^2/(2*Gg) from the
% codebase's global_dissipation convention (see scripts/
% io_energy_consistency.py's _normalization_prefactor docstring) -- both
% are recorded here so the two can be compared term-by-term rather than
% assumed equal.
e_love_uni_matlab_norm = 2*pi*10/(4*pi*IM_uni(end).Gg) * E_k_uni;
e_love_lat_matlab_norm = 2*pi*10/(4*pi*IM_lat(end).Gg) * E_k_lat;

logp('\n------------------------------------------------------------------------\n');
logp('  Forcing-mode complex k (all three forcings, uniform then lateral):\n');
for i = 1:length(Forcing)
    idx_u = find(Love_Uni(i).n == Forcing(i).n & Love_Uni(i).m == Forcing(i).m, 1);
    idx_l = find(Love_Lat(i).n == Forcing(i).n & Love_Lat(i).m == Forcing(i).m, 1);
    logp('    forcing %d (n=%d,m=%+d): k_uni=%+.10f%+.10fi   k_lat=%+.10f%+.10fi\n', ...
        i, Forcing(i).n, Forcing(i).m, ...
        real(Love_Uni(i).k(idx_u)), imag(Love_Uni(i).k(idx_u)), ...
        real(Love_Lat(i).k(idx_l)), imag(Love_Lat(i).k(idx_l)));
end
logp('\n  N coupled modes (lateral, per forcing): %s\n', mat2str(arrayfun(@(s) length(s.n), Love_Lat)));
logp('  Direct energy (uniform, lateral): %.10e   %.10e\n', e_direct_uni, e_direct_lat);
logp('  E_k raw double-sum (uniform, lateral): %.10e   %.10e\n', E_k_uni, E_k_lat);
logp('  Love-derived energy, MATLAB x10/Gg norm (uniform, lateral): %.10e   %.10e\n', ...
    e_love_uni_matlab_norm, e_love_lat_matlab_norm);
logp('  Gg (uniform model, surface layer): %.10e\n', IM_uni(end).Gg);
logp('========================================================================\n\n');
fclose(logf);
fprintf('  saved log: %s\n', log_path);

%% --- selected first-/second-order coupled complex k coefficients ----------
% "First-order" here means a coupled mode reached by a single lateral-
% variation step from the forcing mode; "second-order" means two steps.
% Reported for the (2,0) forcing's own coupled spectrum (Love_Lat(1)),
% which is built from Numerics.perturbation_order = 2 -- i.e. it already
% contains both orders together; this block just lists every non-forcing
% mode present (no separate order bookkeeping is exposed by get_Love, so
% "selected" here means "all coupled modes other than the forcing mode
% itself", sorted by |k|).
n_s = Love_Lat(1).n(:);
m_s = Love_Lat(1).m(:);
k_s = Love_Lat(1).k(:);
is_forcing_mode = (n_s == Forcing(1).n) & (m_s == Forcing(1).m);
[~, ord] = sort(abs(k_s), 'descend');
nshow = min(20, length(ord));

%% SAVE VERIFICATION ARTIFACT
io_energy = struct();
io_energy.Nrbase = Numerics.Nrbase;
io_energy.Nr = Numerics.Nr;
io_energy.perturbation_order = Numerics.perturbation_order;
io_energy.rheology_cutoff = Numerics.rheology_cutoff;
io_energy.Nenergy = Numerics.Nenergy;
io_energy.forcing_n = arrayfun(@(f) f.n, Forcing);
io_energy.forcing_m = arrayfun(@(f) f.m, Forcing);
io_energy.forcing_F = arrayfun(@(f) f.F, Forcing);
io_energy.k_uni_forcing = arrayfun(@(i) Love_Uni(i).k(find(Love_Uni(i).n==Forcing(i).n & Love_Uni(i).m==Forcing(i).m,1)), 1:length(Forcing));
io_energy.k_lat_forcing = arrayfun(@(i) Love_Lat(i).k(find(Love_Lat(i).n==Forcing(i).n & Love_Lat(i).m==Forcing(i).m,1)), 1:length(Forcing));
io_energy.N_coupled_modes = arrayfun(@(s) length(s.n), Love_Lat);
io_energy.n_s_forcing1 = n_s;
io_energy.m_s_forcing1 = m_s;
io_energy.k_s_forcing1 = k_s;
io_energy.is_forcing_mode_forcing1 = is_forcing_mode;
io_energy.top20_idx_by_abs_k_forcing1 = ord(1:nshow);
io_energy.e_direct_uni = e_direct_uni;
io_energy.e_direct_lat = e_direct_lat;
io_energy.E_k_uni = E_k_uni;
io_energy.E_k_lat = E_k_lat;
io_energy.e_love_uni_matlab_norm = e_love_uni_matlab_norm;
io_energy.e_love_lat_matlab_norm = e_love_lat_matlab_norm;
io_energy.Gg_uni = IM_uni(end).Gg;
io_energy.Gg_lat = IM_lat(end).Gg;
io_energy.matlab_version = version;
io_energy.Ks0 = 200e12;
save(fullfile(out_dir, 'io_energy_cross_check.mat'), '-struct', 'io_energy');
fprintf('  saved artifact: %s\n\n', fullfile(out_dir, 'io_energy_cross_check.mat'));


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
function s = read_io_mu_eta_npz(npz_path)
    % Extracts the 1-D numeric arrays this script needs (mu_n, mu_m,
    % mu_amp_real, mu_amp_imag, eta_n, eta_m, eta_amp_real, eta_amp_imag)
    % from a NumPy .npz (a plain zip of .npy files). Scalar/string
    % provenance fields in the npz are ignored (this script hardcodes the
    % model/forcing constants directly -- see header comment).
    if ~isfile(npz_path)
        error(['npz not found: %s\nRun scripts/io_energy_consistency.py ' ...
               'on the Python side first (it writes this export as a ' ...
               'side effect).'], npz_path);
    end
    tmp = tempname;
    mkdir(tmp);
    cleanup = onCleanup(@() rmdir(tmp, 's'));
    unzip(npz_path, tmp);
    s.mu_n        = read_npy(fullfile(tmp, 'mu_n.npy'));
    s.mu_m        = read_npy(fullfile(tmp, 'mu_m.npy'));
    s.mu_amp_real = read_npy(fullfile(tmp, 'mu_amp_real.npy'));
    s.mu_amp_imag = read_npy(fullfile(tmp, 'mu_amp_imag.npy'));
    s.eta_n        = read_npy(fullfile(tmp, 'eta_n.npy'));
    s.eta_m        = read_npy(fullfile(tmp, 'eta_m.npy'));
    s.eta_amp_real = read_npy(fullfile(tmp, 'eta_amp_real.npy'));
    s.eta_amp_imag = read_npy(fullfile(tmp, 'eta_amp_imag.npy'));
end

function arr = read_npy(fname)
    % Minimal reader for 1-D little-endian NumPy .npy v1.0 arrays
    % (dtype '<i8' or '<f8', C-order). Sufficient for the committed field
    % (same helper as scripts/moon_lateral_cross_check.m).
    fid = fopen(fname, 'r');
    if fid < 0, error('cannot open %s', fname); end
    cleanup = onCleanup(@() fclose(fid));
    magic = fread(fid, 6, '*char')';
    if ~strcmp(magic, sprintf('\x93NUMPY'))
        error('%s is not a .npy file', fname);
    end
    fread(fid, 2, 'uint8');
    header_len = fread(fid, 1, 'uint16');
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
