%% MARS 1D CROSS-CHECK (TASK-014 part 1)
% Runs the native LOV3D 1D solver on the exact 4-layer Mars reference model
% from docs/MARS_MODEL.md / pylov3d/mars.py, and prints k2/h2/l2 for direct
% comparison against the Python port (which gives k2=0.169, h2=0.315632,
% l2=0.051596). This gives Mars the same MATLAB anchor the Moon has and
% adjudicates the open h2/k2=1.87 question (real coarse-4-layer artifact vs
% a Python bug).
%
% The model arrays are copied VERBATIM from build_mars_model() output so the
% cross-check exercises the solver, not the density/mu fit:
%   R0 [km]   : 1830.0, 2340.0, 3339.5, 3389.5
%   rho0      : 6128.075996, 4136.503827, 3400, 2900   kg/m^3
%   mu0 [Pa]  : 0, 96482476610.2174, 67537733627.15218, 30e9
%   Ks0 [Pa]  : 155e9, 160e9, 115e9, 70e9
% Purely elastic (eta0 = NaN in every mantle/crust layer).
%
% Run headless from the repo root:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch \
%       "run('scripts/mars_1d_cross_check.m')"

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

%% INTERIOR MODEL (verbatim from pylov3d.mars.build_mars_model)
% Interior_Model goes from core (1) to surface (n).
% Core: liquid (mu0 = 0), represented by LOV3D's native fluid-CMB condition.
Interior_Model(1).R0    = 1830.0;        % CMB radius [km]
Interior_Model(1).rho0  = 6128.075995512780;  % fitted core density [kg/m^3]
Interior_Model(1).rho0_2 = 6128.075995512780; % rho_2 (icy-moon field; = rho0 here)

% Lower mantle
Interior_Model(2).R0   = 2340.0;         % outer radius [km]
Interior_Model(2).rho0 = 4136.503827041973;
Interior_Model(2).Ks0  = 160e9;          % bulk modulus [Pa]
Interior_Model(2).mu0  = 96482476610.2174;    % 100e9 * MARS_MU_SCALE
% elastic layer: eta0 is OMITTED (MATLAB get_rheology treats a layer as
% elastic only when eta0 is EMPTY/absent; eta0=NaN would wrongly enter the
% viscoelastic branch and poison the solve with NaN).

% Upper mantle
Interior_Model(3).R0   = 3339.5;
Interior_Model(3).rho0 = 3400.0;
Interior_Model(3).Ks0  = 115e9;
Interior_Model(3).mu0  = 67537733627.15218;   % 70e9 * MARS_MU_SCALE
% elastic: eta0 omitted (see note above)

% Crust
Interior_Model(4).R0   = 3389.5;
Interior_Model(4).rho0 = 2900.0;
Interior_Model(4).Ks0  = 70e9;
Interior_Model(4).mu0  = 30e9;
% elastic: eta0 omitted (see note above)

% Core density contrast (matches the Python default: rho_core - rho_lm).
Interior_Model(1).Delta_rho0 = Interior_Model(1).rho0_2 - Interior_Model(2).rho0;

%% NUMERICS (matches make_numerics(n_layers=4, method='combination', Nrbase=100))
Numerics.Nlayers = 4;
Numerics.method = 'combination';
Numerics.Nrbase = 100;
Numerics.perturbation_order = 2;
Numerics.solution_cutoff = 12;
Numerics.load_couplings = 1;
Numerics.Nenergy = 12;
Numerics.rheology_cutoff = 2;
Numerics.parallel_sol = 0;
Numerics.parallel_gen = 0;
Numerics.coupling_file_location = [repo_root '/files/couplings/'];

[Numerics, Interior_Model] = set_boundary_indices(Numerics, Interior_Model, 'verbose');

if ~(Numerics.Nlayers == length(Interior_Model))
    error('Numerics.Nlayers must equal length(Interior_Model)');
end

%% FORCING (n=2, m=0 solar semidiurnal; Td = half a sol, irrelevant when elastic)
Forcing(1).Td = 44387.62;   % s (MARS_FORCING_TD)
Forcing(1).n = 2;
Forcing(1).m = 0;
Forcing(1).F = 1;

%% SOLVE
Interior_Model = get_rheology(Interior_Model, Numerics, Forcing);
[Love_Spectra, y] = get_Love(Interior_Model, Forcing, Numerics, 'verbose');

k2 = Love_Spectra.k(1);
h2 = Love_Spectra.h(1);
l2 = Love_Spectra.l(1);

%% REPORT
fprintf('\n================ MARS 1D CROSS-CHECK (MATLAB LOV3D) ================\n');
fprintf('  k2 = %.12f  (imag %.3e)\n', real(k2), imag(k2));
fprintf('  h2 = %.12f  (imag %.3e)\n', real(h2), imag(h2));
fprintf('  l2 = %.12f  (imag %.3e)\n', real(l2), imag(l2));
fprintf('  h2/k2 = %.6f\n', real(h2)/real(k2));
fprintf('-------------------------------------------------------------------\n');
fprintf('  Python (pylov3d) reference: k2=0.169000000000  h2=0.315632205682  l2=0.051595952202\n');
fprintf('  rel err  k2=%.3e  h2=%.3e  l2=%.3e\n', ...
    abs(real(k2)-0.169)/0.169, ...
    abs(real(h2)-0.315632205682)/0.315632205682, ...
    abs(real(l2)-0.051595952202)/0.051595952202);
fprintf('===================================================================\n\n');
