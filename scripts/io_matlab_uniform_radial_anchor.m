%% TASK-046 uniform radial-field MATLAB anchor
% Export the uniform Io radial solution, GSH displacement, stress and strain
% at Nrbase=50 so Python can be compared point-by-point downstream of the
% already validated Love-number solution.
%
% This uses the exact spherically symmetric branch of
% tests/Consistency_test_Energy.m. No lateral coefficient/basis convention is
% involved, so this is the authoritative diagnostic for the remaining uniform
% direct-energy discrepancy.
%
% Run from repo root:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch ...
%       "run('scripts/io_matlab_uniform_radial_anchor.m')"

clearvars
clc

mfile_name = mfilename('fullpath');
if isempty(mfile_name) || contains(mfile_name, 'LiveEditorEvaluationHelper')
    mfile_name = matlab.desktop.editor.getActiveFilename;
end
[pathstr, ~, ~] = fileparts(mfile_name);
repo_root = fileparts(pathstr);
cd(repo_root);
addpath(genpath(repo_root));

coupling_dir = fullfile(repo_root,'data','couplings');
if ~isfolder(coupling_dir), mkdir(coupling_dir); end
out_dir = fullfile(repo_root,'data','tests','io');
if ~isfolder(out_dir), mkdir(out_dir); end

%% Exact four-layer Io model, uniform branch
Interior_Model(1).R0 = 965.0;
Interior_Model(1).rho0 = 5150.0;
Interior_Model(1).rho0_2 = 5150.0;
Interior_Model(1).eta0 = [];

Interior_Model(2).R0 = 1591.6;
Interior_Model(2).rho0 = 3244.0;
Interior_Model(2).Ks0 = 200e12;
Interior_Model(2).mu0 = 6e10;
Interior_Model(2).eta0 = 1e20;

Interior_Model(3).R0 = 1791.6;
Interior_Model(3).rho0 = 3244.0;
Interior_Model(3).Ks0 = 200e12;
Interior_Model(3).mu0 = 7.8e5;
Interior_Model(3).eta0 = 1e11;

Interior_Model(4).R0 = 1821.6;
Interior_Model(4).rho0 = 3244.0;
Interior_Model(4).Ks0 = 200e12;
Interior_Model(4).mu0 = 6.5e10;
Interior_Model(4).eta0 = 1e23;
Interior_Model(1).Delta_rho0 = Interior_Model(1).rho0_2 - Interior_Model(2).rho0;

%% Numerics
Numerics.Nlayers = 4;
Numerics.method = 'combination';
Numerics.Nrbase = 50;
Numerics.perturbation_order = 2;
Numerics.solution_cutoff = 12;
Numerics.load_couplings = 0;
Numerics.Nenergy = 12;
Numerics.rheology_cutoff = 2;
Numerics.minimum_rheology_value = -13;
Numerics.parallel_sol = 0;
Numerics.parallel_gen = 0;
Numerics.coupling_file_location = [coupling_dir filesep];
[Numerics, Interior_Model] = set_boundary_indices(Numerics, Interior_Model);

%% Three eccentricity-tide forcing components
omega0 = 4.1086E-05;
Forcing(1).Td = 2*pi/omega0; Forcing(1).n = 2; Forcing(1).m = 0;  Forcing(1).F = 3/4*sqrt(1/5);
Forcing(2).Td = 2*pi/omega0; Forcing(2).n = 2; Forcing(2).m = -2; Forcing(2).F = -7/8*sqrt(6/5);
Forcing(3).Td = 2*pi/omega0; Forcing(3).n = 2; Forcing(3).m = 2;  Forcing(3).F = 1/8*sqrt(6/5);

IM = get_rheology(Interior_Model,Numerics,Forcing);

Love = repmat(struct(),1,length(Forcing));
y = repmat(struct(),1,length(Forcing));
for j=1:length(Forcing)
    [Love(j),y(j)] = get_Love(IM,Forcing(j),Numerics);
end
Energy = get_energy(y,Numerics,Forcing,IM);

% Uniform response has one solution mode per forcing. Keep complete 24-column
% get_solution output so Python can compare state, GSH displacement, stress,
% and strain without reconstructing MATLAB internals.
y_m0  = squeeze(y(1).y(:,:,1));
y_mm2 = squeeze(y(2).y(:,:,1));
y_mp2 = squeeze(y(3).y(:,:,1));
r = y_m0(:,1);
energy_n = Energy.n;
energy_m = Energy.m;
energy_integral = Energy.energy_integral;
energy_profile = Energy.energy;
Nrlayer = Numerics.Nrlayer;
BCindices = Numerics.BCindices;
Gg = IM(end).Gg;
muC = arrayfun(@(x) x.muC, IM);
lambda = arrayfun(@(x) x.lambda, IM);

% Forcing-mode k values, explicitly archived as a quick identity check.
k_forcing = zeros(1,length(Forcing));
for j=1:length(Forcing)
    idx = find(Love(j).n==Forcing(j).n & Love(j).m==Forcing(j).m,1);
    k_forcing(j) = Love(j).k(idx);
end

out_path = fullfile(out_dir,'io_uniform_radial_anchor.mat');
save(out_path,'r','y_m0','y_mm2','y_mp2','energy_n','energy_m', ...
    'energy_integral','energy_profile','Nrlayer','BCindices','Gg','muC', ...
    'lambda','k_forcing','-v7');

fprintf('\nTASK-046 MATLAB uniform radial anchor\n');
fprintf('Nrbase=50, Nr=%d\n',Numerics.Nr);
fprintf('Nrlayer: %s\n',mat2str(Numerics.Nrlayer));
fprintf('k forcing:');
for j=1:length(k_forcing)
    fprintf(' %+0.10f%+0.10fi',real(k_forcing(j)),imag(k_forcing(j)));
end
fprintf('\n');
idx00 = find(Energy.n==0 & Energy.m==0,1);
fprintf('direct E00: %.12e\n',Energy.energy_integral(idx00));
fprintf('saved: %s\n\n',out_path);