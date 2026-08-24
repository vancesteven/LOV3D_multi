%% TASK-046 raw-grid MATLAB Gate C energy anchor
% Authoritative Io lateral anchor derived from the original physical
% lat/lon input path in tests/Consistency_test_Energy.m.
%
% Run from repo root:
%   matlab -batch "run('scripts/io_matlab_raw_grid_energy_anchor.m')"

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

%% Original Consistency_test_Energy.m physical lateral field
l_max = 100;
delta_lon = 180/(2*(2*l_max-1));
delta_lat = 180/(2*(2*l_max-1));
lon0 = -180 + delta_lon/2;
lonM =  180 - delta_lon/2;
lat0 = -90 + delta_lat/2;
latM =  90 - delta_lon/2;
lon = lon0:delta_lon:lonM;
lat = lat0:delta_lat:latM;
colat = (90-lat)*pi/180;
lon_rad = lon*pi/180;
P_lm = legendre(2,cos(colat)');
gv = zeros(length(colat),length(lon_rad));
phase = 0;
for m=1:3
    if m == 1
        fac = -(33/7);
    elseif m == 3
        fac = (9/14);
    else
        fac = 0;
    end
    v1 = (-1)^(m-1)*fac*squeeze(P_lm(m,:));
    if m == 1
        for i=1:length(v1)
            gv(i,:) = gv(i,:) + v1(i);
        end
    else
        for i=1:length(v1)
            gv(i,:) = gv(i,:) + v1(i)*(cos(phase)*cos((m-1)*lon_rad) + sin(phase)*sin((m-1)*lon_rad));
        end
    end
end
z_val = (21/5 + 0.5*gv)/(21/5);
lon_deg = lon_rad*180/pi;
[lon_g,lat_g] = meshgrid(lon_deg,lat);
Psi_latlon.lon = lon_g;
Psi_latlon.lat = lat_g;
Psi_latlon.z = z_val;
Psi_latlon.lmax = 2*l_max-1;

Q_mean = 2.3;
Q_diff_latlon = Psi_latlon.z*Q_mean - Q_mean;
Phi_diff_latlon = 0.01*Q_diff_latlon;
Phi_mean = 0.1;
eta_start_latlon = exp(-20*Phi_diff_latlon);
B_mu = 67/15;
mu_start_latlon = (1+B_mu*Phi_mean)./(1+B_mu*(Phi_diff_latlon + Phi_mean));
mu_start.lon = Psi_latlon.lon; mu_start.lat = Psi_latlon.lat;
mu_start.z = mu_start_latlon; mu_start.lmax = Psi_latlon.lmax;
eta_start.lon = Psi_latlon.lon; eta_start.lat = Psi_latlon.lat;
eta_start.z = eta_start_latlon; eta_start.lmax = Psi_latlon.lmax;

%% Four-layer Io model
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
Interior_Model(3).mu_latlon = mu_start;
Interior_Model(3).eta_latlon = eta_start;
Interior_Model(4).R0 = 1821.6;
Interior_Model(4).rho0 = 3244.0;
Interior_Model(4).Ks0 = 200e12;
Interior_Model(4).mu0 = 6.5e10;
Interior_Model(4).eta0 = 1e23;
Interior_Model(1).Delta_rho0 = Interior_Model(1).rho0_2 - Interior_Model(2).rho0;

%% Numerics / forcing
Numerics.Nlayers = 4;
Numerics.method = 'combination';
Numerics.Nrbase = 50;
Numerics.perturbation_order = 2;
Numerics.solution_cutoff = 12;
% get_Love defines Couplings only when lateral rheology is present and
% load_couplings is 1 or 2.  Use the original consistency-test path (1):
% load a matching file if present, otherwise compute and cache it.
Numerics.load_couplings = 1;
Numerics.Nenergy = 12;
Numerics.rheology_cutoff = 2;
Numerics.minimum_rheology_value = -13;
Numerics.parallel_sol = 0;
Numerics.parallel_gen = 0;
Numerics.coupling_file_location = [fullfile(repo_root,'data','couplings') filesep];
if ~isfolder(Numerics.coupling_file_location), mkdir(Numerics.coupling_file_location); end
[Numerics, Interior_Model] = set_boundary_indices(Numerics, Interior_Model);

omega0 = 4.1086E-05;
Forcing(1).Td = 2*pi/omega0; Forcing(1).n = 2; Forcing(1).m = 0;  Forcing(1).F = 3/4*sqrt(1/5);
Forcing(2).Td = 2*pi/omega0; Forcing(2).n = 2; Forcing(2).m = -2; Forcing(2).F = -7/8*sqrt(6/5);
Forcing(3).Td = 2*pi/omega0; Forcing(3).n = 2; Forcing(3).m = 2;  Forcing(3).F = 1/8*sqrt(6/5);

%% Uniform counterpart and rheology
Interior_Model_Uni = Interior_Model;
if isfield(Interior_Model_Uni,'mu_latlon'), Interior_Model_Uni = rmfield(Interior_Model_Uni,'mu_latlon'); end
if isfield(Interior_Model_Uni,'eta_latlon'), Interior_Model_Uni = rmfield(Interior_Model_Uni,'eta_latlon'); end
if isfield(Interior_Model_Uni,'k_latlon'), Interior_Model_Uni = rmfield(Interior_Model_Uni,'k_latlon'); end
IM = get_rheology(Interior_Model,Numerics,Forcing);
IMU = get_rheology(Interior_Model_Uni,Numerics,Forcing);

rv = IM(3).rheology_variable;
rv = rv(rv(:,1)>0,:);
counts = zeros(1,3);
for j=1:3
    active = get_active_modes(Numerics.perturbation_order,rv(:,1:2),Forcing(j));
    counts(j) = size(active,1);
end

%% Solve each forcing. Cells avoid dissimilar-struct assignment issues.
Love = cell(1,3); y = cell(1,3);
LoveU = cell(1,3); yU = cell(1,3);
for j=1:3
    [LoveU{j},yU{j}] = get_Love(IMU,Forcing(j),Numerics);
    [Love{j},y{j}] = get_Love(IM,Forcing(j),Numerics);
end

% get_energy expects a struct array. y outputs have common fields even though
% their contained arrays have forcing-dependent mode dimensions.
yS = [y{:}];
yUS = [yU{:}];
Energy = get_energy(yS,Numerics,Forcing,IM,'verbose',0,1,'calc_E_contribution',0);
EnergyU = get_energy(yUS,Numerics,Forcing,IMU,'verbose',0,1,'calc_E_contribution',0);

%% Love-derived forcing work
E_k = 0;
E_k_U = 0;
k_lat = zeros(1,3);
k_uni = zeros(1,3);
for i=1:3
    for j=1:3
        indU = find(LoveU{j}.n==Forcing(i).n & LoveU{j}.m==Forcing(i).m);
        indL = find(Love{j}.n==Forcing(i).n & Love{j}.m==Forcing(i).m);
        if ~isempty(indU)
            E_k_U = E_k_U - Forcing(i).F*Forcing(j).F*imag(LoveU{j}.k(indU));
        end
        if ~isempty(indL)
            E_k = E_k - Forcing(i).F*Forcing(j).F*imag(Love{j}.k(indL));
        end
    end
    selfU = find(LoveU{i}.n==Forcing(i).n & LoveU{i}.m==Forcing(i).m,1);
    selfL = find(Love{i}.n==Forcing(i).n & Love{i}.m==Forcing(i).m,1);
    k_uni(i) = LoveU{i}.k(selfU);
    k_lat(i) = Love{i}.k(selfL);
end

E_love_U = 5/IMU(end).Gg * E_k_U;
E_love = 5/IM(end).Gg * E_k;
E_direct_U = EnergyU.energy_integral(1);
E_direct = Energy.energy_integral(1);

fprintf('\nTASK-046 MATLAB raw-grid Gate C anchor, Nrbase=50\n');
fprintf('retained rheology modes: %d\n',size(rv,1));
fprintf('active solution counts m=[0,-2,+2]: [%d %d %d]\n',counts(1),counts(2),counts(3));
for j=1:3
    fprintf('m=%+d k_uni=%+.12e%+.12ei k_lat=%+.12e%+.12ei\n', ...
        Forcing(j).m,real(k_uni(j)),imag(k_uni(j)),real(k_lat(j)),imag(k_lat(j)));
end
fprintf('direct energy uniform/lateral: %.12e %.12e\n',E_direct_U,E_direct);
fprintf('Love energy uniform/lateral: %.12e %.12e\n',E_love_U,E_love);
fprintf('direct/Love mismatch uniform/lateral [%%]: %.8f %.8f\n', ...
    100*abs(E_direct_U-E_love_U)/abs(E_love_U), ...
    100*abs(E_direct-E_love)/abs(E_love));

outdir = fullfile(repo_root,'data','tests','io');
if ~isfolder(outdir), mkdir(outdir); end
outfile = fullfile(outdir,'io_raw_grid_energy_anchor.mat');
save(outfile,'k_uni','k_lat','counts','E_direct_U','E_direct','E_love_U','E_love','rv','Numerics','Forcing');
fprintf('saved: %s\n',outfile);
