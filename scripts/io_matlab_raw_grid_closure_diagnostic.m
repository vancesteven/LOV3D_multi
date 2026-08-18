%% TASK-046 raw-grid MATLAB closure diagnostic
% Reproduce the ORIGINAL Consistency_test_Energy.m lateral-input path:
% build the closed-form Io heating pattern on the lat/lon grid, feed the full
% mu_latlon and eta_latlon structs to get_rheology, and report the retained
% rheology spectrum plus active-mode counts for the three eccentricity-tide
% forcing components.
%
% This deliberately avoids data/io/io_mu_eta_variable.npz.  The coefficient
% export uses pylov3d's scipy sph_harm_y basis, whereas MATLAB get_rheology.m
% interprets coefficient inputs in LOV3D/SPH_Tools conventions.  The previous
% 125-mode Gate-C anchor therefore remains a useful code-path diagnostic but
% is not authoritative for the intended physical Io field until this raw-grid
% path is compared.
%
% Run from repo root:
%   matlab -batch "run('scripts/io_matlab_raw_grid_closure_diagnostic.m')"

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

%% Original Consistency_test_Energy.m starting pattern
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
c = 0.01;
Phi_mean = 0.1;
Phi_diff_latlon = c*Q_diff_latlon;
B_eta = 20;
eta_start_latlon = exp(-B_eta*Phi_diff_latlon);
B_mu = 67/15;
mu_start_latlon = (1+B_mu*Phi_mean)./(1+B_mu*(Phi_diff_latlon + Phi_mean));

mu_start.lon = Psi_latlon.lon;
mu_start.lat = Psi_latlon.lat;
mu_start.z = mu_start_latlon;
mu_start.lmax = Psi_latlon.lmax;
eta_start.lon = Psi_latlon.lon;
eta_start.lat = Psi_latlon.lat;
eta_start.z = eta_start_latlon;
eta_start.lmax = Psi_latlon.lmax;

%% Exact four-layer Io model
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

%% Numerics and forcing
Numerics.Nlayers = 4;
Numerics.method = 'combination';
Numerics.Nrbase = 10; % closure is independent of radial resolution
Numerics.perturbation_order = 2;
Numerics.solution_cutoff = 12;
Numerics.load_couplings = 0;
Numerics.Nenergy = 12;
Numerics.rheology_cutoff = 2;
Numerics.minimum_rheology_value = -13;
Numerics.parallel_sol = 0;
Numerics.parallel_gen = 0;
Numerics.coupling_file_location = [fullfile(repo_root,'data','couplings') filesep];
[Numerics, Interior_Model] = set_boundary_indices(Numerics, Interior_Model);

omega0 = 4.1086E-05;
Forcing(1).Td = 2*pi/omega0; Forcing(1).n = 2; Forcing(1).m = 0;  Forcing(1).F = 3/4*sqrt(1/5);
Forcing(2).Td = 2*pi/omega0; Forcing(2).n = 2; Forcing(2).m = -2; Forcing(2).F = -7/8*sqrt(6/5);
Forcing(3).Td = 2*pi/omega0; Forcing(3).n = 2; Forcing(3).m = 2;  Forcing(3).F = 1/8*sqrt(6/5);

%% Process rheology exactly through the original grid path
IM = get_rheology(Interior_Model,Numerics,Forcing);
rv = IM(3).rheology_variable;
rv = rv(rv(:,1)>0,:);

fprintf('\nTASK-046 MATLAB raw-grid closure diagnostic\n');
fprintf('retained asthenosphere rheology modes: %d\n',size(rv,1));
if ~isempty(rv)
    fprintf('retained degree range: %d..%d\n',min(rv(:,1)),max(rv(:,1)));
    fprintf('retained (n,m,Re(muC),Im(muC)):\n');
    for i=1:size(rv,1)
        fprintf('  (%2d,%+3d)  %+ .8e  %+ .8e\n',rv(i,1),rv(i,2),real(rv(i,4)),imag(rv(i,4)));
    end
end

counts = zeros(1,length(Forcing));
for j=1:length(Forcing)
    active = get_active_modes(Numerics.perturbation_order,rv(:,1:2),Forcing(j));
    counts(j) = size(active,1);
end
fprintf('active solution counts for m=[0,-2,+2]: [%d %d %d]\n',counts(1),counts(2),counts(3));
fprintf('previous coefficient-path anchor counts: [125 125 125]\n');
fprintf('Python MATLAB-work-grid diagnostic counts: [43 41 41]\n\n');
