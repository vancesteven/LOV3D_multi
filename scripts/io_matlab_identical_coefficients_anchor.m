%% TASK-046 strict identical-coefficient MATLAB solver-parity anchor
% Remove the raw-grid spherical-harmonic transform from the cross-language
% comparison. First construct the authoritative raw-grid rheology, then form
% a canonical six-mode coefficient field by averaging the small +m/-m
% transform asymmetry. Apply those coefficients to the *uniform* complex
% asthenosphere mean in both languages. This fixture tests only coupled-solver,
% Love-number, stress/strain and energy parity.
%
% Run from repo root:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch ...
%       "run('scripts/io_matlab_identical_coefficients_anchor.m')"

% Resolve repository paths before invoking another script. MATLAB run() changes
% the working directory to the target script's directory while it executes, so
% a nested relative path like scripts/... becomes scripts/scripts/... . Use an
% absolute path derived from this file instead.
this_file = mfilename('fullpath');
if isempty(this_file) || contains(this_file, 'LiveEditorEvaluationHelper')
    this_file = matlab.desktop.editor.getActiveFilename;
end
[this_dir, ~, ~] = fileparts(this_file);
repo_root = fileparts(this_dir);
raw_anchor = fullfile(repo_root,'scripts','io_matlab_raw_grid_energy_anchor.m');
run(raw_anchor);

%% Canonicalize the six lateral coefficients
rv_sym = rv;
for n = [2 4]
    ineg = find(rv_sym(:,1)==n & rv_sym(:,2)==-2,1);
    ipos = find(rv_sym(:,1)==n & rv_sym(:,2)==+2,1);
    if isempty(ineg) || isempty(ipos)
        error('Missing +/-2 rheology pair for n=%d',n);
    end
    pair_mean = 0.5*(rv_sym(ineg,4) + rv_sym(ipos,4));
    rv_sym(ineg,4) = pair_mean;
    rv_sym(ipos,4) = pair_mean;
end

% Use the uniform complex mean as the scalar background. This makes the
% strict fixture independent of any degree-0 quadrature difference in the raw
% grid and gives Python and MATLAB the same 1-D background already validated to
% ~1e-11. Only the six explicit lateral coefficients are added.
IMS = IMU;
IMS(3).rheology_variable = rv_sym;
IMS(3).uniform = 0;

counts_strict = zeros(1,length(Forcing));
LoveS = cell(1,length(Forcing));
yS = cell(1,length(Forcing));
for j=1:length(Forcing)
    active = get_active_modes(Numerics.perturbation_order,rv_sym(:,1:2),Forcing(j));
    counts_strict(j) = size(active,1);
    [LoveS{j},yS{j}] = get_Love(IMS,Forcing(j),Numerics);
end

EnergyS = get_energy([yS{:}],Numerics,Forcing,IMS,'verbose',0,1,'calc_E_contribution',0);

k_strict = zeros(1,length(Forcing));
E_k_strict = 0;
for i=1:length(Forcing)
    self = find(LoveS{i}.n==Forcing(i).n & LoveS{i}.m==Forcing(i).m,1);
    k_strict(i) = LoveS{i}.k(self);
    for j=1:length(Forcing)
        ind = find(LoveS{j}.n==Forcing(i).n & LoveS{j}.m==Forcing(i).m);
        if ~isempty(ind)
            E_k_strict = E_k_strict - Forcing(i).F*Forcing(j).F*imag(LoveS{j}.k(ind));
        end
    end
end
E_love_strict = 5/IMS(end).Gg * E_k_strict;
idx00 = find(EnergyS.n==0 & EnergyS.m==0,1);
E_direct_strict = EnergyS.energy_integral(idx00);

fprintf('\nTASK-046 MATLAB identical-coefficient solver-parity anchor, Nrbase=50\n');
fprintf('active solution counts m=[0,-2,+2]: [%d %d %d]\n',counts_strict(1),counts_strict(2),counts_strict(3));
fprintf('canonical rheology coefficients (n,m,Re,Im):\n');
for i=1:size(rv_sym,1)
    fprintf('  (%2d,%+3d) %+ .12e %+ .12e\n',rv_sym(i,1),rv_sym(i,2),real(rv_sym(i,4)),imag(rv_sym(i,4)));
end
for j=1:length(Forcing)
    fprintf('m=%+d k_strict=%+.12e%+.12ei\n',Forcing(j).m,real(k_strict(j)),imag(k_strict(j)));
end
fprintf('direct energy strict: %.12e\n',E_direct_strict);
fprintf('Love energy strict:   %.12e\n',E_love_strict);
fprintf('direct/Love mismatch [%%]: %.8f\n',100*abs(E_direct_strict-E_love_strict)/abs(E_love_strict));

outdir = fullfile(repo_root,'data','tests','io');
if ~isfolder(outdir), mkdir(outdir); end
outfile = fullfile(outdir,'io_identical_coefficients_anchor.mat');
save(outfile,'rv_sym','counts_strict','k_strict','E_direct_strict','E_love_strict','Forcing','Numerics','-v7');
fprintf('saved: %s\n',outfile);
