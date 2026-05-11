function generate_dataset_Project_sim(MAX_CASES)
% ============================================================
% DATASET GENERATOR – TWO SERIES DISTRIBUTED TRANSMISSION LINES
% Total Length = 300 km
% Fault applied at junction of Line_1 and Line_2
% ============================================================

clc;
disp("===== STARTING DATASET GENERATION (SERIES LINES) =====");

% ---------- Quick test ----------
if nargin == 0
    MAX_CASES = inf;
end

% ---------- Model ----------
model = bdroot;
if isempty(model)
    error('Open the Simulink model first.');
end
open_system(model);

% ---------- Dataset path ----------
dataset_path = ...
'C:\Users\Achuoth Akol Achuoth\OneDrive\Documents\MATLAB\dataset';
if ~exist(dataset_path,'dir')
    mkdir(dataset_path);
end

% ---------- Per-unit base ----------
S_BASE = 100e6;          % 100 MVA
V_BASE = 110000;         % 110 kV
I_BASE = S_BASE/(sqrt(3)*V_BASE);

% ---------- Configuration ----------
fault_types = ["NOFAULT","LG","LL","LLG","LLL","LLLG"];
Rf_list = [100 50 10 7 5 1 0.1 0.01 0];

% Fault distances along a 300 km line (from source)
fault_distances = [10 30 60 90 120 150 180 210 240 270]; % km
TOTAL_LINE_LENGTH = 300; % km

% ---------- Block paths ----------
line1 = [model '/Distributed Parameters Line'];
line2 = [model '/Distributed Parameters Line1'];
fault_block = [model '/Three-Phase Fault1'];

% ---------- Storage ----------
ALL_INPUTS = [];
ALL_OUTPUTS = [];
labels_list = table();
case_id = 0;

% ================= MAIN LOOP =================
for ft = 1:numel(fault_types)
    ftype = fault_types(ft);

    for r = 1:numel(Rf_list)
        Rf = Rf_list(r);

        for d = 1:numel(fault_distances)

            case_id = case_id + 1;
            if case_id > MAX_CASES
                break;
            end

            % -------- Distance split --------
            L1 = fault_distances(d);
            L2 = TOTAL_LINE_LENGTH - L1;

            % -------- Location label (300 km 기준) --------
            if L1 <= 100
                loc_label = "near_source";
            elseif L1 <= 200
                loc_label = "mid_line";
            else
                loc_label = "near_load";
            end

            fprintf("Case %d | %s | Rf=%.2f | Fault @ %d km\n", ...
                case_id, ftype, Rf, L1);

            % -------- Set line lengths --------
            set_param(line1,'Length',num2str(L1));
            set_param(line2,'Length',num2str(L2));

            % -------- Reset & apply fault --------
            set_all_faults_OFF({fault_block});
            [phases,G] = set_fault_type(fault_block, ftype);
            set_param(fault_block,'FaultResistance',num2str(Rf));

            % -------- Simulate --------
            out = sim(model,'StopTime','0.07');

            % -------- Extract RMS --------
            Tin  = create_csv_table(out.simout_in);
            Tout = create_csv_table(out.simout_out);

            Tin  = Tin(any(Tin{:,:}~=0,2),:);
            Tout = Tout(any(Tout{:,:}~=0,2),:);

            if isempty(Tin) || isempty(Tout)
                continue;
            end

            % -------- Per-unit normalization --------
            Tin{:,1:3}  = Tin{:,1:3}/I_BASE;
            Tin{:,4:6}  = Tin{:,4:6}/(V_BASE/sqrt(3));
            Tout{:,1:3} = Tout{:,1:3}/I_BASE;
            Tout{:,4:6} = Tout{:,4:6}/(V_BASE/sqrt(3));

            % -------- Labels --------
            Tin.fault_type = repmat(string(ftype),height(Tin),1);
            Tin.location   = repmat(loc_label,height(Tin),1);
            Tin.A = repmat(phases(1),height(Tin),1);
            Tin.B = repmat(phases(2),height(Tin),1);
            Tin.C = repmat(phases(3),height(Tin),1);
            Tin.G = repmat(G,height(Tin),1);
            Tout.fault_type = repmat(string(ftype),height(Tout),1);
            Tout.location   = repmat(loc_label,height(Tout),1);
            Tout.A = repmat(phases(1),height(Tout),1);
            Tout.B = repmat(phases(2),height(Tout),1);
            Tout.C = repmat(phases(3),height(Tout),1);
            Tout.G = repmat(G,height(Tout),1);
            % -------- Append --------
            ALL_INPUTS  = [ALL_INPUTS; Tin];
            ALL_OUTPUTS = [ALL_OUTPUTS; Tout];

            labels_list = [labels_list;
                table(case_id,string(ftype),loc_label,Rf,L1, ...
                'VariableNames', ...
                {'case_id','fault_type','location','Rf','distance_km'})];
        end
    end
end

% ---------- Save ----------
writetable(ALL_INPUTS,  fullfile(dataset_path,'inputs.csv'));
writetable(ALL_OUTPUTS, fullfile(dataset_path,'outputs.csv'));
writetable(labels_list, fullfile(dataset_path,'labels.csv'));

disp("===== DATASET GENERATION COMPLETE =====");
end

% ============================================================
% HELPER FUNCTIONS
% ============================================================

function set_all_faults_OFF(blocks)
for i = 1:numel(blocks)
    set_param(blocks{i}, ...
        'FaultA','off','FaultB','off','FaultC','off','GroundFault','off');
end
end

function [phases,G] = set_fault_type(block,type)

type = upper(string(type));
phases = [0 0 0];
G = 0;

set_all_faults_OFF({block});

switch type
    case 'NOFAULT'
        return;

    case 'LG'
        p = randi(3);
        phases(p) = 1;
        G = 1;

    case 'LL'
        s = randsample(1:3,2);
        phases(s) = 1;

    case 'LLG'
        s = randsample(1:3,2);
        phases(s) = 1;
        G = 1;

    case 'LLL'
        phases = [1 1 1];

    case 'LLLG'
        phases = [1 1 1];
        G = 1;
end

set_param(block, ...
    'FaultA', tern(phases(1)), ...
    'FaultB', tern(phases(2)), ...
    'FaultC', tern(phases(3)), ...
    'GroundFault', tern(G));
end

function s = tern(x)
if x == 1
    s = 'on';
else
    s = 'off';
end
end

function T = create_csv_table(s)
vals = s.signals.values;
T = array2table(vals, ...
    'VariableNames',{'Ia','Ib','Ic','Va','Vb','Vc'});
end
