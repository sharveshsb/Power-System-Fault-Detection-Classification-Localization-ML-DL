function [phases, G] = set_fault_type(block, type)

set_all_faults_OFF({block});
type = upper(string(type));

phases = [0 0 0];   % [A B C]
G = 0;

switch type
    case 'NOFAULT'
        return;

    case 'LG'
        ph = randi(3);
        phases(ph) = 1;
        G = 1;

    case 'LL'
        sel = randsample(1:3,2);
        phases(sel) = 1;

    case 'LLG'
        sel = randsample(1:3,2);
        phases(sel) = 1;
        G = 1;

    case 'LLL'
        phases = [1 1 1];

    case 'LLLG'
        phases = [1 1 1];
        G = 1;
end

set_param(block, ...
    'FaultA', onoff(phases(1)), ...
    'FaultB', onoff(phases(2)), ...
    'FaultC', onoff(phases(3)), ...
    'GroundFault', onoff(G));
end
