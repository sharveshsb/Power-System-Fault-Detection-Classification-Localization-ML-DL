function [A,B,C,G] = fault_to_flags(ftype)

A=0; B=0; C=0; G=0;

switch upper(ftype)

case 'NOFAULT'
    % nothing

case 'LG'
    ph = randi(3);
    if ph==1, A=1; end
    if ph==2, B=1; end
    if ph==3, C=1; end
    G = 1;

case 'LL'
    pairs=[1 2;1 3;2 3];
    sel=pairs(randi(3),:);
    if ismember(1,sel), A=1; end
    if ismember(2,sel), B=1; end
    if ismember(3,sel), C=1; end

case 'LLG'
    pairs=[1 2;1 3;2 3];
    sel=pairs(randi(3),:);
    if ismember(1,sel), A=1; end
    if ismember(2,sel), B=1; end
    if ismember(3,sel), C=1; end
    G = 1;

case 'LLL'
    A=1; B=1; C=1;

case 'LLLG'
    A=1; B=1; C=1; G=1;
end
end
