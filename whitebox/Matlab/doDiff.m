function diffVec = doDiff(vec, tStep)
% 
% Test
% diffVec = doDiff(1:10, .5)
% 
if ~exist("tStep", 'var')
    tStep = 1;
end

diffVec = diff(vec);
diffVec = [diffVec(:); 0] / tStep; %scaled and same length

