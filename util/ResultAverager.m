
function [] = ResultAverager(al, ml)

filepath = 'C:\Users\Byung-Jun Lee\Desktop\»õ Æú´õ\belief-propagation\results\0509\result_subband';
algorithms = {'HIERMULTI_','HIERSINGLE_','LOCAL_','TREE_'};
mls = {'50','100','150','200','250','300','350','400'};

result = load([filepath algorithms{al} mls{ml} '.mat']);
result = result.results;

av = zeros(4,1);
count = 0;
for i=1:100
    if result(1,i) ~= -1
        av = av + result(:,i);
        count = count + 1;
    end
end
av = av / count;
av
end