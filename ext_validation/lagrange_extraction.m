format long
n = 4;
p = 3;
o = 3;
m = 1;

knots = augknt(0:1:n, p+1, m)
N = transpose(spcol(knots, p+1, 0:1/o:n))
for i = 1:n
    pos = m * (i - 1) + 1
    D(1:(p+1), 1:(o+1), i) = N(pos:pos+p, o*(i-1) + 1:o*i + 1);
end
D