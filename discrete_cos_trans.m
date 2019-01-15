function [ dct_mat ] = discrete_cos_trans( N, M )
% DCT : ¿Î…¢”‡œ“±‰ªª
% N : cols, M : rows
% dct_mat : M by N matrix
dct_mat = zeros(M, N);

for i = 1:M
    for j = 1:N
        dct_mat(i,j) = cos(i * pi / N * (j - 0.5));
    end
end

end
