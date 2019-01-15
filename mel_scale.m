function [ mel_f ] = mel_scale( f )
% calculate mel_scale
%   f   : input frequency scalar or frequency vector
% mel_f : output frequency scalar or frequency vector

mel_f = 2595*log10(1 + f/700);

end

