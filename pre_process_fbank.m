%% clear
clear all; close all; clc;

%% Pre-process
% parameter definition
filename = 'sx438.wav';
frame_length_ms = 25;
frame_shift_ms = 10;
dc_offset = 300;
pre_emphasis_coef = 0.97;
rand_noise_factor = 0.6;
add_rand_noise = true;

% step 1 : read wav
wav = audioread(filename, 'native');
figure;
subplot(2,1,1); plot(wav); title('waveform'); hold on;

% step 2 : add dc offset
wav = wav + dc_offset;
subplot(2,1,2); plot(wav); title('add dc offset'); hold on;

% step 3 : extract frame
info = audioinfo(filename); info
num_of_frames = floor((info.Duration * 1000 - frame_length_ms) / frame_shift_ms);   % avoid out of range
num_of_per_frame = (info.SampleRate * frame_length_ms / 1000);
frames = zeros(num_of_frames, num_of_per_frame);
for f = 1:num_of_frames
    idx = (f - 1) * (frame_shift_ms * info.SampleRate / 1000) + 1;  % addrees start from 1
    frames(f,:) = wav(idx:idx+num_of_per_frame-1)';

    if add_rand_noise
        for i = 1:num_of_per_frame
            frames(f,i) = frames(f,i) - 1.0 + 2 * rand();   % rand [-1, 1)
        end
    end
end
figure; 
subplot(3,1,1); plot(frames(51,:)); title('pre-sub-mean'); hold on;


% step 4 : sub mean
for f = 1:num_of_frames
    me = mean(frames(f,:));
    frames(f,:) = frames(f,:) - me;
end
subplot(3,1,2); plot(frames(51,:)); title('post-sub-mean'); hold on;

% step 5 : pre-emphasis
for f = 1:num_of_frames
    for i = num_of_per_frame:-1:2
        frames(f,i) = frames(f,i) - pre_emphasis_coef * frames(f,i-1);
    end
end
subplot(3,1,3); plot(frames(51,:)); title('post-pre-emph'); hold on;

% step 6 : hamming window
hamming = zeros(1, num_of_per_frame);
for i = 1:num_of_per_frame
    hamming(1,i) = 0.54 - 0.46*cos(2*pi*(i-1) / (num_of_per_frame-1));
end
figure;
plot(hamming); title('Hamming Window Function'); hold on;

% step 7 : add window
figure;
subplot(2,1,1); plot(1:400, frames(51,:),'r', 1:400, hamming, 'g'); title('pre-add-win'); hold on;
for f = 1:num_of_frames
    frames(f,:) = frames(f,:) .* hamming;
end
subplot(2,1,2); plot(1:400, frames(51,:)); title('post-add-win'); hold on;


%% FbankFilter Analysis
% fft
NFFT = 2^nextpow2(num_of_per_frame);    % Next power of 2 from length of num_of_per_frame %返回对A中每个元素满足的最小的2的幂的指数
frames_fft = complex(zeros(num_of_frames, NFFT));
for f = 1:num_of_frames
    frames_fft(f,:) = fft(frames(f,:), NFFT);   % pad 0
end
freq = info.SampleRate / 2 * linspace(0, 1, NFFT/2 + 1); % frequency of fft
figure; plot(freq, 2*abs(frames_fft(1,1:NFFT/2+1))); hold on;
xlabel('Frequency(Hz)'); ylabel('|frame\_fft|');

% compute energy
energy_frames = zeros(num_of_frames, NFFT/2);
for f = 1:num_of_frames
    for i = 1:NFFT/2
        energy_frames(f,i) = abs(frames_fft(f,i));
    end
end

% mel filter coef 将Hz2mel域
num_of_bins = 40;
low_freq = 20; high_freq = info.SampleRate / 2; %范围是20Hz-8000Hz

fft_bin_width = info.SampleRate / NFFT; %31.25
mel_low_freq = 2595*log10(1 + low_freq / 700);%31.7483,将20代入
mel_high_freq = 2595*log10(1 + high_freq / 700);%将8000代入，是2840.023
mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_of_bins + 1);%==68.4945

mel_coef = zeros(num_of_bins, (NFFT/2)+3);
for b = 1:num_of_bins %确定每一个的mel域的上边界，下边界，中心频率
    left_mel = mel_low_freq + (b - 1) * mel_freq_delta;
    center_mel = mel_low_freq + (b) * mel_freq_delta;
    right_mel = mel_low_freq + (b + 1) * mel_freq_delta;

    first_index = -1;
    last_index = -1;
    for f = 1:(NFFT/2) %1：512/2=1：256
        tmp_freq = (fft_bin_width * (f - 1)) + 1; %fft_bin_width = info.SampleRate / NFFT; %31.25
        tmp_mel_freq = 2595*log10(1 + tmp_freq / 700);%

        %下面是计算weight值，根据tmp_mel_freq与三个频率的关系来获得
        if tmp_mel_freq > left_mel && tmp_mel_freq < right_mel %首先满足在left_mel,right_mel之间，赋值weight=0.然后再计算weight
            weight = 0;
            if tmp_mel_freq < center_mel %
                weight = (tmp_mel_freq - left_mel) / (center_mel - left_mel);
            elseif tmp_mel_freq < right_mel
                weight = (right_mel - tmp_mel_freq) / (right_mel - center_mel);
            end

            if first_index == -1
                first_index = f;
            end
            last_index = f;

            mel_coef(b, f) = weight;
        end
    end
    mel_coef(b, (NFFT/2)+2) = first_index;%在mel_coef矩阵的最后两列记录first_index与last_index数据
    mel_coef(b, (NFFT/2)+3) = last_index;%这两列数据表明满足上述条件的weight值存在的index
end
figure;         
for b = 1:num_of_bins
    plot(mel_coef(b,1:(NFFT/2))); hold on;
end
xlabel('Frequency'); ylabel('mel_coef');

% take mel filter
mel_frames = zeros(num_of_frames, num_of_bins);
for f = 1:num_of_frames
    for i = 1:num_of_bins
        mel_frames(f,i) = sum(energy_frames(f,:) .* mel_coef(i,1:NFFT/2));%矩阵的行与mel_coef的行相乘
    end
end

% take log
log_mel_frames = log(mel_frames);