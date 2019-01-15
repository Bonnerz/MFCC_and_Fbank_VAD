%% clear
clear all; close all; clc;

% parameter definition
filename = 'sx438.wav';
[x,fs]=audioread(filename);
% sound(x,16000); %可以听出是一段女声英文
frame_length_ms = 25;
frame_shift_ms = 10;
dc_offset = 300; %直流分量
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

% step 3 : extract frame (下面info的作用是直接输出info的信息)
info = audioinfo(filename); info
num_of_frames = floor((info.Duration * 1000 - frame_length_ms) / frame_shift_ms);   % avoid out of range
num_of_per_frame = (info.SampleRate * frame_length_ms / 1000); %每一帧的点数
frames = zeros(num_of_frames, num_of_per_frame);
for f = 1:num_of_frames
    idx = (f - 1) * (frame_shift_ms * info.SampleRate / 1000) + 1;  % addrees start from 1
    frames(f,:) = wav(idx:idx+num_of_per_frame-1)'; %一列信号，所以需要转置，点数是从1-400，161-560，...

    if add_rand_noise
        for i = 1:num_of_per_frame
            frames(f,i) = frames(f,i) - 1.0 + 2 * rand();   % rand [-1, 1)%每一列的值加上随机噪声
        end
    end
end
figure; 
subplot(3,1,1); plot(frames(51,:)); title('pre-sub-mean'); hold on; %frames(51,:)选取的是第51帧信号作为example


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
