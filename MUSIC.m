%By Honghao Tang
% Mika edit
clc
close all
clear all
format long             % The data show that as long shaping scientific
doa=[0 12 60 -20]/180*pi;   % Direction of arrival of sources
N=512;                  % Snapshots, number of samples
w=[pi/4 pi/3 pi/20 pi]';   % Frequency of sources
M=64;                    % Number of array elements
P=length(w);            % The number of signal sources
f = 1e3;                % Frequency
c = 343;                % Speed of sound
lambda = f/c;           % Wavelength
%lambda=150;            % Wavelength
d=lambda/2;             % Element spacing

snr=20;%SNA
D=zeros(P,M); %To creat a matrix with P row and M column
for k=1:P
    D(k,:)=exp(-1j*2*pi*d*sin(doa(k))/lambda*[0:M-1]); %Assignment matrix
end
D=D';
xx=2*exp(1j*(w*[1:N])); %Simulate signal

% final simulated signals for all microphones, stored in x
x=D*xx;
%x=x+awgn(x,snr);%Insert Gaussian white noise

% Start MUSIC algorithm
R=x*x';         % Data covarivance matrix
[N,V]=eig(R);   % Find the eigenvalues and eigenvectors of R
NN=N(:,1:M-P);  % Estimate noise subspace
theta=linspace(-90,90,181); % Angles for peak search
for ii=1:length(theta)
    SS=zeros(1,length(M));
    for mic=0:M-1
        SS(1+mic)=exp(-1j*2*mic*pi*d*sin(theta(ii)/180*pi)/lambda);
    end
    PP=SS*NN*NN'*SS';       % A "power" value at a specific angle
    Pmusic(ii)=abs(1/ PP);
end
Pmusic=10*log10(Pmusic/max(Pmusic)); % Spatial spectrum function

plot(theta,Pmusic,'-k')
xlabel('angle \theta/degree')
ylabel('spectrum function P(\theta) /dB')
title('DOA estimation based on MUSIC algorithm ')
grid on
