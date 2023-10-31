clc;
clear ;
close all;
%I2=imread('fort.jpg');
I2 = imread('Distorted_image.jpg'); %I2=imread('BD.png');
%I2=rgb2gray(I2);
% imshow(I2);



 cor_fac=input('Enter correction factor:');
 N=size(I2,2); %M = Nos. of col = 320
 M=size(I2,1); %N = Nos. of row = 240
 B=zeros(M,N,'double');
 
 d=zeros(M,N,'double');
 dev_fac=zeros(M,N,'double');
 theta=zeros(M,N,'double');
 corrected_x=zeros(M,N,'double');
 corrected_y=zeros(M,N,'double');
 
 im_rad=sqrt(((M-1)^2)+((N-1)^2));
 corrected_dist=im_rad/cor_fac;
 center = [round(M/2) round(N/2)];
 %R = sqrt(center(1)^2 + center(2)^2);

 [xi,yi] = meshgrid(1:N,1:M);
    % Creates converst the mesh into a colum vector of coordiantes relative to the center
    xt = xi(:) - center(2); 
    yt = yi(:) - center(1);
    xt_mat=vec2mat(xt,M);
    xt_mat= xt_mat.';
    yt_mat=vec2mat(yt,M);
    yt_mat= yt_mat.';
    
%     xt_mat= xlsread('xt_mat');
%     yt_mat= xlsread('yt_mat');
%     
 for pxrow= 1:M
     
     for pxcol= 1:N
         
         d(pxrow,pxcol)= sqrt(xt_mat(pxrow,pxcol)^2+yt_mat(pxrow,pxcol)^2);
         
         dev_fac(pxrow,pxcol)=d(pxrow,pxcol)/corrected_dist;
         
         dev_fac_t=dev_fac(pxrow,pxcol);
         
         theta(pxrow,pxcol)= atan(dev_fac_t)/dev_fac_t;
         %theta_d(pxrow,pxcol)= (theta(pxrow,pxcol)*180)/pi;
  
         corrected_y(pxrow,pxcol)= floor((N/2)+  (theta(pxrow,pxcol)*xt_mat(pxrow,pxcol)));
         corrected_x(pxrow,pxcol)=floor((M/2)+  (theta(pxrow,pxcol)*yt_mat(pxrow,pxcol)));
         
%          corr_x1=uint8(corrected_x);
%          corr_y1=uint8(corrected_y);
           
     end 
     
 end
%         corr_x =corr_x1(1:N,1:M);
%         corr_y =corr_y1(1:N,1:M);
        corrected_x(center(1),center(2))=128;
        corrected_y(center(1),center(2))=128;
         
         
 for i=1:M
     for j=1:N
         
%           corrected_y(i,j)= floor((N/2)+  (theta(i,j)*xt_mat(i,j)));
%          corrected_x(i,j)=floor((M/2)+  (theta(i,j)*yt_mat(i,j)));
% %          
       
         B(i,j)=I2(corrected_x(i,j),corrected_y(i,j));
     
     end
 end
% img = image(B);
% figure, imshow(I2);
 
X8 = uint8(round(B-1));
figure(2)
imshow(X8)
%  

% img = sparse( corrected_x, corrected_y, true, size(I2,1), size(I2,2));
% img = sparse(x, y, true, size(I,1), size(I,2)); %// Use this if x is row and y is column
% img = full(img);
% imshow(img);
