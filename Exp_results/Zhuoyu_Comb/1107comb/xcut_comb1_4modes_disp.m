load D:/user/ZhuoYu/Zhuoyu_Xcut_Comb1/xcut_comb1_4modes_disp.mat;
ax1 = axes; 
plot(lum.x0, lum.y0,lum.x1, lum.y1,lum.x2, lum.y2,lum.x3, lum.y3,lum.x4, lum.y4)
set(ax1, 'XLim', [1.4 1.65])
set(ax1, 'YLim', [-16000 2000])
set(ax1,'XGrid', 'on')
set(ax1,'YGrid', 'on')
legend('line 1','line 2','line 3','line 4','line 5')
