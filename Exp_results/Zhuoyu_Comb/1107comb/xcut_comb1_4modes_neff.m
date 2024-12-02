load D:/user/ZhuoYu/Zhuoyu_Xcut_Comb1/xcut_comb1_4modes_neff.mat;
ax1 = axes('NextPlot','add'); 
ax2 = axes('YAxisLocation','right','Color','none','NextPlot','add'); 
colorOrder = get(ax1,'ColorOrder'); 
coSize = size(colorOrder); 
ln0=line(lum.x0, lum.y0,'Parent',ax1, 'Color', colorOrder(mod(0,coSize(1))+1,:))
ln1=line(lum.x1, lum.y1,'Parent',ax2, 'Color', colorOrder(mod(1,coSize(1))+1,:))
ln2=line(lum.x2, lum.y2,'Parent',ax1, 'Color', colorOrder(mod(2,coSize(1))+1,:))
ln3=line(lum.x3, lum.y3,'Parent',ax2, 'Color', colorOrder(mod(3,coSize(1))+1,:))
ln4=line(lum.x4, lum.y4,'Parent',ax1, 'Color', colorOrder(mod(4,coSize(1))+1,:))
ln5=line(lum.x5, lum.y5,'Parent',ax2, 'Color', colorOrder(mod(5,coSize(1))+1,:))
ln6=line(lum.x6, lum.y6,'Parent',ax1, 'Color', colorOrder(mod(6,coSize(1))+1,:))
ln7=line(lum.x7, lum.y7,'Parent',ax2, 'Color', colorOrder(mod(7,coSize(1))+1,:))
ln8=line(lum.x8, lum.y8,'Parent',ax1, 'Color', colorOrder(mod(8,coSize(1))+1,:))
ln9=line(lum.x9, lum.y9,'Parent',ax2, 'Color', colorOrder(mod(9,coSize(1))+1,:))
set(ax1, 'XLim', [1.4 1.65])
set(ax1, 'YLim', [1.75 2.05])
set(ax2, 'XLim', [1.4 1.65])
set(ax2, 'YLim', [6e-05 0.00018])
set(ax1,'XGrid', 'on')
set(ax1,'YGrid', 'on')
legend([ln0;ln1;ln2;ln3;ln4;ln5;ln6;ln7;ln8;ln9],{'line 1','line 2','line 3','line 4','line 5','line 6','line 7','line 8','line 9','line 10'})
