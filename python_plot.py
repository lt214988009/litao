# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:01:25 2017

@author: dell
"""
import ggplot as gp
import matplotlib as mpl
#%matplotlib inline
#如何在图中显示中文
import pandas as pd
meat = gp.meat
p=gp.ggplot(gp.aes(x='date',y='beef'),data=meat)+gp.geom_point(color='red')+gp.ggtitle(u'散点图')
print(p)
p1=gp.ggplot(gp.aes(x='date',y='beef'),data=meat)+gp.geom_line(color='blue')+gp.ggtitle(u'折线图')
print(p1)
p2=gp.ggplot(gp.aes(x='date',y='beef'),data=meat)+gp.geom_point(color='red')+gp.geom_line(color='blue')+gp.ggtitle(u'折线图')
print(p2)
meat_lng = pd.melt(meat[['date','beef','pork','broilers']],id_vars='date')
p3= gp.ggplot(gp.aes(x='date',y='value',colour='variable'),data=meat_lng)+gp.geom_point()+gp.facet_wrap('variable')
print(p3)
meat_lng = pd.melt(meat[['date', 'beef', 'pork', 'broilers']], id_vars='date')
p4=gp.ggplot(gp.aes(x='date', y='value', colour='variable'), data=meat_lng) + gp.geom_point()
print(p4)
p5=gp.ggplot(gp.diamonds, gp.aes(x='price', color='cut')) + gp.geom_density()+gp.ggtitle('Density Plot')
print(p5)
