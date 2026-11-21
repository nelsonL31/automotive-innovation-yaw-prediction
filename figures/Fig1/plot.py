"""Plot archived simulation trajectory 3 (zero-based), evaluation origin 0.
Source publication_ai_20261121_planned_steering/simulation_truth_and_controls.npz,
SHA256 5cc9a403b8f06de664ff7129e2bc3092257fb295cf57d80fd4316054873edfd6.
Trace columns correspond to raw[k,origin+j,3] and A/B/C[k,0,j]; sampling interval 0.02 s.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import csv

base=Path(__file__).resolve().parent
with (base/'steering_trace.csv').open(encoding='utf-8-sig') as f:
    records=list(csv.DictReader(f))
def column(name):
    return np.array([float(row[name]) if row[name] else np.nan for row in records])
t=column('time_s')
history,hold,plan,actual=[column(field)*1000 for field in
                         ['history_rad','hold_rad','original_plan_wheel_rad','executed_rad']]
prediction=t>=0
gap=actual[prediction]-plan[prediction]
np.testing.assert_allclose(gap[:6],0)
assert len(gap)==30 and 8.90 < gap[-1] < 8.92
def render(lang='en'):
    zh=lang=='zh'
    def tx(en,cn): return cn if zh else en
    plt.rcParams.update({'font.family':'Microsoft YaHei' if zh else 'Arial','font.size':10.2,
                         'axes.labelsize':10.4,'xtick.labelsize':9.3,'ytick.labelsize':9.3,
                         'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    fig,(a,b)=plt.subplots(1,2,figsize=(7.65,3.35),gridspec_kw={'width_ratios':[1.60,1]})
    fig.subplots_adjust(left=.094,right=.975,wspace=.32,bottom=.18,top=.75)
    for ax in [a,b]:
        ax.spines[['right','top']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#8D9BA4')
        ax.grid(axis='y',color='#E4E9EC',linewidth=.7)
        ax.tick_params(length=3,color='#8D9BA4')
        ax.set_axisbelow(True)
    charcoal='#44474A'; green='#009E73'; blue='#0072B2'; vermilion='#D55E00'
    a.axvspan(-.3,0,color='#EDF2F6',zorder=0)
    a.axvline(0,color='#778690',lw=.9,ls=':')
    artists=[]
    for y,c,ls,label in [
       (history,charcoal,'-',tx('History','已有历史')),
       (hold,green,(0,(6,3)),tx('A  Hold','A  保持')),
       (plan,blue,'-',tx('B  Issued plan','B  起点计划')),
       (actual,vermilion,(0,(2.5,2)),tx('C  Executed','C  实际执行'))]:
        line,=a.plot(t,y,c=c,ls=ls,lw=2.15,label=label)
        artists.append(line)
    a.set_xlim(-.31,.605); a.set_ylim(-62,23)
    a.set_xticks([-.3,0,.2,.4,.6]); a.set_yticks([-60,-40,-20,0,20])
    a.set_xlabel(tx('Time from forecast origin (s)','相对预测起点的时间 (s)'))
    a.set_ylabel(tx('Wheel angle (mrad)','车轮转角 (mrad)'))
    b.axvline(.10,color='#72808D',lw=1,ls=':')
    b.fill_between(t[prediction],0,gap,color='#FBE9DE',alpha=.80)
    b.plot(t[prediction],gap,lw=2.1,color=vermilion,marker='o',mfc='white',mew=1,ms=3.3,markevery=[5,10,15,20,25,29])
    b.text(.112,8.6,tx('First update','首次更新'),fontsize=9,color=charcoal,va='center')
    b.set_xlim(-.015,.605);b.set_ylim(-.34,10);b.set_xticks([0,.2,.4,.6]);b.set_yticks([0,3,6,9])
    b.set_xlabel(tx('Time from forecast origin (s)','相对预测起点的时间 (s)'))
    b.set_ylabel(tx('Executed − planned (mrad)','执行角 − 计划角 (mrad)'))
    for ax,title in [(a,tx('(a)   Steering inputs','(a)   三种转向输入')),(b,tx('(b)   Effect of replanning','(b)   重规划后的转角差'))]:
        ax.set_title(title,x=0,loc='left',fontweight='bold',fontsize=11.8,pad=7)
    fig.legend(handles=artists,loc='upper center',bbox_to_anchor=(.50,1.005),frameon=False,
               ncol=4,fontsize=10.1,handlelength=2.1,columnspacing=1.55,handletextpad=.45)
    for ext in ['png','pdf','svg']:
        fig.savefig(Path.cwd()/f'Fig1.{ext}',dpi=450,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    render()
