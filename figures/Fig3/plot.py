from pathlib import Path
import csv
import numpy as np
from matplotlib import colors, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.font_manager import FontProperties

p=Path(__file__).resolve().parent
data_terracotta='#C25B41'
ode_saffron='#E6B24F'
ink='#352D30'
gray='#655B57'
slow_burgundy='#8F3859'
multisine_amber='#BC752D'
amber_text='#834D16'
grid='#E8DFD5'
rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.family':'Arial',
                 'axes.linewidth':0.8,'axes.labelcolor':ink,'xtick.color':gray,'ytick.color':gray,
                 'font.size':9,'hatch.linewidth':.52,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
with (p/'paired_trajectories.csv').open(encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
with (p/'per_fit_trajectory.csv').open(encoding='utf-8-sig') as f:
    raw=list(csv.DictReader(f))
assert len(rows)==12 and len(raw)==216
for row in rows:
    for k in 'ABC':
        for model in ['data','ode']:
            obs=[float(z['yaw_rmse_rad_s']) for z in raw if all(z[key]==row[key] for key in ('scene_seed','maneuver','mu')) and z['model']==model and z['mode']==k]
            assert len(obs)==3 and abs(np.mean(obs)-float(row[model+'_'+k]))<1e-12
        assert abs((float(row['ode_'+k])-float(row['data_'+k]))-float(row['gap_'+k]))<1e-12
    assert abs(float(row['gap_B'])-float(row['gap_A'])-float(row['gap_B_minus_A']))<1e-12
def mean(field,regime=None):
    return np.mean([float(r[field]) for r in rows if regime is None or r['maneuver']==regime])
def render(lang):
    en=lang=='en'
    normal='Arial' if en else 'Microsoft YaHei'
    rcParams['font.family']=normal
    rcParams['axes.unicode_minus']=True
    def word(e,c): return e if en else c
    def label_axis(ax):
        ax.spines['left'].set_color('#B5A59A');ax.spines['bottom'].set_color('#B5A59A')
        ax.tick_params(width=.7,length=3,pad=3,labelsize=8)
        ax.set_axisbelow(True);ax.grid(axis='y',color=grid,linewidth=.7)
    fig=plt.figure(figsize=(7.2,4.35),facecolor='white')
    a=fig.add_axes([.083,.541,.426,.350])
    b=fig.add_axes([.646,.541,.32,.350])
    heat=fig.add_axes([.106,.121,.86,.170])
    def heading(x,y,code,title):
        fig.text(x,y,code,ha='left',va='center',weight='bold',fontsize=11,color='#352D30')
        fig.text(x+.046,y,title,ha='left',va='center',weight='bold',fontsize=10.1,color=ink)
    heading(.029,.959,'(a)',word('Yaw prediction at 0.6 s','0.6 s 横摆角速度预测'))
    heading(.558,.959,'(b)',word('Tracking the issued plan','计划与执行角的偏差'))
    heading(.029,.388,'(c)',word('Hold-to-plan shift in RMSE gap (ODE-PINN − data only)','从保持转角到采用计划的 RMSE 差值变化（ODE-PINN 减去无物理对照）'))
    x=np.arange(3);bw=.29
    vals0=[mean('data_'+k) for k in 'ABC'];vals1=[mean('ode_'+k) for k in 'ABC']
    label_axis(a);a.set_xlim(-.55,2.55);a.set_ylim(0,.17)
    a.set_yticks([0,.04,.08,.12,.16],['0','0.04','0.08','0.12','0.16'])
    a.set_ylabel(word('Yaw-rate RMSE (rad/s)','横摆角速度 RMSE (rad/s)'),fontsize=8.5)
    a.set_xticks(x,[word('Hold','保持'),word('Plan','计划'),word('Replay','回放')]);a.tick_params(axis='x',length=0)
    for k,vs,offset,color,name in [(0,vals0,-bw*.57,data_terracotta,word('Data only','无物理约束')),(1,vals1,bw*.57,ode_saffron,'ODE-PINN')]:
        hatch_style={'hatch':'//','edgecolor':'#8E5B13','linewidth':.36} if k==1 else {}
        rects=a.bar(x+offset,vs,width=bw,color=color,label=name,zorder=2,**hatch_style)
        for r,v in zip(rects,vs):
            a.annotate(f'{v:.3f}',(r.get_x()+r.get_width()/2,v),xytext=(0,2.2),textcoords='offset points',ha='center',va='bottom',size=7.0,color=ink)
    a.legend(loc='upper right',frameon=False,ncol=2,columnspacing=1,handlelength=.95,handletextpad=.4,borderpad=0,fontsize=7.5, bbox_to_anchor=(1,.999))
    # Logarithmic y preserves the slow steering case while comparing physical angle RMS differences.
    label_axis(b);b.set_xlim(-.38,1.38);b.set_yscale('log')
    b.set_ylim(.009,31);b.set_yticks([.01,.1,1,10],['0.01','0.1','1','10']);b.yaxis.set_minor_locator(NullLocator())
    b.set_ylabel(word('RMS angle error (mrad)','转角差均方根 (mrad)'),fontsize=8.5)
    b.set_xticks([0,1],[word('Issued command','原始计划指令'),word('Actuator-mapped','经执行器映射')]);b.tick_params(axis='x',length=0,labelsize=8)
    for reg,col,label_col,style,mark,tag in [('slow',slow_burgundy,slow_burgundy,'-','o',word('Slow steer','缓变转向')),('varying',multisine_amber,amber_text,(0,(4.5,2.0)),'D',word('Multisine','多正弦转向'))]:
        vs=np.array([mean('issued_command_C_rms_rad',reg),mean('wheel_B_C_rms_rad',reg)])*1000
        b.plot([0,1],vs,marker=mark,color=col,linestyle=style,linewidth=2.0,markersize=5.0,clip_on=False)
        for xx,v in enumerate(vs):
            lab=(f'{v:.3f}' if v<1 else f'{v:.2f}')
            b.annotate(lab,(xx,v),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=7.7,color=label_col,weight='bold')
        xx=.55
        b.annotate(tag,(xx,np.exp(np.interp(xx,[0,1],np.log(vs)))),xytext=(-4,-12),textcoords='offset points',ha='center',va='top',fontsize=7.8,color=label_col)
    heat.set_xlim(0,6.22);heat.set_ylim(2,0);heat.axis('off')
    order=sorted(set(r['scene_seed'] for r in rows));mus=[.9,.45]
    cmap=colors.LinearSegmentedColormap.from_list('signed_shift',['#732C40','#933C47','#B25542','#D67F54','#EDB17A','#FFF0CD'])
    norm=colors.Normalize(-50,0)
    from matplotlib.patches import Rectangle
    for m,mu in enumerate(mus):
        heat.text(-.060,m+.5,f'μ = {mu:.2f}',ha='right',va='center',fontsize=8.3,color=ink)
        for j,reg in enumerate(['slow','varying']):
            for k,phase in enumerate(order):
                row=next(r for r in rows if r['maneuver']==reg and float(r['mu'])==mu and r['scene_seed']==phase)
                v=1000*float(row['gap_B_minus_A']);u=j*3.22+k
                # Use one signed scale across all 12 trajectories, from -50 (dark) to zero.
                assert norm.vmin <= v <= norm.vmax
                rgb=cmap(norm(v))
                heat.add_patch(Rectangle((u+.016,m+.035),.968,.93,fc=rgb,ec='none'))
                srgb=np.asarray(rgb[:3]);linear=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
                luminance=linear@np.array([.2126,.7152,.0722])
                fg='white' if luminance<=1.05/4.5-.05 else '#160F10'
                heat.text(u+.5,m+.49,f'{v:.2f}'.replace('-','−'),ha='center',va='center',fontsize=9.3,weight='bold',color=fg)
    for xcenter,tag,color in [(1.5,word('Slowly varying steer','缓变转向'),slow_burgundy),(4.72,word('Multisine steer','多正弦转向'),amber_text)]:
        heat.text(xcenter,-.10,tag,ha='center',va='bottom',size=8.5,color=color,weight='bold')
    for j in range(2):
        for k in range(3):
            heat.text(j*3.22+k+.5,2.08,word('Phase ','相位 ')+str(k+1),ha='center',va='top',fontsize=7.5,color=gray)
    fig.text(.966,.387,word('mrad/s','mrad/s'),ha='right',va='center',size=8,color=gray)
    # The numerical values and colour key use the same signed measurement.
    fig.text(.44,.026,word('Shared RMSE shift (mrad/s)','共用 RMSE 变化色标 (mrad/s)'),ha='right',va='center',size=7.3,color=gray)
    fig.text(.498,.026,'−50',ha='right',va='center',size=7.3,color=gray)
    swatch=fig.add_axes([.507,.018,.173,.017]);swatch.imshow(np.linspace(-50,0,256)[None,:],aspect='auto',cmap=cmap,norm=norm);swatch.axis('off')
    fig.text(.69,.026,'0',ha='left',va='center',size=7.3,color=gray)
    for ext in ('png','pdf','svg'):
        fig.savefig(Path.cwd()/f'Fig3.{ext}',dpi=450)
    plt.close(fig)
if __name__=='__main__':
    render('en')
