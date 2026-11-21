from pathlib import Path
import json,torch,math
from official_build_network import build_network
C=json.loads((Path(__file__).parent/'kit_contract.json').read_text())
IP=Path(__file__).parent/'integration_result.json'
SUBSTEPS=json.loads(IP.read_text())['chosen_substeps'] if IP.exists() else 2
def rhs(s,u,p=None):
 v,d=u.unbind(-1);y,w=s.unbind(-1);q=C['physics']
 cf,cr=(q['Cf'],q['Cr']) if p is None else p.unbind(-1)
 den=torch.sqrt(v*v+q['vx_floor']**2);gate=v*v/(v*v+q['speed_gate']**2)
 af=(d-torch.atan2(y+q['lf']*w,den))*gate;ar=-torch.atan2(y-q['lr']*w,den)*gate
 ff=cf*af*torch.cos(d);fr=cr*ar
 return torch.stack(((ff+fr)/q['m']-v*w,(q['lf']*ff-q['lr']*fr)/q['Iz']),-1)
def step(s,u,p=None,substeps=None):
 if substeps is None:substeps=SUBSTEPS
 if substeps is None:raise RuntimeError("Integration gate not passed")
 dt=C['dt']/substeps
 for _ in range(substeps):
  a=rhs(s,u,p);b=rhs(s+dt*a/2,u,p);c=rhs(s+dt*b/2,u,p);d=rhs(s+dt*c,u,p);s=s+dt*(a+2*b+2*c+d)/6
 return s
class Base(torch.nn.Module):
 def __init__(self,norm):
  super().__init__()
  for k in ['mean','std','scale']:self.register_buffer(k,torch.tensor(norm[k],dtype=torch.float32))
 def inputs(self,s,u,h):
  now=torch.cat((u[:,:1],s,u[:,1:]),1)
  return torch.cat(((now-self.mean)/self.std,((h-self.mean)/self.std).flatten(1)),1)
class Flow(Base):
 def __init__(self,norm):
  super().__init__(norm);self.encoder=torch.nn.Sequential(torch.nn.Linear(208,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh());self.decoder=torch.nn.Sequential(torch.nn.Linear(65,64),torch.nn.Tanh(),torch.nn.Linear(64,2))
  torch.nn.init.zeros_(self.decoder[-1].weight);torch.nn.init.zeros_(self.decoder[-1].bias)
 def flow(self,s,u,h,t):
  z=self.encoder(self.inputs(s,u,h));q=t/C['physics_reference_time_s']
  return s+t*self.scale*self.decoder(torch.cat((z,q),1))
 def forward(self,s,u,h):return self.flow(s,u,h,s.new_full((len(s),1),C['dt']))
class PGNN(Base):
 def __init__(self,norm,reference=False):
  super().__init__(norm);self.reference=reference
  if reference:
   cfg={'STATE':['vx','vy','r','delta'],'ACTIONS':[],'MODEL':{'HORIZON':51,'LAYERS':[{'GRU':None,'OUT_FEATURES':2601,'LAYERS':3},{'DENSE':None,'OUT_FEATURES':188,'ACTIVATION':'Mish'},{'DENSE':None,'OUT_FEATURES':188,'ACTIVATION':'Mish'}]}}
   layers=build_network(cfg);self.gru=layers[0];self.dense=torch.nn.Sequential(*layers[1:]);self.head=torch.nn.Linear(192,2)
  else:self.encoder=torch.nn.Sequential(torch.nn.Linear(208,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh());self.head=torch.nn.Linear(64,2)
  lo,hi=C['physics']['bound_Cf_Cr'];z=torch.tensor([(C['physics'][k]-lo)/(hi-lo) for k in ['Cf','Cr']]);torch.nn.init.zeros_(self.head.weight)
  with torch.no_grad():self.head.bias.copy_(torch.logit(z))
 def parameters_for(self,s,u,h):
  if self.reference:
   q,_=self.gru((h-self.mean)/self.std);z=self.dense(q.flatten(1));now=torch.cat((u[:,:1],s,u[:,1:]),1);z=torch.cat((z,(now-self.mean)/self.std),1)
  else:z=self.encoder(self.inputs(s,u,h))
  lo,hi=C['physics']['bound_Cf_Cr'];return lo+(hi-lo)*torch.sigmoid(self.head(z))
 def forward(self,s,u,h):return step(s,u,self.parameters_for(s,u,h))
def factory(arm,norm):return PGNN(norm,arm=='pgnn_ref') if arm.startswith('pgnn') else Flow(norm)
def derivatives(model,s,u,h,t,create_graph=True):
 s=s.detach().requires_grad_(True);t=t.detach().requires_grad_(True);u=u.detach();h=h.detach();value=model.flow(s,u,h,t);dt=[];jac=[]
 for j in range(2):
  gt,gs=torch.autograd.grad(value[:,j].sum(),(t,s),create_graph=create_graph,retain_graph=True);dt.append(gt);jac.append(gs)
 return value,torch.cat(dt,1),torch.stack(jac,1),s

def objective(model,b,arm,coll):
 pred=model(b['s'],b['u'],b['h']);data=(b['w']*((pred-b['y'])/model.scale).square().mean(1)).mean();out={'data':data,'total':data,'prediction':pred}
 if arm in ['ode','pde','joint']:
  k=coll.shape[1];s,u,h=[b[x].repeat_interleave(k,0) for x in ['s','u','h']];value,dt,jac,initial=derivatives(model,s,u,h,coll.reshape(-1,1))
  residual={'ode':dt-rhs(value,u),'pde':dt-torch.einsum('bij,bj->bi',jac,rhs(initial,u))}
  for key,v in residual.items():out[key]=(b['w']*(C['physics_reference_time_s']*v/model.scale).square().reshape(len(pred),k,2).mean((1,2))).mean()
  out['physics']=(out['ode']+out['pde'])/2 if arm=='joint' else out[arm];out['total']=data+C['physics_lambda']*out['physics']
 return out
